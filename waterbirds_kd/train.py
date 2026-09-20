import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from .config import METHODS
from .data import Waterbirds, loader
from .masking import select_tokens
from .metrics import accuracy_metrics
from .models import build_model
from .probe import ProbeLogger
from .utils import (
    autocast,
    compatible_config,
    load_checkpoint,
    metadata_hash,
    save_checkpoint,
    seed_all,
    setup_device,
    sha256,
    write_json,
)


def run_path(cfg, role, method="student"):
    return Path(cfg.output_root) / f"seed_{cfg.seed}" / ("teacher" if role == "teacher" else method)


def weights_for_training(cfg):
    frame = Waterbirds(cfg.data_root, 0).frame
    return [(frame.group == g).mean() for g in range(4)]


@torch.inference_mode()
def evaluate(model, dataset, cfg, device, weights):
    model.eval()
    logits, labels, groups = [], [], []
    for batch in loader(dataset, cfg):
        with autocast(cfg, device):
            output = model(batch["image"].to(device, non_blocking=True))
        logits.append(output.float().cpu().numpy())
        labels.append(batch["label"].numpy())
        groups.append(batch["group"].numpy())
    return accuracy_metrics(np.concatenate(logits), np.concatenate(labels), np.concatenate(groups), weights)


def learning_rate(cfg, epoch, total_epochs, role):
    # Official student LR is scaled by effective global batch / 512.
    base = cfg.teacher_lr if role == "teacher" else cfg.lr * cfg.batch_size * cfg.accumulation_steps / 512
    warmup = min(cfg.warmup_epochs, max(0, total_epochs - 1))
    if epoch <= warmup:
        return base * epoch / max(1, warmup)
    fraction = (epoch - warmup - 1) / max(1, total_epochs - warmup - 1)
    floor = min(cfg.min_lr, base)
    return floor + (base - floor) * (1 + math.cos(math.pi * fraction)) / 2


def optimizer_for(model, cfg):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        target = no_decay if parameter.ndim == 1 or name.endswith(".bias") or name in ("pos_embed", "cls_token") else decay
        target.append(parameter)
    return torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr)


def distillation_loss(logits, labels, teacher_logits, cfg):
    ce = F.cross_entropy(logits.float(), labels, label_smoothing=cfg.label_smoothing)
    if teacher_logits is None:
        return ce, ce.detach(), logits.new_zeros(())
    temperature = cfg.temperature
    kd = F.kl_div(F.log_softmax(logits.float() / temperature, dim=1),
                  F.softmax(teacher_logits.float() / temperature, dim=1),
                  reduction="batchmean") * temperature**2
    return (1 - cfg.kd_alpha) * ce + cfg.kd_alpha * kd, ce.detach(), kd.detach()


def train_epoch(model, teacher, dataset, optimizer, scaler, cfg, device, method, epoch):
    model.train()
    if teacher is not None:
        teacher.eval()
    # Reset per epoch: resume reproduces augmentation/dropout without persisting workers.
    seed_all(cfg.seed + epoch * 100003)
    mask_rng = torch.Generator(device=device).manual_seed(cfg.seed + epoch * 200003)
    batches = loader(dataset, cfg, train=True, epoch=epoch)
    total_batches = min(len(batches), cfg.max_train_batches or len(batches))
    optimizer.zero_grad(set_to_none=True)
    sums = {"loss": 0.0, "ce": 0.0, "kd": 0.0, "swaps": 0.0}
    seen = 0
    optimizer_updates, skipped_updates = 0, 0
    started = time.monotonic()
    for step, batch in enumerate(tqdm(batches, total=total_batches, desc=f"{method} epoch {epoch}", leave=False)):
        if step >= total_batches:
            break
        x = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        foreground = batch["foreground"].to(device, non_blocking=True)
        with autocast(cfg, device):
            logits, attention = model(x, return_attention=True)
            teacher_logits, swaps = None, None
            if teacher is not None and method not in ("ce", "teacher"):
                indices, swaps = select_tokens(attention, method, cfg.keep_tokens, foreground, mask_rng)
                with torch.no_grad():
                    teacher_logits = teacher(x, indices)
            loss, ce, kd = distillation_loss(logits, labels, teacher_logits, cfg)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch={epoch}, step={step}")
        # Weight uneven final micro-batches by their true number of samples.
        window_start = (step // cfg.accumulation_steps) * cfg.accumulation_steps
        window_end = min(window_start + cfg.accumulation_steps, total_batches)
        window_samples = min(window_end * cfg.batch_size, len(dataset)) - window_start * cfg.batch_size
        scaler.scale(loss * len(x) / window_samples).backward()
        if step + 1 == window_end:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=not scaler.is_enabled())
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < previous_scale:
                skipped_updates += 1
            else:
                optimizer_updates += 1
            optimizer.zero_grad(set_to_none=True)
        for name, value in (("loss", loss), ("ce", ce), ("kd", kd)):
            sums[name] += value.item() * len(x)
        sums["swaps"] += swaps.sum().item() if swaps is not None else 0
        seen += len(x)
    return {**{key: value / seen for key, value in sums.items()}, "samples": seen,
            "optimizer_updates": optimizer_updates, "amp_skipped_updates": skipped_updates,
            "seconds": time.monotonic() - started}


def train(cfg, role="student", method="student", teacher_checkpoint=None):
    if role not in ("student", "teacher") or (role == "student" and method not in METHODS):
        raise ValueError("Unknown role/method")
    method = "teacher" if role == "teacher" else method
    if "rescue" in method and not cfg.seg_root:
        raise ValueError("Rescue training requires seg_root")
    device = setup_device(cfg)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    directory = run_path(cfg, role, method)
    directory.mkdir(parents=True, exist_ok=True)
    current_hash = metadata_hash(cfg)
    if (directory / "config.json").exists():
        compatible_config(json.loads((directory / "config.json").read_text()), cfg.to_dict())
    if role == "student" and method != "ce":
        teacher_checkpoint = Path(teacher_checkpoint or run_path(cfg, "teacher") / "best.pt")
        if not teacher_checkpoint.exists():
            raise FileNotFoundError(f"Train the teacher first: {teacher_checkpoint}")
    teacher_hash = sha256(teacher_checkpoint) if teacher_checkpoint else None
    if (directory / "result.json").exists():
        result = json.loads((directory / "result.json").read_text())
        if result["metadata_sha256"] != current_hash or result["teacher_sha256"] != teacher_hash:
            raise ValueError("Completed run uses different data/teacher. Choose a new output_root.")
        print(f"Already complete: {directory}", flush=True)
        return directory / "best.pt"
    write_json(directory / "config.json", cfg.to_dict())
    last_path = directory / "last.pt"
    state = load_checkpoint(last_path) if last_path.exists() else None
    if state and (state["metadata_sha256"] != current_hash or state["teacher_sha256"] != teacher_hash):
        raise ValueError("Resume data/teacher fingerprint mismatch")
    seed_all(cfg.seed)
    pretrained = (cfg.teacher_pretrained if role == "teacher" else cfg.student_init == "imagenet") and state is None
    model = build_model(role, cfg, pretrained=pretrained).to(device)
    teacher = None
    if teacher_checkpoint:
        teacher_state = load_checkpoint(teacher_checkpoint)
        if teacher_state["role"] != "teacher" or teacher_state["metadata_sha256"] != current_hash:
            raise ValueError("Teacher checkpoint has wrong role or metadata fingerprint")
        if teacher_state["config"]["model_scale"] != cfg.model_scale:
            raise ValueError("Teacher/student model_scale mismatch")
        teacher = build_model("teacher", cfg).to(device)
        teacher.load_state_dict(teacher_state["model"])
        teacher.requires_grad_(False).eval()
    optimizer = optimizer_for(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    start_epoch, best_wga, best_epoch, history = 0, -1.0, None, []
    if state:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best_wga, best_epoch = state["epoch"], state["best_wga"], state["best_epoch"]
        history = state["history"]
        print(f"Resuming {directory} after epoch {start_epoch}", flush=True)
        del state
    # Baselines also receive masks for probes; training masks are read only for rescue.
    train_data = Waterbirds(cfg.data_root, 0, cfg.seg_root if "rescue" in method else None,
                           train=True, threshold=cfg.foreground_threshold)
    validation = Waterbirds(cfg.data_root, 1)
    test_data = Waterbirds(cfg.data_root, 2)
    weights = weights_for_training(cfg)
    epochs = cfg.teacher_epochs if role == "teacher" else cfg.epochs
    probe = ProbeLogger(cfg, directory, teacher, device) if role == "student" else None
    if probe and start_epoch == 0:
        probe.log(model, 0, epochs)
    started = time.monotonic()
    for epoch in range(start_epoch + 1, epochs + 1):
        lr = learning_rate(cfg, epoch, epochs, role)
        for group in optimizer.param_groups:
            group["lr"] = lr
        training = train_epoch(model, teacher, train_data, optimizer, scaler, cfg, device, method, epoch)
        validation_metrics = evaluate(model, validation, cfg, device, weights)
        if validation_metrics["worst_group_accuracy"] is None:
            raise ValueError("Validation must contain all four groups")
        history.append({"epoch": epoch, "lr": lr, "train": training, "validation": validation_metrics})
        if probe:
            probe.log(model, epoch, epochs)
        payload = {"model": model.state_dict(), "config": cfg.to_dict(), "role": role, "method": method,
                   "epoch": epoch, "metadata_sha256": current_hash, "teacher_sha256": teacher_hash,
                   "train_group_weights": weights}
        score = validation_metrics["worst_group_accuracy"]
        if score > best_wga:
            best_wga, best_epoch = score, epoch
            save_checkpoint(directory / "best.pt", {**payload, "validation": validation_metrics})
        save_checkpoint(last_path, {**payload, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                                    "best_wga": best_wga, "best_epoch": best_epoch, "history": history})
        write_json(directory / "history.json", history)
        peak = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0
        print(f"{method} {epoch}/{epochs}: loss={training['loss']:.4f}, val WGA={score:.4f}, "
              f"lr={lr:.3g}, epoch={training['seconds']:.1f}s, peak GPU={peak:.2f}GiB", flush=True)
    best = load_checkpoint(directory / "best.pt")
    model.load_state_dict(best["model"])
    result = {"method": method, "seed": cfg.seed, "best_epoch": best_epoch,
              "validation": best["validation"], "test": evaluate(model, test_data, cfg, device, weights),
              "metadata_sha256": current_hash, "teacher_sha256": teacher_hash,
              "train_group_weights": weights, "model_scale": cfg.model_scale,
              "student_init": cfg.student_init, "keep_tokens": cfg.keep_tokens,
              "partial_training": cfg.max_train_batches is not None,
              "seconds_this_invocation": time.monotonic() - started,
              "checkpoint_selection": "maximum validation WGA; earliest epoch breaks ties"}
    write_json(directory / "result.json", result)
    print(f"Complete {directory}: test WGA={result['test']['worst_group_accuracy']:.4f}", flush=True)
    return directory / "best.pt"
