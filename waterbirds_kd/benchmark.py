"""Short real-architecture memory check; random tensors, no saved training run."""
import gc
import time

import torch

from .models import build_model
from .train import distillation_loss, optimizer_for
from .utils import autocast, seed_all, setup_device


def benchmark(cfg, steps=3):
    device = setup_device(cfg)
    results = []
    for case in ("teacher_training", "student_full_kd", "student_masked_kd"):
        seed_all(cfg.seed)
        teacher_training = case == "teacher_training"
        model = build_model("teacher" if teacher_training else "student", cfg).to(device).train()
        teacher = None if teacher_training else build_model("teacher", cfg).to(device).eval().requires_grad_(False)
        optimizer = optimizer_for(model, cfg)
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
        x = torch.randn(cfg.batch_size, 3, 224, 224, device=device)
        y = torch.arange(cfg.batch_size, device=device) % 2
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.monotonic()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            with autocast(cfg, device):
                logits, attention = model(x, return_attention=True)
                target = None
                if teacher is not None:
                    indices = attention.detach().topk(cfg.keep_tokens, 1).indices if case == "student_masked_kd" else None
                    with torch.no_grad():
                        target = teacher(x, indices)
                loss, _, _ = distillation_loss(logits, y, target, cfg)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        result = {"case": case, "device": str(device), "batch_size": cfg.batch_size,
                  "steps": steps, "seconds_per_step": (time.monotonic() - started) / steps}
        if device.type == "cuda":
            result.update(peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024**3,
                          peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 1024**3)
        results.append(result)
        del model, teacher, optimizer, scaler, x, y, logits, attention, loss, target
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results
