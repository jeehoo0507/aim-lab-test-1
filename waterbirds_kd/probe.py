import os
from pathlib import Path

import numpy as np
import torch

from .data import Waterbirds, loader, probe_ids
from .masking import binary_mask, select_tokens
from .utils import autocast, write_json


class ProbeLogger:
    def __init__(self, cfg, run_dir, teacher, device):
        self.cfg, self.teacher, self.device = cfg, teacher, device
        validation = Waterbirds(cfg.data_root, 1)
        self.ids = probe_ids(validation, cfg.probe_per_group, cfg.seed)
        self.dataset = Waterbirds(cfg.data_root, 1, cfg.seg_root or None,
                                  threshold=cfg.foreground_threshold, indices=self.ids)
        self.path = Path(run_dir) / "probe"
        self.path.mkdir(parents=True, exist_ok=True)
        write_json(self.path / "manifest.json", {"sample_ids": self.ids, "split": "validation",
                                                "seed": cfg.seed, "transform": "resize256_center224"})
        self.full_logits = None

    @torch.inference_mode()
    def log(self, model, epoch, final_epoch):
        model.eval()
        cfg, device = self.cfg, self.device
        batches = []
        diagnostics = self.teacher is not None and (epoch in cfg.diagnostic_epochs or epoch == final_epoch)
        methods = ["student", "random"]
        if cfg.seg_root:
            methods += [f"{prefix}_{k}" for k in (5, 10, 20)
                        for prefix in ("foreground_rescue", "random_rescue")]
        # Per-method RNG streams are identical across epochs for paired comparisons.
        generators = {method: torch.Generator(device=device).manual_seed(cfg.seed + 90001 + i)
                      for i, method in enumerate(methods)}
        offset = 0
        for batch in loader(self.dataset, cfg):
            x = batch["image"].to(device)
            foreground = batch["foreground"].to(device)
            with autocast(cfg, device):
                logits, attention = model(x, return_attention=True)
            indices, _ = select_tokens(attention, "student", cfg.keep_tokens)
            record = {k: batch[k].numpy() for k in ("sample_id", "label", "place", "group", "foreground", "coverage")}
            record.update(attention=attention.float().cpu().numpy(),
                          selected_indices=indices.cpu().numpy(),
                          selected_mask=binary_mask(indices).cpu().numpy(),
                          student_logits=logits.float().cpu().numpy(),
                          student_probabilities=logits.float().softmax(1).cpu().numpy())
            if diagnostics:
                if self.full_logits is None:
                    with autocast(cfg, device):
                        full = self.teacher(x).float().cpu().numpy()
                else:
                    full = self.full_logits[offset:offset + len(x)]
                record["teacher_full_logits"] = full
                for method in methods:
                    chosen, swaps = select_tokens(attention, method, cfg.keep_tokens, foreground, generators[method])
                    with autocast(cfg, device):
                        masked = self.teacher(x, chosen)
                    record[f"teacher_logits__{method}"] = masked.float().cpu().numpy()
                    record[f"indices__{method}"] = chosen.cpu().numpy()
                    record[f"swaps__{method}"] = swaps.cpu().numpy()
            batches.append(record)
            offset += len(x)
        result = {key: np.concatenate([record[key] for record in batches]) for key in batches[0]}
        if diagnostics and self.full_logits is None:
            self.full_logits = result["teacher_full_logits"]
        result["epoch"] = np.array(epoch)
        result["has_segmentation"] = np.array(bool(cfg.seg_root))
        path = self.path / f"epoch_{epoch:03d}.npz"
        temp = path.with_suffix(".tmp")
        with open(temp, "wb") as file:
            np.savez_compressed(file, **result)
        os.replace(temp, path)
