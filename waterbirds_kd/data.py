import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode, RandomResizedCrop
from torchvision.transforms import functional as TF


class Waterbirds(Dataset):
    def __init__(self, root, split, seg_root=None, train=False, threshold=0.5, indices=None):
        self.root = Path(root).expanduser()
        self.seg_root = Path(seg_root).expanduser() if seg_root else None
        metadata = pd.read_csv(self.root / "metadata.csv")
        required = {"img_filename", "y", "place", "split"}
        if not required.issubset(metadata.columns):
            raise ValueError(f"Missing metadata columns: {required - set(metadata.columns)}")
        for column, allowed in (("y", {0, 1}), ("place", {0, 1}), ("split", {0, 1, 2})):
            if not set(metadata[column].unique()).issubset(allowed):
                raise ValueError(f"Invalid {column} values in metadata.csv")
        if metadata.img_filename.duplicated().any():
            raise ValueError("Duplicate image paths across metadata rows/splits")
        metadata["sample_id"] = np.arange(len(metadata))
        frame = metadata.loc[metadata.split == split].copy()
        frame["group"] = 2 * frame.y + frame.place
        if indices is not None:
            frame = frame.set_index("sample_id").loc[list(indices)].reset_index()
        self.frame = frame.reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"Empty split: {split}")
        self.train = train
        self.threshold = threshold

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        rel = Path(row.img_filename)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"Expected a relative image path: {rel}")
        with Image.open(self.root / rel) as file:
            image = file.convert("RGB")
        mask = None
        if self.seg_root:
            mask_path = self.seg_root / rel.with_suffix(".png")
            with Image.open(mask_path) as file:
                mask = file.convert("L")
            if mask.size != image.size:
                raise ValueError(f"Image/segmentation size mismatch: {rel}: {image.size} vs {mask.size}")
        if self.train:
            crop = RandomResizedCrop.get_params(image, scale=(0.7, 1.0), ratio=(0.75, 4 / 3))
            image = TF.resized_crop(image, *crop, [224, 224], InterpolationMode.BICUBIC, antialias=True)
            if mask is not None:
                mask = TF.resized_crop(mask, *crop, [224, 224], InterpolationMode.NEAREST)
            if torch.rand(()).item() < 0.5:
                image = TF.hflip(image)
                if mask is not None:
                    mask = TF.hflip(mask)
        else:
            image = TF.center_crop(TF.resize(image, 256, InterpolationMode.BICUBIC, antialias=True), 224)
            if mask is not None:
                mask = TF.center_crop(TF.resize(mask, 256, InterpolationMode.NEAREST), 224)
        image = TF.normalize(TF.to_tensor(image), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if mask is not None:
            segmentation = TF.to_tensor(mask)
            coverage = torch.nn.functional.avg_pool2d(segmentation, 16, 16).flatten()
            foreground = coverage >= self.threshold
        else:
            coverage = torch.full((196,), float("nan"))
            foreground = torch.zeros(196, dtype=torch.bool)
        return {"image": image, "label": int(row.y), "place": int(row.place),
                "group": int(row.group), "sample_id": int(row.sample_id),
                "foreground": foreground, "coverage": coverage,
                "has_segmentation": mask is not None}


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def loader(dataset, cfg, train=False, epoch=0):
    generator = torch.Generator().manual_seed(cfg.seed + epoch * 100003)
    return DataLoader(dataset, batch_size=cfg.batch_size if train else cfg.eval_batch_size,
                      shuffle=train, num_workers=cfg.num_workers, pin_memory=cfg.device.startswith("cuda"),
                      drop_last=False, worker_init_fn=seed_worker, generator=generator,
                      persistent_workers=False)


def probe_ids(dataset, per_group, seed):
    rng = np.random.default_rng(seed)
    chosen = []
    for group in range(4):
        ids = dataset.frame.loc[dataset.frame.group == group, "sample_id"].to_numpy()
        if len(ids) < per_group:
            raise ValueError(f"Validation group {group} has {len(ids)} samples, needs {per_group}")
        chosen.extend(rng.choice(ids, per_group, replace=False).tolist())
    return sorted(chosen)
