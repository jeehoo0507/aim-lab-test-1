import platform
import shutil
from pathlib import Path

import torch
from PIL import Image

from .config import GROUP_NAMES
from .data import Waterbirds, probe_ids
from .utils import metadata_hash, setup_device, write_json


def preflight(cfg, full_image_check=False):
    device = setup_device(cfg)
    report = {"python": platform.python_version(), "torch": torch.__version__,
              "device": str(device), "metadata_sha256": metadata_hash(cfg),
              "effective_batch_size": cfg.batch_size * cfg.accumulation_steps,
              "segmentation_available": bool(cfg.seg_root), "splits": {}}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        report["gpu"] = {"name": props.name, "total_gib": props.total_memory / 1024**3,
                         "free_gib": torch.cuda.mem_get_info(device)[0] / 1024**3}
    for split, name in enumerate(("train", "validation", "test")):
        dataset = Waterbirds(cfg.data_root, split, cfg.seg_root or None, threshold=cfg.foreground_threshold)
        groups = {group_name: int((dataset.frame.group == g).sum()) for g, group_name in enumerate(GROUP_NAMES)}
        if min(groups.values()) == 0:
            raise ValueError(f"Missing group in {name}: {groups}")
        for i, row in dataset.frame.iterrows():
            rel = Path(row.img_filename)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"Invalid image path: {rel}")
            image_path = Path(cfg.data_root) / rel
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            if cfg.seg_root:
                mask_path = Path(cfg.seg_root) / rel.with_suffix(".png")
                if not mask_path.is_file():
                    raise FileNotFoundError(f"Missing CUB segmentation: {mask_path}")
            if full_image_check:
                with Image.open(image_path) as im:
                    size = im.size
                    im.verify()
                if cfg.seg_root:
                    with Image.open(mask_path) as im:
                        if im.size != size:
                            raise ValueError(f"Image/mask size mismatch: {rel}")
                        im.verify()
        check_indices = dataset.frame.groupby("group", sort=True).head(2).index
        for i in check_indices:
            item = dataset[i]
            assert item["image"].shape == (3, 224, 224)
            assert item["foreground"].shape == (196,)
        if split == 1:
            probe_ids(dataset, cfg.probe_per_group, cfg.seed)
        report["splits"][name] = {"count": len(dataset), "groups": groups}
    root = Path(cfg.output_root)
    root.mkdir(parents=True, exist_ok=True)
    report["disk_free_gib"] = shutil.disk_usage(root).free / 1024**3
    report["image_integrity_check"] = "all images" if full_image_check else "all paths; two transformed examples per group/split"
    write_json(root / "preflight.json", report)
    return report
