import copy

import pytest
import torch
from PIL import Image

from waterbirds_kd.config import Config
from waterbirds_kd.data import Waterbirds, probe_ids
from waterbirds_kd.models import DeiT
from waterbirds_kd.train import distillation_loss, train_epoch


def test_data_splits_probe_and_segmentation(synthetic):
    data, masks = synthetic
    train, val, test = [Waterbirds(data, split, masks) for split in range(3)]
    assert set(train.frame.sample_id).isdisjoint(val.frame.sample_id)
    assert set(test.frame.sample_id).isdisjoint(val.frame.sample_id)
    chosen = probe_ids(val, 2, 3)
    assert chosen == probe_ids(val, 2, 3)
    item = train[0]
    assert item["image"].shape == (3, 224, 224)
    assert item["foreground"].sum() > 0
    assert item["coverage"].max() <= 1
    with pytest.raises(ValueError, match="needs"):
        probe_ids(val, 3, 0)


def test_mask_size_mismatch_rejected(synthetic):
    from pathlib import Path
    data, masks = synthetic
    dataset = Waterbirds(data, 0, masks)
    path = Path(masks) / Path(dataset.frame.iloc[0].img_filename).with_suffix(".png")
    Image.new("L", (10, 10)).save(path)
    with pytest.raises(ValueError, match="size mismatch"):
        dataset[0]


def test_training_geometry_keeps_foreground_and_image_aligned(tmp_path):
    import numpy as np
    import pandas as pd
    root, masks = tmp_path / "images", tmp_path / "masks"
    root.mkdir()
    masks.mkdir()
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[30:210, 70:190] = 255
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[:, :, 0] = mask
    Image.fromarray(image).save(root / "bird.png")
    Image.fromarray(mask).save(masks / "bird.png")
    pd.DataFrame([{"img_filename": "bird.png", "y": 0, "place": 0, "split": 0}]).to_csv(root / "metadata.csv", index=False)
    data = Waterbirds(root, 0, masks, train=True)
    for seed in range(4):
        torch.manual_seed(seed)
        item = data[0]
        red_pixels = item["image"][0:1] * 0.229 + 0.485
        red_coverage = torch.nn.functional.avg_pool2d(red_pixels, 16, 16).flatten()
        assert (red_coverage - item["coverage"]).abs().mean() < 0.01


def test_ce_loss_is_not_scaled_by_kd_alpha():
    logits = torch.randn(4, 2, requires_grad=True)
    labels = torch.tensor([0, 1, 1, 0])
    cfg = Config(kd_alpha=0.99)
    actual, _, kd = distillation_loss(logits, labels, None, cfg)
    expected = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=cfg.label_smoothing)
    torch.testing.assert_close(actual, expected)
    assert kd == 0


def test_ce_training_never_calls_teacher(synthetic):
    class ForbiddenTeacher(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError("CE training must not call teacher")
    data, _ = synthetic
    dataset = Waterbirds(data, 0, train=True)
    model = DeiT(12, 1, 3)
    cfg = Config(device="cpu", batch_size=2, accumulation_steps=2, num_workers=0, max_train_batches=2)
    opt = torch.optim.SGD(model.parameters(), lr=0.001)
    result = train_epoch(model, ForbiddenTeacher(), dataset, opt, torch.amp.GradScaler("cuda", enabled=False),
                         cfg, torch.device("cpu"), "ce", 1)
    assert result["kd"] == 0 and result["samples"] == 4


def test_gradient_accumulation_handles_uneven_last_microbatch():
    class FixedDataset(torch.utils.data.Dataset):
        def __init__(self):
            rng = torch.Generator().manual_seed(123)
            self.x = torch.randn(5, 3, 224, 224, generator=rng)
        def __len__(self):
            return 5
        def __getitem__(self, i):
            return {"image": self.x[i], "label": i % 2, "foreground": torch.zeros(196, dtype=torch.bool)}
    torch.manual_seed(14)
    large = DeiT(12, 1, 3, drop_path=0)
    small = copy.deepcopy(large)
    data = FixedDataset()
    for model, batch, accumulation in [(large, 5, 1), (small, 2, 3)]:
        cfg = Config(device="cpu", batch_size=batch, accumulation_steps=accumulation, num_workers=0, grad_clip=100)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        train_epoch(model, None, data, optimizer, torch.amp.GradScaler("cuda", enabled=False), cfg,
                    torch.device("cpu"), "ce", 1)
    for a, b in zip(large.parameters(), small.parameters()):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=2e-6)
