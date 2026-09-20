import importlib
from dataclasses import replace

import pytest
import torch

from waterbirds_kd.config import Config
from waterbirds_kd.utils import load_checkpoint


def test_interrupted_student_resumes_to_identical_parameters(synthetic, tmp_path, monkeypatch):
    module = importlib.import_module("waterbirds_kd.train")
    data, masks = synthetic
    cfg = Config(data_root=data, seg_root=masks, output_root=str(tmp_path / "uninterrupted"),
                 device="cpu", model_scale="debug", teacher_pretrained=False,
                 epochs=2, teacher_epochs=1, num_workers=0, batch_size=2, eval_batch_size=4,
                 accumulation_steps=2, max_train_batches=2, probe_per_group=1,
                 diagnostic_epochs=[0, 1, 2], num_threads=2)
    teacher = module.train(cfg, role="teacher")
    complete = module.train(cfg, method="foreground_rescue_10", teacher_checkpoint=teacher)
    resumed_cfg = replace(cfg, output_root=str(tmp_path / "interrupted"))
    original = module.train_epoch

    def interrupt_on_second_epoch(*args, **kwargs):
        epoch = args[-1]
        if epoch == 2:
            raise RuntimeError("simulated interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "train_epoch", interrupt_on_second_epoch)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        module.train(resumed_cfg, method="foreground_rescue_10", teacher_checkpoint=teacher)
    interrupted_last = module.run_path(resumed_cfg, "student", "foreground_rescue_10") / "last.pt"
    assert load_checkpoint(interrupted_last)["epoch"] == 1
    monkeypatch.setattr(module, "train_epoch", original)
    resumed = module.train(resumed_cfg, method="foreground_rescue_10", teacher_checkpoint=teacher)
    a, b = load_checkpoint(complete.parent / "last.pt"), load_checkpoint(resumed.parent / "last.pt")
    assert a["epoch"] == b["epoch"] == 2
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    # A completed run returns its checkpoint rather than training again.
    monkeypatch.setattr(module, "train_epoch", lambda *a, **kw: pytest.fail("Completed run should be skipped"))
    assert module.train(resumed_cfg, method="foreground_rescue_10", teacher_checkpoint=teacher) == resumed


def test_resume_rejects_changed_hyperparameters(synthetic, tmp_path):
    module = importlib.import_module("waterbirds_kd.train")
    data, masks = synthetic
    cfg = Config(data_root=data, seg_root=masks, output_root=str(tmp_path / "run"), device="cpu",
                 model_scale="debug", teacher_pretrained=False, teacher_epochs=1,
                 num_workers=0, batch_size=2, eval_batch_size=4, max_train_batches=1, num_threads=2)
    module.train(cfg, role="teacher")
    with pytest.raises(ValueError, match="different settings"):
        module.train(replace(cfg, batch_size=4), role="teacher")
