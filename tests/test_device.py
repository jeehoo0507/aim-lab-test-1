"""Exercise CUDA argument validation without requiring a local GPU."""
import pytest
import torch
from torch.cuda._utils import _get_device_index

from waterbirds_kd.config import Config
from waterbirds_kd.utils import setup_device


@pytest.mark.parametrize("requested,current,expected", [("cuda", 0, 0), ("cuda", 2, 2), ("cuda:1", 0, 1)])
def test_cuda_device_is_resolved_before_set_device(monkeypatch, requested, current, expected):
    selected = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current)
    # Use the actual PyTorch argument validation from cuda.set_device. The GPU
    # driver call is omitted so the reported server failure is testable on CPU.
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: selected.append(_get_device_index(device)))
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", torch.backends.cudnn.benchmark)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", torch.backends.cudnn.deterministic)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", torch.backends.cuda.matmul.allow_tf32)
    device = setup_device(Config(device=requested))
    assert device == torch.device("cuda", expected)
    assert selected == [expected]


def test_cpu_does_not_select_cuda_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: pytest.fail("CPU execution must not select a CUDA device"))
    assert setup_device(Config(device="cpu")) == torch.device("cpu")


def test_unavailable_cuda_has_actionable_error(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: pytest.fail("Unavailable CUDA must not be selected"))
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        setup_device(Config(device="cuda"))
