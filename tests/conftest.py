import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch

from waterbirds_kd.synthetic import create_synthetic


@pytest.fixture(autouse=True)
def limit_cpu_threads():
    torch.set_num_threads(2)


@pytest.fixture
def synthetic(tmp_path):
    return create_synthetic(tmp_path / "fixture")
