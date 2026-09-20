import numpy as np
import pytest
import torch

from waterbirds_kd.analysis import row_spearman
from waterbirds_kd.masking import binary_mask, select_tokens
from waterbirds_kd.metrics import accuracy_metrics, degradation, selection_metrics


@pytest.mark.parametrize("method", ["student", "random", "foreground_rescue_5", "foreground_rescue_10", "foreground_rescue_20", "random_rescue_10"])
def test_token_budget_and_rescue_invariants(method):
    attention = torch.arange(196).float().repeat(3, 1)
    foreground = torch.zeros(3, 196, dtype=torch.bool)
    foreground[0, :20] = True
    foreground[1, :3] = True
    foreground[2, 190:] = True  # Nothing to rescue.
    base, _ = select_tokens(attention, "student", 98)
    chosen, swaps = select_tokens(attention, method, 98, foreground, torch.Generator().manual_seed(10))
    assert chosen.shape == (3, 98)
    for row in chosen:
        assert row.unique().numel() == 98
        assert row.min() >= 0 and row.max() < 196
    mask, base_mask = binary_mask(chosen), binary_mask(base)
    if "rescue" in method:
        assert torch.equal((mask & ~base_mask).sum(1), swaps)
        assert torch.equal((base_mask & ~mask).sum(1), swaps)
        assert swaps[1] == 3 and swaps[2] == 0
    if method.startswith("foreground"):
        assert torch.equal(((mask & ~base_mask) & foreground).sum(1), swaps)
        assert not ((base_mask & ~mask) & foreground).any()


def test_random_rescue_and_foreground_rescue_have_same_feasible_count():
    attention = torch.rand(8, 196)
    foreground = torch.rand(8, 196) > 0.97
    _, foreground_swaps = select_tokens(attention, "foreground_rescue_10", 98, foreground)
    _, random_swaps = select_tokens(attention, "random_rescue_10", 98, foreground)
    assert torch.equal(foreground_swaps, random_swaps)


def test_degradation_kl_direction_and_probability_drop():
    full = torch.tensor([[2.0, -1.0], [-2.0, 1.0]])
    masked = torch.tensor([[0.0, 0.0], [2.0, -1.0]])
    labels = torch.tensor([0, 1])
    result = degradation(full, masked, labels)
    p, q = full.softmax(1), masked.softmax(1)
    expected = (p * (p.log() - q.log())).sum(1)
    torch.testing.assert_close(result["kl"], expected)
    torch.testing.assert_close(result["gt_probability_drop"], p[range(2), labels] - q[range(2), labels])
    assert result["prediction_flip"].tolist() == [0, 1]
    assert torch.equal(degradation(full, full, labels)["kl"], torch.zeros(2))


def test_foreground_missing_zero_denominator_is_nan():
    foreground = torch.zeros(2, 196, dtype=torch.bool)
    foreground[1, :4] = True
    mask = torch.zeros_like(foreground)
    mask[:, :2] = True
    metrics = selection_metrics(mask, foreground, foreground.float())
    assert torch.isnan(metrics["foreground_missing_ratio"][0])
    assert metrics["foreground_missing_ratio"][1] == 0.5
    assert metrics["foreground_selection_ratio"][1] == 1


def test_group_metrics_weighting_and_missing_groups():
    labels = np.array([0, 0, 1, 1, 1])
    groups = np.array([0, 1, 2, 3, 3])
    logits = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [1, 0]])
    result = accuracy_metrics(logits, labels, groups, [0.4, 0.1, 0.1, 0.4])
    assert result["accuracy"] == 0.6
    assert result["worst_group_accuracy"] == 0
    assert result["train_weighted_accuracy"] == pytest.approx(0.7)
    assert accuracy_metrics(logits[:2], labels[:2], groups[:2], [0.25] * 4)["worst_group_accuracy"] is None


def test_spearman_ties_and_constant_rows():
    a = np.array([[1, 2, 2, 3], [2, 2, 2, 2]])
    b = np.array([[3, 2, 2, 1], [1, 2, 3, 4]])
    values = row_spearman(a, b)
    assert values[0] == -1
    assert np.isnan(values[1])
