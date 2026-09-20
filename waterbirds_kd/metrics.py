import numpy as np
from torch.nn import functional as F

from .config import GROUP_NAMES


def accuracy_metrics(logits, labels, groups, train_group_weights):
    pred = np.asarray(logits).argmax(axis=1)
    labels, groups = np.asarray(labels), np.asarray(groups)
    correct = pred == labels
    group_acc, counts = {}, {}
    for g, name in enumerate(GROUP_NAMES):
        selected = groups == g
        counts[name] = int(selected.sum())
        group_acc[name] = float(correct[selected].mean()) if selected.any() else None
    valid = all(value is not None for value in group_acc.values())
    return {"accuracy": float(correct.mean()), "group_accuracy": group_acc, "group_counts": counts,
            "worst_group_accuracy": min(group_acc.values()) if valid else None,
            "train_weighted_accuracy": float(sum(train_group_weights[g] * group_acc[name]
                                                for g, name in enumerate(GROUP_NAMES))) if valid else None}


def degradation(full_logits, masked_logits, labels):
    full_log = F.log_softmax(full_logits.float(), dim=1)
    masked_log = F.log_softmax(masked_logits.float(), dim=1)
    full_p, masked_p = full_log.exp(), masked_log.exp()
    return {"kl": (full_p * (full_log - masked_log)).sum(1).clamp_min(0),
            "prediction_flip": (full_p.argmax(1) != masked_p.argmax(1)).float(),
            "gt_probability_drop": full_p.gather(1, labels[:, None]).squeeze(1)
                                     - masked_p.gather(1, labels[:, None]).squeeze(1),
            "gt_probability": masked_p.gather(1, labels[:, None]).squeeze(1),
            "correct": (masked_p.argmax(1) == labels).float()}


def selection_metrics(mask, foreground, coverage):
    fg_count = foreground.sum(1)
    selected_fg = (mask & foreground).sum(1)
    selected_count = mask.sum(1)
    missing = 1 - selected_fg.float() / fg_count.clamp_min(1)
    missing = missing.masked_fill(fg_count == 0, float("nan"))
    continuous_total = coverage.sum(1)
    continuous_retained = (mask * coverage).sum(1) / continuous_total.clamp_min(1e-8)
    continuous_retained = continuous_retained.masked_fill(continuous_total <= 0, float("nan"))
    return {"foreground_selection_ratio": selected_fg.float() / selected_count,
            "background_selection_ratio": 1 - selected_fg.float() / selected_count,
            "foreground_missing_ratio": missing,
            "foreground_area_ratio": foreground.float().mean(1),
            "foreground_coverage_retained": continuous_retained,
            "foreground_patch_count": fg_count}
