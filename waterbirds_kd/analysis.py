import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import rankdata, spearmanr
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .config import GROUP_NAMES, METHODS, Config
from .data import Waterbirds
from .masking import binary_mask
from .metrics import degradation, selection_metrics
from .utils import write_json

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


def row_spearman(a, b):
    a, b = rankdata(a, axis=1), rankdata(b, axis=1)
    a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    denom = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return np.divide((a * b).sum(1), denom, out=np.full(len(a), np.nan), where=denom > 0)


def to_numpy_dict(values):
    return {key: value.cpu().numpy() for key, value in values.items()}


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def trajectory_plot(table, column, title, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for group, rows in table.groupby("group"):
        curve = rows.groupby("epoch")[column].mean()
        ax.plot(curve.index, curve.values, label=GROUP_NAMES[int(group)])
    curve = table.groupby("epoch")[column].mean()
    ax.plot(curve.index, curve.values, color="black", linestyle="--", label="balanced probe mean")
    ax.set(xlabel="Epoch (0 = before training)", ylabel=column, title=title)
    ax.legend(fontsize=7)
    save_figure(fig, path)


def analyze_run(run_dir, data_root=None, seg_root=None):
    run_dir = Path(run_dir)
    files = sorted((run_dir / "probe").glob("epoch_*.npz"))
    if not files:
        return None
    cfg = Config.load(run_dir / "config.json", {"data_root": data_root, "seg_root": seg_root})
    out = run_dir / "analysis"
    out.mkdir(exist_ok=True)
    with np.load(files[-1], allow_pickle=False) as archive:
        final = {key: archive[key] for key in archive.files}
    trajectory, diagnostics = [], []
    previous_attention = None
    last_diagnostic = None
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            record = {key: archive[key] for key in archive.files}
        if not np.array_equal(record["sample_id"], final["sample_id"]):
            raise ValueError(f"Probe image/order changed: {path}")
        epoch = int(record["epoch"])
        attention = record["attention"]
        row = pd.DataFrame({"sample_id": record["sample_id"], "group": record["group"], "epoch": epoch,
                            "final_spearman": row_spearman(attention, final["attention"]),
                            "previous_spearman": row_spearman(attention, previous_attention) if previous_attention is not None else np.nan,
                            "topk_final_overlap": (record["selected_mask"] & final["selected_mask"]).sum(1) / cfg.keep_tokens})
        foreground = torch.from_numpy(record["foreground"])
        coverage = torch.from_numpy(record["coverage"])
        if bool(record["has_segmentation"]):
            ratios = selection_metrics(torch.from_numpy(record["selected_mask"]), foreground, coverage)
            for key, value in to_numpy_dict(ratios).items():
                row[key] = value
        trajectory.append(row)
        previous_attention = attention
        if "teacher_full_logits" not in record:
            continue
        last_diagnostic = record
        full = torch.from_numpy(record["teacher_full_logits"])
        labels = torch.from_numpy(record["label"]).long()
        full_correct = full.argmax(1).numpy() == record["label"]
        student_correct = record["student_logits"].argmax(1) == record["label"]
        baseline = degradation(full, torch.from_numpy(record["teacher_logits__student"]), labels)
        for key, masked_logits in record.items():
            if not key.startswith("teacher_logits__"):
                continue
            method = key.split("__", 1)[1]
            values = degradation(full, torch.from_numpy(masked_logits), labels)
            mask = binary_mask(torch.from_numpy(record[f"indices__{method}"]))
            entry = pd.DataFrame({"sample_id": record["sample_id"], "group": record["group"], "epoch": epoch,
                                  "mask_method": method, "actual_swaps": record[f"swaps__{method}"],
                                  "full_teacher_correct": full_correct, "student_correct": student_correct,
                                  "correction_opportunity": full_correct & ~student_correct})
            for metric, value in to_numpy_dict(values).items():
                entry[metric] = value
            entry["kl_reduction_vs_student_mask"] = (baseline["kl"] - values["kl"]).numpy()
            entry["gt_probability_gain_vs_student_mask"] = (baseline["gt_probability_drop"] - values["gt_probability_drop"]).numpy()
            if bool(record["has_segmentation"]):
                for metric, value in to_numpy_dict(selection_metrics(mask, foreground, coverage)).items():
                    entry[metric] = value
            diagnostics.append(entry)
    trajectory = pd.concat(trajectory, ignore_index=True)
    trajectory.to_csv(out / "attention_per_image.csv", index=False)
    trajectory.groupby(["epoch", "group"]).mean(numeric_only=True).drop(columns="sample_id").to_csv(out / "attention_by_group.csv")
    trajectory.groupby("epoch").mean(numeric_only=True).drop(columns=["sample_id", "group"]).to_csv(out / "attention_overall.csv")
    trajectory_plot(trajectory, "final_spearman", "Attention convergence to final epoch", out / "01_attention_convergence.png")
    trajectory_plot(trajectory, "previous_spearman", "Attention similarity to previous recorded epoch", out / "01b_consecutive_attention.png")
    trajectory_plot(trajectory, "topk_final_overlap", "Top-K overlap with final epoch", out / "01c_topk_overlap.png")
    if "background_selection_ratio" in trajectory:
        trajectory_plot(trajectory, "background_selection_ratio", "Background selection", out / "02_background_selection.png")
        trajectory_plot(trajectory, "foreground_missing_ratio", "Fraction of foreground patches omitted", out / "03_foreground_missing.png")
    if diagnostics:
        diagnostics = pd.concat(diagnostics, ignore_index=True)
        diagnostics.to_csv(out / "teacher_degradation_per_image.csv", index=False)
        diagnostics.groupby(["epoch", "mask_method", "group"]).mean(numeric_only=True).drop(columns="sample_id").to_csv(out / "teacher_degradation_by_group.csv")
        diagnostics.groupby(["epoch", "mask_method"]).mean(numeric_only=True).drop(columns=["sample_id", "group"]).to_csv(out / "teacher_degradation_overall.csv")
        eligible = diagnostics.loc[diagnostics.correction_opportunity]
        eligible.groupby(["epoch", "mask_method", "group"]).mean(numeric_only=True).drop(columns="sample_id").to_csv(out / "teacher_correction_opportunities.csv")
        counts = diagnostics.groupby(["epoch", "mask_method", "group"])["correction_opportunity"].agg(["count", "sum"])
        counts.to_csv(out / "correction_opportunity_counts.csv")
        plot_rescue(diagnostics, out)
        if "background_selection_ratio" in diagnostics:
            correlations = []
            for (epoch, method), rows in diagnostics.groupby(["epoch", "mask_method"]):
                for group in [-1, 0, 1, 2, 3]:
                    group_rows = rows if group == -1 else rows.loc[rows.group == group]
                    for x, y in [("background_selection_ratio", "kl"), ("foreground_missing_ratio", "kl"),
                                 ("background_selection_ratio", "gt_probability_drop")]:
                        clean = group_rows[[x, y]].dropna()
                        rho = float(spearmanr(clean[x], clean[y]).statistic) if len(clean) >= 3 and clean[x].nunique() > 1 and clean[y].nunique() > 1 else None
                        correlations.append({"epoch": epoch, "mask_method": method, "group": group,
                                             "x": x, "y": y, "spearman": rho, "n": len(clean)})
            pd.DataFrame(correlations).to_csv(out / "correlations.csv", index=False)
            latest = diagnostics.loc[(diagnostics.epoch == diagnostics.epoch.max()) & (diagnostics.mask_method == "student")]
            fig, ax = plt.subplots(figsize=(7, 4))
            for group, rows in latest.groupby("group"):
                ax.scatter(rows.background_selection_ratio, rows.kl, s=15, alpha=0.6, label=GROUP_NAMES[int(group)])
            ax.set(xlabel="Background selection ratio", ylabel="KL(full || masked)", title="Final probe: student mask")
            ax.legend(fontsize=7)
            save_figure(fig, out / "04_background_vs_kl.png")
    if last_diagnostic is not None and bool(last_diagnostic["has_segmentation"]):
        visualize_examples(last_diagnostic, cfg, out / "07_examples.png")
    write_json(out / "analysis_info.json", {"run": str(run_dir), "epochs": [int(x) for x in trajectory.epoch.unique()],
                                           "reference": "final training epoch; not the validation-selected best checkpoint",
                                           "probe_size": len(final["sample_id"]),
                                           "nan_policy": "undefined FG ratios or correlations remain missing; never zero-filled"})
    return out


def plot_rescue(table, out):
    latest = table.loc[table.epoch == table.epoch.max()]
    summary = latest.groupby("mask_method")[["kl", "gt_probability", "correct", "actual_swaps",
                                               "kl_reduction_vs_student_mask", "gt_probability_gain_vs_student_mask"]].mean()
    summary.to_csv(out / "teacher_rescue_final.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ["kl", "gt_probability"]):
        ax.bar(summary.index, summary[metric], color="#327a8e")
        ax.tick_params(axis="x", labelrotation=65, labelsize=7)
        ax.set(ylabel=metric, title=f"Final probe teacher: {metric}")
    save_figure(fig, out / "teacher_rescue.png")


def visualize_examples(record, cfg, destination):
    dataset = Waterbirds(cfg.data_root, 1, cfg.seg_root, threshold=cfg.foreground_threshold,
                         indices=record["sample_id"])
    fig, axes = plt.subplots(4, 6, figsize=(16, 11))
    for group in range(4):
        i = int(np.flatnonzero(record["group"] == group)[0])
        item = dataset[i]
        image = item["image"].permute(1, 2, 0).numpy() * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        image = np.clip(image, 0, 1)
        row = dataset.frame.iloc[i]
        with Image.open(Path(cfg.seg_root) / Path(row.img_filename).with_suffix(".png")) as im:
            segmentation = TF.center_crop(TF.resize(im.convert("L"), 256, InterpolationMode.NEAREST), 224)
        axes[group, 0].imshow(image)
        axes[group, 0].set_title(f"{GROUP_NAMES[group]}\nid={item['sample_id']}")
        axes[group, 1].imshow(segmentation, cmap="gray", vmin=0, vmax=255)
        axes[group, 1].set_title("Bird segmentation")
        axes[group, 2].imshow(record["attention"][i].reshape(14, 14), cmap="magma")
        axes[group, 2].set_title("Student attention")
        for col, method in [(3, "student"), (4, "foreground_rescue_10")]:
            selected = np.zeros(196)
            selected[record[f"indices__{method}"][i]] = 1
            pixel_mask = np.repeat(np.repeat(selected.reshape(14, 14), 16, 0), 16, 1)
            axes[group, col].imshow(image * (0.15 + pixel_mask[..., None] * 0.85))
            count = int(record[f"swaps__{method}"][i])
            axes[group, col].set_title("Student mask" if col == 3 else f"FG rescue-10: {count} swaps")
        text = [f"GT: {int(record['label'][i])} (0=land, 1=water)"]
        for name, key in [("Full", "teacher_full_logits"), ("Masked", "teacher_logits__student"),
                          ("Rescue", "teacher_logits__foreground_rescue_10")]:
            p = torch.from_numpy(record[key][i]).softmax(0).numpy()
            text.append(f"{name}: pred={p.argmax()}\np(GT)={p[int(record['label'][i])]:.3f}")
        axes[group, 5].text(0, 0.95, "\n\n".join(text), va="top", transform=axes[group, 5].transAxes)
        for ax in axes[group]:
            ax.axis("off")
    fig.suptitle(f"Epoch {int(record['epoch'])}: first fixed probe example in each group (no cherry-picking)")
    save_figure(fig, destination)


def summarize(root):
    root = Path(root)
    out = root / "summary"
    out.mkdir(exist_ok=True)
    rows = []
    for path in sorted(root.glob("seed_*/*/result.json")):
        result = json.loads(path.read_text())
        row = {"method": result["method"], "seed": result["seed"], "best_epoch": result["best_epoch"],
               "model_scale": result["model_scale"], "partial_training": result["partial_training"],
               "accuracy": result["test"]["accuracy"], "worst_group_accuracy": result["test"]["worst_group_accuracy"],
               "train_weighted_accuracy": result["test"]["train_weighted_accuracy"], **result["test"]["group_accuracy"]}
        rows.append(row)
    if not rows:
        return out
    table = pd.DataFrame(rows)
    table.to_csv(out / "results_per_seed.csv", index=False)
    table.groupby("method")[["accuracy", "train_weighted_accuracy", "worst_group_accuracy", *GROUP_NAMES]].agg(["mean", "std", "count"]).to_csv(out / "results_mean_std.csv")
    student_rows = table.loc[table.method != "teacher"]
    if student_rows.empty:
        return out
    order = [method for method in METHODS if method in set(student_rows.method)]
    means = student_rows.groupby("method").mean(numeric_only=True).reindex(order)
    stds = student_rows.groupby("method").std(numeric_only=True).reindex(order).fillna(0)
    smoke_label = " [SYNTHETIC/DEBUG OR PARTIAL RUN]" if (table.model_scale == "debug").any() or table.partial_training.any() else ""
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(order))
    for group, name in enumerate(GROUP_NAMES):
        ax.bar(x + (group - 1.5) * 0.2, means[name], 0.2, yerr=stds[name], capsize=2, label=name)
    ax.set_xticks(x, order, rotation=35, ha="right")
    ax.set(ylabel="Test accuracy", ylim=(0, 1.05), title="Four-group accuracy (mean ± sample SD across seeds)" + smoke_label)
    ax.legend(fontsize=8)
    save_figure(fig, out / "05_group_accuracy.png")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(order, means.worst_group_accuracy, yerr=stds.worst_group_accuracy, capsize=3, color="#327a8e")
    ax.tick_params(axis="x", labelrotation=35)
    ax.set(ylabel="Test worst-group accuracy", ylim=(0, 1.05), title="Worst-group accuracy (mean ± sample SD)" + smoke_label)
    save_figure(fig, out / "06_worst_group_accuracy.png")
    # Paired seed differences are more informative than overlapping error bars.
    paired = []
    pivot = student_rows.pivot(index="seed", columns="method", values="worst_group_accuracy")
    if "student" in pivot:
        for method in order:
            if method == "student":
                continue
            differences = (pivot[method] - pivot["student"]).dropna()
            paired.append({"method": method, "reference": "student", "paired_seeds": len(differences),
                           "mean_wga_delta": differences.mean(), "std_wga_delta": differences.std()})
    pd.DataFrame(paired).to_csv(out / "paired_wga_differences.csv", index=False)
    return out
