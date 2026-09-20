#!/usr/bin/env python3
"""Single-GPU entry point. Run `python run.py --help`."""
import argparse
import json
from pathlib import Path

from waterbirds_kd.config import METHODS, Config


def common(parser):
    parser.add_argument("--config", default="configs/pilot.json")
    for name in ("data-root", "seg-root", "output-root", "device", "student-init"):
        parser.add_argument(f"--{name}")
    for name in ("seed", "batch-size", "eval-batch-size", "accumulation-steps", "num-workers", "epochs", "teacher-epochs"):
        parser.add_argument(f"--{name}", type=int)


def configuration(args):
    keys = ("data_root", "seg_root", "output_root", "device", "student_init", "seed", "batch_size",
            "eval_batch_size", "accumulation_steps", "num_workers", "epochs", "teacher_epochs")
    return Config.load(args.config, {key: getattr(args, key, None) for key in keys})


def pipeline(cfg, stage="baselines", teacher_checkpoint=None, make_plots=True):
    from waterbirds_kd.analysis import analyze_run, summarize
    from waterbirds_kd.preflight import preflight
    from waterbirds_kd.train import train
    preflight(cfg)
    methods = METHODS if stage == "all" else METHODS[:4] if stage == "baselines" else METHODS[4:]
    if any("rescue" in method for method in methods) and not cfg.seg_root:
        raise ValueError("The rescue stage requires segmentation; set seg_root.")
    teacher_checkpoint = teacher_checkpoint or train(cfg, role="teacher")
    for method in methods:
        checkpoint = train(cfg, method=method, teacher_checkpoint=teacher_checkpoint)
        if make_plots:
            analyze_run(checkpoint.parent, cfg.data_root, cfg.seg_root)
    if make_plots:
        summarize(cfg.output_root)


def main():
    parser = argparse.ArgumentParser(description="Waterbirds MaskedKD / Oracle Rescue experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("preflight", help="Check data paths, segmentation, group counts, and CUDA")
    common(check)
    check.add_argument("--full-image-check", action="store_true")
    train_parser = commands.add_parser("train", help="Train/resume one teacher or student")
    common(train_parser)
    train_parser.add_argument("--role", choices=["teacher", "student"], default="student")
    train_parser.add_argument("--method", choices=METHODS, default="student")
    train_parser.add_argument("--teacher-checkpoint")
    batch = commands.add_parser("pipeline", help="Run experiments sequentially; resume completed/partial runs")
    common(batch)
    batch.add_argument("--stage", choices=["baselines", "rescue", "all"], default="baselines")
    batch.add_argument("--teacher-checkpoint")
    batch.add_argument("--no-plots", action="store_true")
    analysis = commands.add_parser("analyze", help="Regenerate CSVs and figures")
    common(analysis)
    analysis.add_argument("--run-dir")
    summary = commands.add_parser("summarize", help="Aggregate completed runs across seeds")
    summary.add_argument("--output-root", default="outputs/pilot")
    evaluation = commands.add_parser("evaluate", help="Evaluate a saved best/last checkpoint")
    common(evaluation)
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--split", choices=["validation", "test"], default="test")
    bench = commands.add_parser("benchmark", help="Brief real-model training/memory check using random inputs")
    common(bench)
    bench.add_argument("--steps", type=int, default=3)
    smoke = commands.add_parser("smoke", help="Synthetic end-to-end test; no weights/data download")
    smoke.add_argument("--output-root", default="outputs/smoke")
    smoke.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "smoke":
        from waterbirds_kd.synthetic import create_synthetic
        data, segmentation = create_synthetic(Path(args.output_root) / "synthetic_fixture")
        cfg = Config.load(overrides={"data_root": data, "seg_root": segmentation, "output_root": args.output_root,
                                    "device": args.device, "model_scale": "debug", "teacher_pretrained": False,
                                    "student_init": "scratch", "batch_size": 2, "eval_batch_size": 4,
                                    "accumulation_steps": 2, "num_workers": 0, "num_threads": 2,
                                    "epochs": 2, "teacher_epochs": 1, "warmup_epochs": 0,
                                    "max_train_batches": 2, "probe_per_group": 1, "diagnostic_epochs": [0, 1, 2]})
        pipeline(cfg, "all")
        print(f"Smoke test complete (synthetic, not research results): {args.output_root}")
        return
    if args.command == "summarize":
        from waterbirds_kd.analysis import summarize
        print(summarize(args.output_root))
        return
    cfg = configuration(args)
    if args.command == "preflight":
        from waterbirds_kd.preflight import preflight
        print(json.dumps(preflight(cfg, args.full_image_check), indent=2))
    elif args.command == "benchmark":
        from waterbirds_kd.benchmark import benchmark
        if args.steps < 1:
            raise ValueError("steps must be positive")
        print(json.dumps(benchmark(cfg, args.steps), indent=2))
    elif args.command == "train":
        from waterbirds_kd.train import train
        train(cfg, args.role, args.method, args.teacher_checkpoint)
    elif args.command == "pipeline":
        pipeline(cfg, args.stage, args.teacher_checkpoint, not args.no_plots)
    elif args.command == "analyze":
        from waterbirds_kd.analysis import analyze_run, summarize
        runs = [Path(args.run_dir)] if args.run_dir else sorted(Path(cfg.output_root).glob("seed_*/*"))
        for run in runs:
            if (run / "probe").is_dir():
                print(analyze_run(run, args.data_root, args.seg_root))
        print(summarize(cfg.output_root))
    elif args.command == "evaluate":
        from waterbirds_kd.data import Waterbirds
        from waterbirds_kd.models import build_model
        from waterbirds_kd.train import evaluate
        from waterbirds_kd.utils import load_checkpoint, metadata_hash, setup_device
        state = load_checkpoint(args.checkpoint)
        # Architecture/preprocessing follow checkpoint, not an unrelated config file.
        saved_cfg = Config.load(overrides={**state["config"], "data_root": cfg.data_root,
                                           "device": cfg.device, "num_workers": cfg.num_workers,
                                           "eval_batch_size": cfg.eval_batch_size})
        if state["metadata_sha256"] != metadata_hash(saved_cfg):
            raise ValueError("Evaluation dataset differs from checkpoint metadata")
        device = setup_device(saved_cfg)
        model = build_model(state["role"], saved_cfg).to(device)
        model.load_state_dict(state["model"])
        data = Waterbirds(saved_cfg.data_root, 1 if args.split == "validation" else 2)
        print(json.dumps(evaluate(model, data, saved_cfg, device, state["train_group_weights"]), indent=2))


if __name__ == "__main__":
    main()
