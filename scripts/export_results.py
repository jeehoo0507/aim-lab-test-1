"""Export reviewable experiment evidence without datasets or model weights."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from waterbirds_kd.config import METHODS, Config


def disk_usage(paths):
    """Count allocated bytes once per inode, including uv hard-linked packages."""
    combined = set()
    total = 0
    rows = {}
    for label, value in paths.items():
        if not value:
            continue
        root = Path(value).expanduser().resolve()
        seen = set()
        allocated = 0
        for directory, _, filenames in os.walk(root):
            for name in filenames:
                path = Path(directory) / name
                if path.is_symlink():
                    continue
                stat = path.stat()
                key = (stat.st_dev, stat.st_ino)
                size = getattr(stat, "st_blocks", 0) * 512 or stat.st_size
                if key not in seen:
                    seen.add(key)
                    allocated += size
                if key not in combined:
                    combined.add(key)
                    total += size
        rows[label] = {"path": str(root), "exists": root.exists(), "allocated_gib": allocated / 1024**3}
    return {"paths": rows, "unique_allocated_gib": total / 1024**3,
            "note": "Overlapping paths/hard links counted once in total. Shared caches may include other projects; "
                    "filesystem metadata and nested symlink targets are excluded."}


def export_results(cfg, destination, include_probes=False):
    source = Path(cfg.output_root).resolve()
    runs = sorted(path for path in source.glob("seed_*/*") if path.is_dir())
    if not runs:
        raise ValueError(f"No experiment runs found under {source}")
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Use a new report directory: {destination}")
    files = list(source.glob("preflight.json"))
    files += [p for p in (source / "summary").glob("*") if p.suffix in (".csv", ".png", ".json")]
    status = {}
    for run in runs:
        name = str(run.relative_to(source))
        status[name] = {"completed": (run / "result.json").is_file(),
                        "analysis_available": (run / "analysis/analysis_info.json").is_file()}
        files += [run / name for name in ("config.json", "history.json", "result.json", "probe/manifest.json")]
        files += [p for p in (run / "analysis").glob("*") if p.suffix in (".csv", ".png", ".json")]
        if include_probes:
            files += list((run / "probe").glob("epoch_*.npz"))
    files = sorted({p for p in files if p.is_file()})
    destination.mkdir(parents=True)
    manifest = []
    for path in files:
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        manifest.append({"path": str(path.relative_to(source)), "bytes": target.stat().st_size,
                         "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=False).stdout.strip()
    seeds = sorted({p.parent.name for p in runs})
    missing = [f"{seed}/{method}" for seed in seeds for method in ["teacher", *METHODS]
               if not status.get(f"{seed}/{method}", {}).get("completed")]
    metadata = {"source": str(source), "export_commit": revision, "runs": status,
                "missing_or_incomplete_runs": missing, "include_probes": include_probes,
                "files": manifest, "total_bytes": sum(p["bytes"] for p in manifest)}
    (destination / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    cache = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
    usage = disk_usage({"project": str(ROOT), "dataset": cfg.data_root, "segmentations": cfg.seg_root,
                        "experiment_outputs": cfg.output_root,
                        "uv_cache": os.environ.get("UV_CACHE_DIR", str(cache / "uv")),
                        "torch_cache": os.environ.get("TORCH_HOME", str(cache / "torch"))})
    (destination / "disk_usage.json").write_text(json.dumps(usage, indent=2) + "\n")
    lines = ["# Experiment evidence", "", f"Source: `{source}`", f"Export commit: `{revision}`", "",
             f"Exported artifacts: {metadata['total_bytes'] / 1024**2:.2f} MiB", "",
             "| Run | Training complete | Analysis available |", "|---|---|---|"]
    lines += [f"| {name} | {row['completed']} | {row['analysis_available']} |" for name, row in status.items()]
    lines += ["", f"Missing/incomplete expected runs: {', '.join(missing) or 'none'}", "",
              "Includes config/history/results, per-image and per-group analysis CSVs, figures and summaries.",
              f"Raw epoch probe NPZ files included: {include_probes}.",
              "Without NPZ files, CSVs support convergence/result review but recomputing new attention metrics needs server originals.",
              "Model checkpoints and datasets stay on the server. Missing analyses are not regenerated by this export.",
              "Export commit identifies the exporter checkout; it does not prove which code originally trained the runs.",
              "See manifest.json for artifact checksums and disk_usage.json for current measured storage."]
    (destination / "README.md").write_text("\n".join(lines) + "\n")
    print(f"Experiment report: {destination}\nExported artifacts: {metadata['total_bytes'] / 1024**2:.2f} MiB", flush=True)
    print(f"Measured project + configured data/results + shared caches: {usage['unique_allocated_gib']:.2f} GiB", flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot.json")
    for name in ("data-root", "seg-root", "output-root", "report-dir"):
        parser.add_argument(f"--{name}")
    parser.add_argument("--include-probes", action="store_true")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    cfg = Config.load(args.config, {name: getattr(args, name) for name in ("data_root", "seg_root", "output_root")})
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    destination = Path(args.report_dir or ROOT / "reports/experiments" / timestamp).resolve()
    if args.push and not destination.is_relative_to(ROOT / "reports"):
        parser.error("--push requires a report directory inside this project's reports/")
    report = export_results(cfg, destination, args.include_probes)
    if args.push:
        relative = str(report.relative_to(ROOT))
        subprocess.run(["git", "add", "--", relative], check=True)
        subprocess.run(["git", "commit", "--only", "-m", "chore: add experiment results", "--", relative], check=True)
        subprocess.run(["git", "push"], check=True)
        subprocess.run(["git", "rev-parse", "HEAD"], check=True)


if __name__ == "__main__":
    main()
