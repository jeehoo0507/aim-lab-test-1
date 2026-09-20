"""Storage evidence and successful-stage reuse for setup_server.sh (stdlib only)."""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from waterbirds_kd.config import Config


def command_output(command):
    if not shutil.which(command[0]):
        return "unavailable"
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


def path_storage(value):
    requested = Path(value).expanduser().absolute()
    resolved = requested.resolve()
    ancestor = resolved
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    usage = shutil.disk_usage(ancestor)
    mount = command_output(["findmnt", "--json", "--target", str(ancestor),
                            "--output", "SOURCE,TARGET,FSTYPE,OPTIONS"])
    try:
        mount = json.loads(mount)
    except ValueError:
        pass
    return {"requested_path": str(requested), "resolved_path": str(resolved),
            "exists": resolved.exists(), "checked_ancestor": str(ancestor),
            "free_gib": round(usage.free / 1024**3, 2),
            "total_gib": round(usage.total / 1024**3, 2), "mount": mount}


def storage(args):
    cfg = Config.load(args.config, {name: getattr(args, name) for name in
                                  ("data_root", "seg_root", "output_root")})
    paths = {"project": str(ROOT), "venv": str(ROOT / ".venv"),
             "data": cfg.data_root, "segmentations": cfg.seg_root,
             "checkpoints_and_results": cfg.output_root, "smoke": args.smoke_output,
             "setup_cache": str(ROOT / "outputs/setup_cache"),
             "logs": str(ROOT / "logs"), "report": args.report_dir,
             "downloads": str(ROOT / "data/downloads"),
             "torch_cache": os.environ.get("TORCH_HOME", str(Path(os.environ.get(
                 "XDG_CACHE_HOME", "~/.cache")).expanduser() / "torch")),
             "uv_cache": os.environ.get("UV_CACHE_DIR", str(Path(os.environ.get(
                 "XDG_CACHE_HOME", "~/.cache")).expanduser() / "uv"))}
    report = {"paths": {name: path_storage(path) for name, path in paths.items() if path},
              "block_devices": command_output(["lsblk", "--json", "--output",
                                                "NAME,SIZE,ROTA,TYPE,FSTYPE,MOUNTPOINT,MODEL"])}
    directory = Path(args.report_dir)
    (directory / "effective_config.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")
    (directory / "storage.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def inventory(path):
    """Detect deleted/replaced/edited inputs without rereading GBs of images."""
    if not path:
        return []
    root = Path(path).expanduser().resolve()
    entries = [root, *sorted(root.rglob("*"))] if root.is_dir() else [root]
    result = []
    for entry in entries:
        if entry.exists():
            stat = entry.stat()
            result.append((str(entry), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
        else:
            result.append((str(entry), "missing"))
    return result


def fingerprint(args):
    command = list(args.command)
    if args.stage == "smoke":
        command[command.index("--output-root") + 1] = "<smoke-output>"
    packages = args.stage.startswith("install-")
    files = [ROOT / "requirements.txt", ROOT / "scripts/setup_support.py"]
    if not packages:
        files += [ROOT / "run.py", ROOT / "setup.sh", *sorted((ROOT / "waterbirds_kd").glob("*.py")),
                  *sorted((ROOT / "tests").glob("*.py")), *sorted((ROOT / "scripts").glob("*.py")),
                  *sorted((ROOT / "scripts").glob("*.sh")), *sorted((ROOT / "vendor").rglob("*.py"))]
    records = sorted((ROOT / ".venv/lib").glob("python*/site-packages/*.dist-info/RECORD"))
    if args.stage == "install-cuda":
        records = [p for p in records if p.parent.name.startswith(("torch-", "torchvision-"))]
    state = {"command": command, "stage": args.stage,
             "source": [(str(p), hashlib.sha256(p.read_bytes()).hexdigest()) for p in files if p.exists()],
             "python": (sys.version, str(Path(sys.executable).resolve())),
             "host": (platform.node(), platform.platform()),
             "environment": {name: os.environ.get(name) for name in
                             ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG")},
             "venv": inventory(ROOT / ".venv/pyvenv.cfg") + inventory(ROOT / ".venv/bin/python"),
             "packages": [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in records]}
    if not packages:
        cfg = json.loads(Path(args.config).read_text())
        state["config"] = cfg
        state["gpu"] = command_output(["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"])
        if args.stage in ("preflight", "prepare-data"):
            state["data"] = inventory(cfg["data_root"]) + inventory(cfg["seg_root"])
        if args.stage == "prepare-data":
            state["prepared_data"] = inventory(ROOT / "data/waterbird_complete95_forest2water2") + inventory(
                ROOT / "data/CUB_200_2011/segmentations")
            state["archives"] = [inventory(command[i + 1]) for i, arg in enumerate(command)
                                 if arg in ("--waterbirds-archive", "--segmentation-archive")]
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


def cached(args):
    directory = ROOT / "outputs/setup_cache" / args.stage
    directory.mkdir(parents=True, exist_ok=True)
    key = fingerprint(args)
    record_path = directory / f"{key}.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else None
    if not args.force and record and Path(record["log"]).is_file() and all(
            Path(path).is_file() for path in record["artifacts"]):
        # Copy evidence into this report so pushing only this folder is sufficient.
        print(f"[재사용] {args.stage}; 이전 성공 로그: {record['log']}; "
              f"원본 commit: {record.get('commit')}; fingerprint: {key}", flush=True)
        print(Path(record["log"]).read_text(), flush=True)
        Path(args.status_file).write_text("REUSED")
        return 0
    print(f"[실행] {args.stage}", flush=True)
    result = subprocess.run(args.command, check=False)
    if result.returncode:
        return result.returncode
    artifacts = []
    if args.stage == "smoke":
        output = Path(args.command[args.command.index("--output-root") + 1])
        artifacts = [str(p) for p in output.glob("seed_*/*/result.json")]
        if len(artifacts) != 9:
            print("Smoke success artifacts missing; stage will not be cached.", flush=True)
            return 0
    if args.stage == "preflight":
        artifacts = [str(Path(json.loads(Path(args.config).read_text())["output_root"]) / "preflight.json")]
    # Install/download stages change their own inputs. Save the post-success key.
    record_path = directory / f"{fingerprint(args)}.json"
    temp = record_path.with_suffix(".tmp")
    temp.write_text(json.dumps({"log": str(Path(args.log).resolve()), "artifacts": artifacts,
                               "commit": command_output(["git", "rev-parse", "HEAD"])}) + "\n")
    temp.replace(record_path)
    return 0


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    paths = subparsers.add_parser("storage")
    paths.add_argument("--config", required=True)
    for name in ("data-root", "seg-root", "output-root"):
        paths.add_argument(f"--{name}")
    paths.add_argument("--report-dir", required=True)
    paths.add_argument("--smoke-output", required=True)
    cache = subparsers.add_parser("cached")
    for name in ("stage", "config", "log", "status-file"):
        cache.add_argument(f"--{name}", required=True)
    cache.add_argument("--force", action="store_true")
    cache.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.action == "storage":
        storage(args)
        return 0
    if args.command and args.command[0] == "--":
        args.command.pop(0)
    return cached(args)


if __name__ == "__main__":
    sys.exit(main())
