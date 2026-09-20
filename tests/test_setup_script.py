"""Verify orchestration without installing CUDA or contacting a server."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def setup_project(tmp_path):
    source = Path(__file__).resolve().parents[1]
    project = tmp_path / "project with spaces"
    (project / "scripts").mkdir(parents=True)
    shutil.copy(source / "setup.sh", project / "setup.sh")
    shutil.copy(source / "scripts/setup_server.sh", project / "scripts/setup_server.sh")
    shutil.copy(source / "scripts/setup_support.py", project / "scripts/setup_support.py")
    (project / "waterbirds_kd").mkdir()
    shutil.copy(source / "waterbirds_kd/config.py", project / "waterbirds_kd/config.py")
    (project / "configs").mkdir()
    shutil.copy(source / "configs/pilot.json", project / "configs/pilot.json")
    executable = tmp_path / "fake_commands"
    executable.mkdir()
    fake_script = '''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
name = Path(sys.argv[0]).name
if args and args[0] == "scripts/setup_support.py":
    os.execv(sys.executable, [sys.executable, *args])
with open(os.environ["SETUP_TRACE"], "a") as log:
    log.write(json.dumps({"name": name, "args": args, "cwd": os.getcwd()}) + "\\n")
if name == "uv" and args and args[0] == "venv":
    target = Path(args[-1]) / "bin/python"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(Path(__file__).read_text())
    target.chmod(0o755)
if os.environ.get("SETUP_FAIL_STAGE") in args:
    print("simulated failure")
    sys.exit(23)
if "smoke" in args:
    root = Path(args[args.index("--output-root") + 1])
    for index in range(9):
        path = root / "seed_0" / str(index) / "result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
if "preflight" in args:
    root = Path(args[args.index("--output-root") + 1]) if "--output-root" in args else Path("outputs/pilot")
    root.mkdir(parents=True, exist_ok=True)
    (root / "preflight.json").write_text("{}")
print("simulated command succeeded:", name, args[:2])
'''
    for name in ("uv", "nvidia-smi"):
        path = executable / name
        path.write_text(fake_script)
        path.chmod(0o755)
    trace = tmp_path / "trace.jsonl"
    env = {**os.environ, "PATH": f"{executable}{os.pathsep}{os.environ['PATH']}", "SETUP_TRACE": str(trace)}
    return project, env, trace


def execute(project, env, *args):
    # Deliberately launch from outside the project to verify automatic cd.
    return subprocess.run(["bash", str(project / "setup.sh"), *args], cwd=project.parent,
                          env=env, capture_output=True, text=True, timeout=20, check=False)


def trace_rows(trace):
    return [json.loads(line) for line in trace.read_text().splitlines()]


def test_setup_runs_checks_before_training_and_preserves_paths(setup_project):
    project, env, trace = setup_project
    result = execute(project, env, "--train", "--data-root", "/dataset with spaces/birds",
                     "--seg-root", "/dataset with spaces/masks")
    assert result.returncode == 0, result.stdout + result.stderr
    rows = trace_rows(trace)
    assert all(row["cwd"] == str(project) for row in rows)
    calls = [row["args"] for row in rows if row["name"] == "python"]
    modes = [next(arg for arg in args if arg in ("smoke", "benchmark", "preflight", "pipeline"))
             for args in calls if any(arg in args for arg in ("smoke", "benchmark", "preflight", "pipeline"))]
    assert modes == ["smoke", "benchmark", "preflight", "pipeline"]
    for args in calls:
        if "preflight" in args or "pipeline" in args:
            assert args[args.index("--data-root") + 1] == "/dataset with spaces/birds"
    assert "완료 결과" in result.stdout
    assert len(list((project / "logs").glob("setup_*.log"))) == 1
    report = next((project / "reports/setup").glob("*/README.md"))
    assert "Status: **PASSED**" in report.read_text()
    assert "실제 데이터 검사 | PASSED" in report.read_text()
    assert "simulated command succeeded" in (report.parent / "08.log").read_text()
    assert "/dataset\\ with\\ spaces/birds" in (report.parent / "08.log").read_text()


def test_failed_data_check_stops_before_training(setup_project):
    project, env, trace = setup_project
    result = execute(project, {**env, "SETUP_FAIL_STAGE": "preflight"}, "--train")
    assert result.returncode == 23
    assert "[실패] 실제 데이터 검사" in result.stdout
    assert not any("pipeline" in row["args"] for row in trace_rows(trace))
    assert "완료 결과" not in result.stdout
    report = next((project / "reports/setup").glob("*/README.md"))
    assert "Status: **FAILED**" in report.read_text()
    assert "Exit code: 23" in report.read_text()
    assert "simulated failure" in (report.parent / "08.log").read_text()


def test_default_setup_prints_training_command_without_starting_training(setup_project):
    project, env, trace = setup_project
    result = execute(project, env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not any("pipeline" in row["args"] for row in trace_rows(trace))
    assert "전체 학습 시작 명령" in result.stdout


def test_dry_run_creates_no_environment_or_log(setup_project):
    project, env, trace = setup_project
    result = execute(project, env, "--dry-run", "--train", "--download-data")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not trace.exists()
    assert not (project / ".venv").exists()
    assert not (project / "logs").exists()
    assert not (project / "reports").exists()
    assert "scripts/prepare_data.py" in result.stdout
    assert "실행 예정 순서" in result.stdout


def test_missing_option_value_fails_before_setup(setup_project):
    project, env, trace = setup_project
    result = execute(project, env, "--data-root")
    assert result.returncode == 2
    assert not trace.exists()


def test_failed_gpu_check_preserves_report_before_python_setup(setup_project):
    project, env, trace = setup_project
    env = {**env, "SETUP_FAIL_STAGE": "--query-gpu=name,memory.total,memory.free,driver_version"}
    result = execute(project, env)
    assert result.returncode == 23
    assert not (project / ".venv").exists()
    report = next((project / "reports/setup").glob("*/README.md"))
    assert "서버 사양 확인 | FAILED" in report.read_text()
    assert "simulated failure" in (report.parent / "01.log").read_text()
    assert not any(row["name"] == "uv" for row in trace_rows(trace))


def test_rerun_reuses_successes_and_retries_failed_stage(setup_project):
    project, env, trace = setup_project
    assert execute(project, {**env, "SETUP_FAIL_STAGE": "preflight"}).returncode == 23
    trace.write_text("")
    result = execute(project, env)
    assert result.returncode == 0, result.stdout + result.stderr
    args = [row["args"] for row in trace_rows(trace)]
    assert not any("smoke" in call or "benchmark" in call or "pytest" in call or "install" in call for call in args)
    assert any("preflight" in call for call in args)
    assert any("pip" in call and "check" in call for call in args)
    report = max((project / "reports/setup").glob("*/README.md"), key=lambda p: p.stat().st_mtime_ns)
    assert "CUDA 전체 흐름 smoke test | REUSED" in report.read_text()
    assert "simulated command succeeded" in (report.parent / "06.log").read_text()


@pytest.mark.parametrize("change", ["config", "source", "environment", "packages", "force", "smoke-artifact"])
def test_changed_inputs_or_force_invalidate_cached_smoke(setup_project, change):
    project, env, trace = setup_project
    assert execute(project, env).returncode == 0
    if change == "config":
        cfg_path = project / "configs/pilot.json"
        cfg = json.loads(cfg_path.read_text())
        cfg["batch_size"] = 16
        cfg_path.write_text(json.dumps(cfg))
    elif change == "source":
        (project / "waterbirds_kd/new_module.py").write_text("# changed code\n")
    elif change == "environment":
        env = {**env, "CUDA_VISIBLE_DEVICES": "2"}
    elif change == "packages":
        record = project / ".venv/lib/python3.11/site-packages/torch-2.5.1.dist-info/RECORD"
        record.parent.mkdir(parents=True)
        record.write_text("changed installation")
    elif change == "smoke-artifact":
        next((project / "outputs/setup_smoke").glob("*/seed_0/*/result.json")).unlink()
    trace.write_text("")
    result = execute(project, env, *(["--force-checks"] if change == "force" else []))
    assert result.returncode == 0, result.stdout + result.stderr
    assert any("smoke" in row["args"] for row in trace_rows(trace))


def test_failed_smoke_is_retried_with_new_output_directory(setup_project):
    project, env, trace = setup_project
    assert execute(project, {**env, "SETUP_FAIL_STAGE": "smoke"}).returncode == 23
    assert execute(project, env).returncode == 0
    calls = [row["args"] for row in trace_rows(trace) if "smoke" in row["args"]]
    assert len(calls) == 2
    assert calls[0][calls[0].index("--output-root") + 1] != calls[1][calls[1].index("--output-root") + 1]


def test_data_edit_invalidates_preflight_without_repeating_gpu_checks(setup_project):
    project, env, trace = setup_project
    data = project / "data/waterbird_complete95_forest2water2"
    data.mkdir(parents=True)
    (data / "bird.jpg").write_text("first")
    assert execute(project, env).returncode == 0
    (data / "bird.jpg").write_text("changed")
    trace.write_text("")
    assert execute(project, env).returncode == 0
    calls = [row["args"] for row in trace_rows(trace)]
    assert any("preflight" in args for args in calls)
    assert not any("smoke" in args or "benchmark" in args for args in calls)


def test_storage_report_resolves_symlinks_and_cli_override(setup_project, tmp_path):
    project, env, _ = setup_project
    destination = tmp_path / "storage drive"
    destination.mkdir()
    link = project / "external"
    link.symlink_to(destination, target_is_directory=True)
    result = execute(project, env, "--output-root", str(link / "results"))
    assert result.returncode == 0, result.stdout + result.stderr
    report = next((project / "reports/setup").glob("*/storage.json"))
    paths = json.loads(report.read_text())["paths"]
    output = paths["checkpoints_and_results"]
    assert output["resolved_path"] == str(destination / "results")
    assert output["checked_ancestor"] == str(destination)
    assert output["free_gib"] > 0
    assert "mount" in output
    assert {"data", "segmentations", "logs", "report", "venv", "torch_cache", "uv_cache"} <= paths.keys()
    cfg = json.loads((report.parent / "effective_config.json").read_text())
    assert cfg["output_root"] == str(link / "results")


@pytest.mark.parametrize("fail_stage", [None, "preflight"])
def test_push_report_commits_only_report_even_on_setup_failure(setup_project, tmp_path, fail_stage):
    project, env, _ = setup_project
    remote = tmp_path / "remote.git"

    def git(*args):
        return subprocess.run(["git", *args], cwd=project, env=env, capture_output=True,
                              text=True, check=True).stdout

    git("init")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Setup test")
    git("add", "setup.sh")
    git("commit", "-m", "initial")
    git("init", "--bare", str(remote))
    git("remote", "add", "origin", str(remote))
    git("push", "-u", "origin", "HEAD")
    (project / "unrelated.txt").write_text("must stay staged")
    git("add", "unrelated.txt")
    if fail_stage:
        env = {**env, "SETUP_FAIL_STAGE": fail_stage}
    result = execute(project, env, "--push-report")
    assert result.returncode == (23 if fail_stage else 0), result.stdout + result.stderr
    assert git("diff", "--cached", "--name-only").strip() == "unrelated.txt"
    committed = git("diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD").splitlines()
    assert committed and all(path.startswith("reports/setup/") for path in committed)
    report_path = next(path for path in committed if path.endswith("README.md"))
    expected = "FAILED" if fail_stage else "PASSED"
    assert f"Status: **{expected}**" in git("show", f"HEAD:{report_path}")
    assert git("rev-parse", "HEAD").strip() in git("ls-remote", "origin")
