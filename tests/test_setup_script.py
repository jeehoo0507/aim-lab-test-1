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
    executable = tmp_path / "fake_commands"
    executable.mkdir()
    fake_script = '''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
name = Path(sys.argv[0]).name
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
    assert "simulated command succeeded" in (report.parent / "07.log").read_text()
    assert "/dataset\\ with\\ spaces/birds" in (report.parent / "07.log").read_text()


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
    assert "simulated failure" in (report.parent / "07.log").read_text()


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
