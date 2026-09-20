import json
import os

from scripts import export_results as exporter
from waterbirds_kd.config import Config


def test_export_keeps_analysis_excludes_weights_and_marks_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, "ROOT", tmp_path)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    output = tmp_path / "outputs"
    run = output / "seed_0/student"
    (run / "analysis").mkdir(parents=True)
    (run / "probe").mkdir()
    for name in ("config.json", "history.json", "result.json", "probe/manifest.json", "analysis/analysis_info.json"):
        (run / name).write_text("{}")
    (run / "analysis/attention_per_image.csv").write_text("epoch,value\n1,0.5\n")
    (run / "analysis/plot.png").write_bytes(b"example image")
    (run / "best.pt").write_bytes(b"weights")
    (run / "probe/epoch_000.npz").write_bytes(b"raw probes")
    cfg = Config(output_root=str(output), data_root=str(tmp_path / "data"), seg_root="")
    report = exporter.export_results(cfg, tmp_path / "report")
    assert (report / "seed_0/student/analysis/attention_per_image.csv").is_file()
    assert (report / "seed_0/student/history.json").is_file()
    assert not list(report.rglob("*.pt"))
    assert not list(report.rglob("*.npz"))
    manifest = json.loads((report / "manifest.json").read_text())
    assert "seed_0/teacher" in manifest["missing_or_incomplete_runs"]
    assert "seed_0/student" not in manifest["missing_or_incomplete_runs"]
    assert manifest["runs"]["seed_0/student"]["analysis_available"]
    assert all(len(row["sha256"]) == 64 for row in manifest["files"])
    with_probes = exporter.export_results(cfg, tmp_path / "with-probes", include_probes=True)
    assert (with_probes / "seed_0/student/probe/epoch_000.npz").is_file()


def test_storage_total_deduplicates_overlapping_paths_and_hardlinks(tmp_path):
    inner = tmp_path / "inner"
    inner.mkdir()
    first = inner / "first"
    first.write_bytes(b"x" * 8192)
    os.link(first, inner / "second")
    usage = exporter.disk_usage({"project": str(tmp_path), "nested": str(inner)})
    assert usage["unique_allocated_gib"] == first.stat().st_blocks * 512 / 1024**3
    assert usage["paths"]["nested"]["allocated_gib"] == usage["unique_allocated_gib"]
