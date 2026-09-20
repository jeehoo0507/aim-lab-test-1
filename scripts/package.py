"""Create a portable source archive, excluding environments, datasets and runs."""
import tarfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
destination = root / "maskedkd_waterbirds_a5000.tar.gz"
excluded = {".venv", ".git", "__pycache__", ".pytest_cache", ".ruff_cache", "outputs", "logs", "reports", "data", ".DS_Store"}
with tarfile.open(destination, "w:gz") as archive:
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (path.is_file() and not any(part in excluded for part in relative.parts)
                and path != destination and path.suffix not in (".pyc", ".gz")):
            archive.add(path, arcname=Path("maskedkd_waterbirds") / relative)
print(destination)
