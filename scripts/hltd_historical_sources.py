"""Resolve historical code evidence without relaxing live execution receipts."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = Path("docs/source_snapshots/f6dd7a941624f0a21a2b6d7b8314aef38a85c5e1")


def _matches(path: Path, record: dict) -> bool:
    if not path.is_file() or path.stat().st_size != record["bytes"]:
        return False
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() == record["sha256"]


def historical_source_path(record: dict, root: Path = ROOT) -> Path:
    """Return matching live or archived repo-local Python, never archived data."""
    relative = Path(record["path"])
    if (relative.is_absolute() or ".." in relative.parts or relative.suffix != ".py"
            or relative.parts[0] not in {"scripts", "tests", "spiral_hodge.py"}):
        raise ValueError(f"not repository-local source: {relative}")
    for path in (root / relative, root / SNAPSHOT / (str(relative) + ".txt")):
        if _matches(path, record):
            return path
    raise ValueError(f"historical source unavailable or changed: {relative}")


def verify_historical_inputs(protocol: dict, root: Path = ROOT) -> list[dict]:
    """Audit a prior run for a new freeze; live data and external code stay exact.

    This is not an execution preflight. New protocols must record live source
    bytes, and runners must continue using strict ``verify_frozen_files``.
    """
    snapshots = []
    for record in protocol["frozen_files"]:
        path = root / record["path"]
        if _matches(path, record):
            continue
        source = historical_source_path(record, root)
        snapshots.append({**record, "snapshot": str(source.relative_to(root))})
    return snapshots
