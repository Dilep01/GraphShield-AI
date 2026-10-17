"""Resolve all writable paths within the repository, including symlink checks."""

from pathlib import Path
import os

_checkout = Path(__file__).resolve().parents[2]
# Installed wheels need an explicit data workspace; editable checkouts retain their default.
ROOT = Path(os.environ.get('GRAPHSHIELD_ROOT', str(_checkout if (_checkout/'pyproject.toml').is_file() else Path.cwd()))).resolve()


def inside_root(path: str | Path, root: Path = ROOT) -> Path:
    root = root.resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Path must stay inside project folder: {root}")
    return candidate
