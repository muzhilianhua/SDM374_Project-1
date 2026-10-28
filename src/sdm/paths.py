"""Repository locations; relative paths in configs resolve against REPO_ROOT, not the cwd."""
from pathlib import Path

# Valid for the editable install `uv sync` creates (src/sdm/paths.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"


def resolve(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path
