"""Run directories: one folder per train / evaluate / optimize call holding its config, provenance and outputs."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from importlib import metadata
import json
from pathlib import Path
import pickle
import platform
import shutil
import subprocess
from typing import Any

from .data import Dataset
from .paths import DEFAULT_RUNS_DIR, REPO_ROOT

PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "xgboost")
# Only changes here make a run irreproducible from its recorded commit.
TRACKED_FOR_DIRTY = ("src", "experiments", "pyproject.toml", "uv.lock")


def git_state() -> dict[str, Any]:
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    try:
        return {"commit": git("rev-parse", "HEAD"),
                "dirty": bool(git("status", "--porcelain", "--", *TRACKED_FOR_DIRTY))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


@dataclass(frozen=True)
class Run:
    path: Path

    @property
    def id(self) -> str:
        return self.path.name

    @property
    def config_path(self) -> Path:
        return self.path / "config.toml"

    def write_json(self, name: str, data) -> None:
        text = json.dumps(data, indent=2, ensure_ascii=False)
        (self.path / name).write_text(text + "\n", encoding="utf-8")

    def read_json(self, name: str):
        return json.loads((self.path / name).read_text(encoding="utf-8"))

    def has(self, name: str) -> bool:
        return (self.path / name).exists()

    def save_model(self, model) -> None:
        with open(self.path / "model.pkl", "wb") as file:
            pickle.dump(model, file)

    def load_model(self):
        # Pickles execute code on load; only open runs produced by this repository.
        with open(self.path / "model.pkl", "rb") as file:
            return pickle.load(file)


def create_run(kind: str, config_path: Path, *, name: str, options: dict[str, Any],
               dataset: Dataset, root: Path | None = None) -> Run:
    root = Path(root) if root else DEFAULT_RUNS_DIR
    root.mkdir(parents=True, exist_ok=True)
    base = f"{datetime.now():%Y%m%d-%H%M%S}-{kind}-{name}"
    for suffix in range(1, 1000):
        path = root / (base if suffix == 1 else f"{base}-{suffix}")
        try:
            path.mkdir()
            break
        except FileExistsError:
            continue
    shutil.copyfile(config_path, path / "config.toml")
    run = Run(path)
    run.write_json("meta.json", {
        "kind": kind,
        "name": name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "options": options,
        "source_config": str(config_path),
        "dataset": {"path": str(dataset.path), "version": dataset.version, "sha256": dataset.sha256},
        "git": git_state(),
        "python": platform.python_version(),
        "packages": {package: metadata.version(package) for package in PACKAGES},
    })
    return run


def resolve_run(reference: str, root: Path | None = None) -> Run:
    """Accept a run directory path or a bare run id under the runs root."""
    path = Path(reference)
    if not path.is_dir():
        path = (Path(root) if root else DEFAULT_RUNS_DIR) / reference
    if not (path / "meta.json").exists():
        raise FileNotFoundError(f"no run found at {reference}")
    return Run(path)


def list_runs(root: Path | None = None) -> list[Run]:
    root = Path(root) if root else DEFAULT_RUNS_DIR
    if not root.exists():
        return []
    # Creation order: run ids share a timestamp when created within the same second.
    paths = [path for path in root.iterdir() if (path / "meta.json").exists()]
    return [Run(path) for path in sorted(paths, key=lambda path: (path / "meta.json").stat().st_mtime_ns)]
