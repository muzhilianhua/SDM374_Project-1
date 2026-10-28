"""Loading the official training table, verified against its manifest checksum."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .simplex import FEATURES, check_weights

TARGET = "ramping"


class DataIntegrityError(ValueError):
    """The data on disk does not match its manifest."""


@dataclass(frozen=True)
class Dataset:
    ids: tuple[str, ...]
    X: np.ndarray  # (n, 21) weights
    y: np.ndarray  # (n,) mean ramping over weeks 1-8, lower is better
    version: str
    sha256: str
    path: Path


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(directory, *, verify: bool = True) -> Dataset:
    """Load <directory>/train.csv; with verify=True the file must match manifest.json byte for byte."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    csv = directory / "train.csv"
    digest = sha256_of(csv)
    if verify and digest != manifest["train_sha256"]:
        raise DataIntegrityError(f"{csv} sha256 {digest} does not match manifest {manifest['train_sha256']}")

    table = pd.read_csv(csv, float_precision="round_trip", dtype={"candidate_id": str})
    expected = {"candidate_id", *FEATURES, TARGET}
    if set(table.columns) != expected:
        raise ValueError(f"unexpected columns; missing {sorted(expected - set(table.columns))}, "
                         f"extra {sorted(set(table.columns) - expected)}")
    if table.candidate_id.isna().any() or table.candidate_id.duplicated().any():
        raise ValueError("candidate_id must be present and unique")
    if verify and len(table) != manifest["rows"]:
        raise DataIntegrityError(f"{len(table)} rows, manifest says {manifest['rows']}")
    # Row order decides the K-fold split, so sort exactly like baseline.py to keep results comparable.
    table = table.sort_values("candidate_id", kind="stable").reset_index(drop=True)

    X = check_weights(table[list(FEATURES)].to_numpy(dtype=float))
    y = table[TARGET].to_numpy(dtype=float)
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError(f"{TARGET} must be finite and non-negative")
    return Dataset(tuple(table.candidate_id), X, y, manifest["version"], digest, directory)
