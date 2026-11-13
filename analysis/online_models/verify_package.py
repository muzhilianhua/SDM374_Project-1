# -*- coding: utf-8 -*-
"""Verify the hp_noise001 bundle: file hashes + the in-sample metrics from metrics.json.

Run from the repository root:
    uv run python analysis/online_models/verify_package.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PACKAGE = HERE / "hp_noise001"
TRAIN = REPO / "dataset_same8" / "train.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_files() -> None:
    expected = json.loads((PACKAGE / "CHECKSUMS.json").read_text(encoding="utf-8"))
    # CHECKSUMS.json is the reference file itself — exclude it from the integrity check.
    expected = {k: v for k, v in expected.items() if k != "CHECKSUMS.json"}
    for name, want in expected.items():
        got = sha256(PACKAGE / name)
        status = "OK" if got == want else "MISMATCH"
        print(f"[{status}] {name}: {got}")
        if got != want:
            raise SystemExit(f"sha256 mismatch for {name}")


def load_rows():
    sys.path.insert(0, str(HERE))
    from modellib import FEATURES

    with open(TRAIN, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    X = np.asarray([[float(row[name]) for name in FEATURES] for row in rows], dtype=np.float64)
    y = np.asarray([float(row["ramping"]) for row in rows], dtype=np.float64)
    return X, y


def main() -> None:
    check_files()

    sys.path.insert(0, str(HERE))
    from modellib import load_model

    model = load_model(PACKAGE)
    X, y = load_rows()
    pred = np.asarray(model.predict(X, return_std=False), dtype=np.float64)

    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    mae = float(np.mean(np.abs(pred - y)))
    metrics = json.loads((PACKAGE / "metrics.json").read_text(encoding="utf-8"))["in_sample"]["overall"]
    print(f"in-sample: n={len(y)} RMSE={rmse:.6f} MAE={mae:.6f} "
          f"(metrics.json: RMSE={metrics['rmse']:.6f} MAE={metrics['mae']:.6f})")

    if abs(rmse - metrics["rmse"]) > 1e-6 or abs(mae - metrics["mae"]) > 1e-6:
        raise SystemExit("in-sample metrics do not match metrics.json")
    print("PASS: file hashes and metrics reproduced")


if __name__ == "__main__":
    main()
