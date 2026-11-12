# -*- coding: utf-8 -*-
"""Rebuild hp_noise001 from the frozen, untrained spec and verify it against the committed package.

Everything is fixed in `hp_noise001_untrained/train_config.json`: 7 length scales, the global
interaction weight, noise (0.01), jitter (1e-8), unit sample weights, the target scaler rule and the
dataset hash. No optimiser and no RNG are involved, so training is a single Cholesky solve.

Run from the repository root:
    .venv/Scripts/python analysis/online_models/train_hp_noise001.py
    .venv/Scripts/python analysis/online_models/train_hp_noise001.py --write out/model.pkl

The default mode compares the freshly computed alpha vector, scaler and training predictions with
`hp_noise001/model.pkl`; `--write` additionally dumps the freshly computed payload.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SPEC = HERE / "hp_noise001_untrained" / "train_config.json"
TRAINED = HERE / "hp_noise001"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(spec):
    rel = spec["dataset"]["path"]
    path = REPO / rel
    got = sha256(path)
    want = spec["dataset"]["sha256"]
    if got != want:
        raise SystemExit(f"dataset sha256 mismatch: {got} != {want}")
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != spec["dataset"]["rows"]:
        raise SystemExit(f"dataset row count mismatch: {len(rows)} != {spec['dataset']['rows']}")

    features = spec["feature_order"]
    X = np.asarray([[float(row[name]) for name in features] for row in rows], dtype=np.float64)
    y = np.asarray([float(row[spec["target"]["column"]]) for row in rows], dtype=np.float64)
    print(f"dataset OK: {rel} rows={len(y)} sha256={got}")
    return X, y


def train(spec, X, y):
    sys.path.insert(0, str(HERE))
    from modellib import block_ard_cov

    kernel = spec["kernel"]
    loc = float(y.mean())
    scale = float(y.std(ddof=0))
    K = block_ard_cov(X, X, np.asarray(kernel["length_scale"], dtype=np.float64),
                      kernel["global_weight"])
    n = len(y)
    weights = np.ones(n, dtype=np.float64)
    K_reg = K.copy()
    K_reg[np.diag_indices_from(K_reg)] += (kernel["noise"] + kernel["jitter"]) / weights
    alpha = cho_solve(cho_factor(K_reg), (y - loc) / scale)
    # Predictions use the noise-free kernel, matching ModelBundle._gp_mean.
    pred = loc + scale * (K @ alpha)
    return {
        "schema": spec["schema"],
        "kind": "gp",
        "target": "raw",
        "feature_order": spec["feature_order"],
        "n_features": len(spec["feature_order"]),
        "X_train": X,
        "alpha": alpha,
        "length_scale": np.asarray(kernel["length_scale"], dtype=np.float64),
        "global_weight": kernel["global_weight"],
        "noise": kernel["noise"],
        "jitter": kernel["jitter"],
        "weights": weights,
        "loc": loc,
        "scale": scale,
        "train_rows": n,
    }, pred


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path, default=None,
                        help="dump the freshly computed payload to this path")
    args = parser.parse_args()

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    X, y = load_dataset(spec)
    payload, pred = train(spec, X, y)

    with open(TRAINED / "model.pkl", "rb") as handle:
        committed = pickle.load(handle)

    sys.path.insert(0, str(HERE))
    from modellib import load_model

    committed_pred = np.asarray(load_model(TRAINED).predict(X, return_std=False), dtype=np.float64)
    checks = {
        "loc": abs(payload["loc"] - committed["loc"]),
        "scale": abs(payload["scale"] - committed["scale"]),
        "alpha_max_abs_diff": float(np.max(np.abs(payload["alpha"] - committed["alpha"]))),
        "prediction_max_abs_diff": float(np.max(np.abs(pred - committed_pred))),
    }

    tol = float(spec["determinism"]["expected_alpha_max_abs_diff"])
    print("reproduction diffs:", {key: f"{value:.3e}" for key, value in checks.items()})
    for key, value in checks.items():
        if value > tol:
            raise SystemExit(f"{key} = {value:.3e} exceeds tolerance {tol:.0e}")

    if args.write is not None:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        with open(args.write, "wb") as handle:
            pickle.dump(payload, handle, protocol=4)
        print(f"wrote freshly trained payload: {args.write}")

    print("PASS: untrained spec reproduces hp_noise001 within float round-off")


if __name__ == "__main__":
    main()
