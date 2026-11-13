# -*- coding: utf-8 -*-
"""Numpy-only inference kernel for the hp_noise001 model bundle (numpy + scipy, no scikit-learn).

Why implement the kernel ourselves? The training code uses a custom scikit-learn GPR kernel subclass;
pickling the sklearn estimator directly pins the loader to that package path and sklearn version. We
instead serialise the *fitted state* (training rows, alpha vector, length scales, noise, sample
weights) as plain numpy arrays, so the loader needs only numpy and scipy.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# ───────────────────────────── Hard contract ─────────────────────────────
METRICS = ("cost", "carbon", "peak", "discomfort", "degradation", "ramping", "resilience")
FEATURES = tuple(f"b{b}_{m}" for b in (1, 2, 3) for m in METRICS)
N_FEATURES = 21
N_BUILDINGS = 3
N_METRICS = 7

SCHEMA = "sdm374.model_library.v1"


class ContractError(ValueError):
    """Input violates the 3x7 / 21-dim hard contract or the simplex constraint."""


def pack_weights(obj):
    """Pack 3x7 weights (or one row) or a 21-dim vector into an (n, 21) matrix in canonical order.

    Canonical order: building 1 -> building 3; within each building, cost -> carbon -> peak ->
    discomfort -> degradation -> ramping -> resilience.

    Accepts: nested list, np.ndarray with shape (3,7), (21,), (1,21) or (n,21). Any other shape
    raises ContractError — fail loudly rather than silently mispredict.
    """
    arr = np.asarray(obj, float)

    if arr.ndim == 1:
        if arr.size != N_FEATURES:
            raise ContractError(f"21-dim vector required, got {arr.size}")
        arr = arr.reshape(1, N_FEATURES)
    elif arr.ndim == 2:
        if arr.shape == (N_BUILDINGS, N_METRICS):
            arr = arr.reshape(1, N_FEATURES)
        elif arr.shape[1] == N_FEATURES:
            pass
        elif arr.shape == (N_METRICS, N_BUILDINGS):
            raise ContractError("Expected (3,7) not (7,3); rows = buildings, cols = metrics")
        else:
            raise ContractError(f"2-dim input must be (3,7) or (n,21); got {arr.shape}")
    else:
        raise ContractError(f"Unsupported ndim = {arr.ndim}")

    if not np.all(np.isfinite(arr)):
        raise ContractError("NaN or inf in weights")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _check_simplex(X: np.ndarray) -> None:
    """Enforce Decision 21: every building's 7 weights must be non-negative and sum to 1.

    The GP kernel is NOT scale-invariant — the same normalised point with a different proportional
    scaling of one building's weights produces a different prediction (README §4 example). We
    therefore reject any input that fails the simplex check instead of silently mispredicting.

    Tolerance is 1e-9, matching `sdm.simplex.check_weights`.
    """
    tol = 1e-9
    for b in range(N_BUILDINGS):
        block = X[:, 7 * b: 7 * b + 7]
        row_sums = block.sum(axis=1)
        row_sum_err = np.abs(row_sums - 1.0)
        neg_err = np.maximum(0.0, -block.min(axis=1))
        bad = np.where((row_sum_err > tol) | (neg_err > tol))[0]
        if bad.size:
            raise ContractError(
                f"weights fail simplex check: building {b + 1}, rows {bad.tolist()} sum != 1 "
                f"(max |sum-1|={row_sum_err.max():.3e}) or contain negatives (max neg={neg_err.max():.3e}). "
                "Normalise each building's 7 weights to sum to 1."
            )


# ───────────────────────────── Matérn-5/2 (sklearn convention) ─────────────────────────────
def _scale_sqdist(X, Y, ls):
    A = X / ls
    B = Y / ls
    d2 = np.sum(A * A, axis=1)[:, None] + np.sum(B * B, axis=1)[None, :] - 2.0 * (A @ B.T)
    np.maximum(d2, 0.0, out=d2)
    return d2


def matern52(X, Y, ls):
    """Covariance identical to sklearn.gaussian_process.kernels.Matern(nu=2.5, ARD)."""
    d2 = _scale_sqdist(X, Y, np.asarray(ls, float))
    r = np.sqrt(5.0 * d2)
    return (1.0 + r + (5.0 / 3.0) * d2) * np.exp(-r)


def block_ard_cov(X, Y, length_scale, global_weight):
    """Block-wise ARD Matérn-5/2 + full-region interaction. Equivalent to the training
    `BlockARDKernel`."""
    ls = np.asarray(length_scale, float)
    w = float(global_weight)
    parts = None
    for b in range(N_BUILDINGS):
        k = matern52(X[:, 7 * b:7 * b + 7], Y[:, 7 * b:7 * b + 7], ls)
        parts = k if parts is None else parts + k
    K = parts * ((1.0 - w) / N_BUILDINGS)
    K += w * matern52(X, Y, np.tile(ls, N_BUILDINGS))
    return K


class ModelBundle:
    """Loaded model bundle — GP only (numpy + scipy inference)."""

    def __init__(self, payload, path=None):
        self.payload = payload
        self.path = Path(path) if path else None
        self._L = None
        self._validate()

    def _validate(self):
        p = self.payload
        if p.get("schema") != SCHEMA:
            raise ContractError(f"Unknown model schema: {p.get('schema')}")
        if tuple(p["feature_order"]) != FEATURES:
            raise ContractError(
                "model feature_order does not match hard contract — refusing to load"
            )
        if p["kind"] != "gp":
            raise ContractError(f"Unknown kind: {p['kind']}; only 'gp' is supported here")
        if p["X_train"].shape[1] != N_FEATURES:
            raise ContractError("X_train must have 21 columns")
        if len(p["alpha"]) != len(p["X_train"]):
            raise ContractError("alpha length must match training-set rows")
        if len(p["length_scale"]) != N_METRICS:
            raise ContractError("length_scale must have 7 entries")
        if len(p["weights"]) != len(p["X_train"]):
            raise ContractError("sample_weights length must match training-set rows")

    # ── GP core ──
    def _gp_mean(self, Xq):
        g = self.payload
        Ks = block_ard_cov(g["X_train"], Xq,
                           g["length_scale"], g["global_weight"])
        mz = g["loc"] + g["scale"] * (Ks.T @ g["alpha"])
        if g["target"] == "log":
            return np.exp(np.clip(mz, -20.0, 20.0)), Ks, mz
        return mz, Ks, mz

    def _gp_chol(self):
        g = self.payload
        if self._L is None:
            K = block_ard_cov(g["X_train"], g["X_train"],
                              g["length_scale"], g["global_weight"])
            K[np.diag_indices_from(K)] += (g["noise"] + g["jitter"]) / g["weights"]
            self._L = np.linalg.cholesky(K)
        return self._L

    def _gp_std(self, Xq, Ks, mz):
        from scipy.linalg import solve_triangular
        g = self.payload
        L = self._gp_chol()
        Z = solve_triangular(L, Ks, lower=True)
        v = 1.0 - np.sum(Z * Z, axis=0)
        sd_z = g["scale"] * np.sqrt(np.maximum(v + g["noise"], 1e-7))
        if g["target"] == "log":
            return np.exp(np.clip(mz, -20.0, 20.0)) * sd_z
        return sd_z

    def predict(self, Xq, return_std=True):
        """Predict R for normalised weights.

        Each building's 7 weights must be non-negative and sum to 1 (Decision 21). The GP
        kernel is NOT scale-invariant: rescaling a single building's weights without renormalising
        changes the prediction for the same normalised point.
        """
        Xq = pack_weights(Xq)
        _check_simplex(Xq)

        mean, Ks, mz = self._gp_mean(Xq)
        if not return_std:
            return mean
        return mean, self._gp_std(Xq, Ks, mz)


def load_model(model_dir):
    """Load `<bundle_dir>/model.npz` (or legacy `.pkl` for backward compatibility)."""
    model_dir = Path(model_dir)
    npz = model_dir / "model.npz"
    pkl = model_dir / "model.pkl"

    if npz.exists():
        data = np.load(npz, allow_pickle=False)
        payload = dict(data)
        # Scalars saved as 0-d arrays; convert back to Python floats/ints/strings.
        scalars = ("kind", "target", "schema", "global_weight", "noise", "jitter",
                   "loc", "scale", "train_rows")
        for key in scalars:
            if key in payload and np.ndim(payload[key]) == 0:
                payload[key] = payload[key].item()
        payload["feature_order"] = tuple(
            str(x) for x in payload["feature_order"].tolist())
        return ModelBundle(payload, path=npz)

    if pkl.exists():
        import pickle
        with open(pkl, "rb") as handle:
            payload = pickle.load(handle)
        return ModelBundle(payload, path=pkl)

    raise FileNotFoundError(f"Neither model.npz nor model.pkl found in {model_dir}")


def describe():
    return {"feature_order": list(FEATURES), "n_features": N_FEATURES, "schema": SCHEMA}


# ─────────────────────────────── .npz serializer ───────────────────────────────
def save_payload(payload: dict, path: Path) -> None:
    """Serialise a payload dict of arrays and scalars to .npz (no pickle).

    Usage:
        save_payload(payload, Path("bundle/model.npz"))
    """
    out = {}
    for key, value in payload.items():
        if key == "feature_order":
            out[key] = np.asarray(list(value), dtype=np.str_)
        elif np.isscalar(value):
            if isinstance(value, bool):
                out[key] = np.asarray(value, dtype=np.bool_)
            elif isinstance(value, (int, np.integer)):
                out[key] = np.asarray(value, dtype=np.int64)
            elif isinstance(value, float):
                out[key] = np.asarray(value, dtype=np.float64)
            else:
                out[key] = np.asarray(str(value), dtype=np.str_)
        elif isinstance(value, str):
            out[key] = np.asarray(value, dtype=np.str_)
        else:
            arr = np.asarray(value)
            # Promote object dtypes that hold scalars to their primitive counterparts.
            if arr.dtype == object and arr.ndim == 0:
                try:
                    arr = np.asarray(arr.item())
                except Exception:
                    pass
            out[key] = arr
    np.savez(path, **out)
