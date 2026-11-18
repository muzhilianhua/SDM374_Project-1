"""v7 surrogate: the fixed-constant block-additive ARD Matérn-5/2 GP of PR #24 (`hp_noise001`), refitted on all data.

Kernel = (1 - w)/3 * sum_b Matern52(W_b; ls) + w * Matern52(W; ls tiled), plus white noise. Every constant
(7 length scales, w, noise, jitter) is read from `analysis/online_models/hp_noise001_untrained/train_config.json`,
which PR #24 selected on the 2048 training rows. Nothing is optimised at run time: a fit is the target
standardisation and one Cholesky solve on the training rows plus the run's own measured points, so the
result does not depend on an optimiser's path (decision 31).

Selected offline by `Project-1/surrogate_compare/` (call-by-call replay of the T37 logs; decision-27 condition).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

SPEC = Path(__file__).resolve().parents[1] / "online_models" / "hp_noise001_untrained" / "train_config.json"


def _load_constants():
    k = json.loads(SPEC.read_text(encoding="utf-8"))["kernel"]
    return (np.asarray(k["length_scale"], float), float(k["global_weight"]), float(k["noise"]),
            float(k.get("jitter", 1e-8)))


LS, GW, NOISE, JITTER = _load_constants()


def _matern52(A, B, ls):
    a, b = A / ls, B / ls
    d2 = np.maximum((a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T, 0.0)
    r = np.sqrt(5.0 * d2)
    return (1.0 + r + (5.0 / 3.0) * d2) * np.exp(-r)


def block_cov(A, B):
    K = sum(_matern52(A[:, 7 * i:7 * i + 7], B[:, 7 * i:7 * i + 7], LS) for i in range(3)) * ((1.0 - GW) / 3.0)
    return K + GW * _matern52(A, B, np.tile(LS, 3))


def _norm(X):
    W = np.asarray(X, float).reshape(-1, 3, 7)
    return (W / W.sum(axis=2, keepdims=True)).reshape(-1, 21)


class HPSurrogate:
    """Same interface as `surrogate.Surrogate` (fit(X, y, center), predict(X) -> mean, std, fit_secs).
    `center`, `local_rows` and `seed` are accepted and unused: the model is global and deterministic."""

    def __init__(self, local_rows: int = 600, seed: int = 0, **_):
        self.fit_secs = 0.0

    def fit(self, X, y, center=None):
        t0 = time.time()
        self.Xtr = _norm(X)
        y = np.asarray(y, float)
        self.loc, self.scale = float(y.mean()), float(y.std()) or 1.0
        K = block_cov(self.Xtr, self.Xtr)
        K[np.diag_indices_from(K)] += NOISE + JITTER
        self.cf = cho_factor(K)
        self.alpha = cho_solve(self.cf, (y - self.loc) / self.scale)
        self.fit_secs = time.time() - t0
        return self

    def predict(self, X):
        Ks = block_cov(_norm(X), self.Xtr)
        v = cho_solve(self.cf, Ks.T)
        var = np.maximum(1.0 - np.einsum("ij,ji->i", Ks, v), 0.0) + NOISE
        return self.loc + self.scale * (Ks @ self.alpha), self.scale * np.sqrt(var)
