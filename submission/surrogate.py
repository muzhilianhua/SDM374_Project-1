"""Navigation surrogate: fixed linear trend plus a fixed-kernel local GP.

The graded package refits this model after every successful solver call.  Kernel values are supplied
as constants and ``optimizer=None`` is intentional: fitting hyperparameters at runtime can produce
different batches on the course computer and on the development machine.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

MINOR = [0, 1, 2, 3, 6]
FLOOR = 1e-6


def features(X) -> np.ndarray:
    """Per building: r = deg/(deg+ramp), log10(deg+ramp share), log10(minor/(deg+ramp)) floored at 1e-6."""
    W = np.asarray(X, float)
    if W.size % 21:
        raise ValueError("weights must contain 21 values per point")
    W = W.reshape(-1, 3, 7)
    if not np.isfinite(W).all() or (W < 0).any():
        raise ValueError("weights must be finite and non-negative")
    sums = W.sum(axis=2, keepdims=True)
    if (sums <= 0).any():
        raise ValueError("each building must have a positive weight sum")
    W = W / sums
    s = W[:, :, 4] + W[:, :, 5]
    s_safe = np.maximum(s, FLOOR)
    r = np.where(s > 0, W[:, :, 4] / s_safe, 0.5)
    m = np.log10(np.clip(W[:, :, MINOR] / s_safe[:, :, None], FLOOR, 1e6))
    return np.concatenate([r[:, :, None], np.log10(s_safe)[:, :, None], m], axis=2).reshape(len(W), -1)


class Surrogate:
    def __init__(self, local_rows: int = 600, seed: int = 0, alpha: float = 1.0,
                 length_scale: float = 1.0, noise_level: float = 0.1):
        self.local_rows = local_rows
        self.seed = seed
        self.alpha = alpha
        self.length_scale = length_scale
        self.noise_level = noise_level
        self.fit_secs = 0.0

    def fit(self, X, y, center):
        t0 = time.time()
        F = features(X)
        y = np.asarray(y, dtype=float)
        if F.shape[0] != len(y) or len(y) < 2 or not np.isfinite(y).all():
            raise ValueError("surrogate.fit needs aligned finite labels and at least two rows")
        if (not isinstance(self.local_rows, (int, np.integer)) or self.local_rows < 2
                or not np.isfinite((self.alpha, self.length_scale, self.noise_level)).all()
                or self.alpha <= 0 or self.length_scale <= 0 or self.noise_level < 0):
            raise ValueError("invalid fixed surrogate parameters")
        self.sc = StandardScaler().fit(F)
        Z = self.sc.transform(F)
        self.trend = Ridge(alpha=self.alpha).fit(Z, y)
        res = y - self.trend.predict(Z)
        zc = self.sc.transform(features(np.asarray(center)[None]))[0]
        d = np.abs(Z - zc).max(axis=1) + 1e-3 * np.linalg.norm(Z - zc, axis=1)
        idx = np.argsort(d)[: self.local_rows]
        kernel = (ConstantKernel(1.0, constant_value_bounds="fixed")
                  * RBF(np.full(Z.shape[1], self.length_scale), length_scale_bounds="fixed")
                  + WhiteKernel(self.noise_level, noise_level_bounds="fixed"))
        self.gp = GaussianProcessRegressor(kernel, normalize_y=True, optimizer=None,
                                           random_state=self.seed, alpha=self.alpha)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gp.fit(Z[idx], res[idx])
        self.fit_secs = time.time() - t0
        return self

    def predict(self, X):
        Z = self.sc.transform(features(X))
        m, s = self.gp.predict(Z, return_std=True)
        mean = self.trend.predict(Z) + m
        return mean, np.maximum(np.asarray(s, dtype=float), 0.0)
