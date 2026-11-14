"""Navigation surrogate: per-building ratio features, linear trend on all data, GP on local residuals."""
from __future__ import annotations

import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

MINOR = [0, 1, 2, 3, 6]
FLOOR = 1e-6


def features(X) -> np.ndarray:
    """Per building: r = deg/(deg+ramp), log10(deg+ramp share), log10(minor/(deg+ramp)) floored at 1e-6."""
    W = np.asarray(X, float).reshape(-1, 3, 7)
    W = W / W.sum(axis=2, keepdims=True)
    s = W[:, :, 4] + W[:, :, 5]
    s_safe = np.maximum(s, FLOOR)
    r = np.where(s > 0, W[:, :, 4] / s_safe, 0.5)
    m = np.log10(np.clip(W[:, :, MINOR] / s_safe[:, :, None], FLOOR, 1e6))
    return np.concatenate([r[:, :, None], np.log10(s_safe)[:, :, None], m], axis=2).reshape(len(W), -1)


class Surrogate:
    def __init__(self, local_rows: int = 600, seed: int = 0, alpha: float = 1.0):
        self.local_rows = local_rows
        self.seed = seed
        self.alpha = alpha
        self.fit_secs = 0.0

    def fit(self, X, y, center):
        t0 = time.time()
        F = features(X)
        self.sc = StandardScaler().fit(F)
        Z = self.sc.transform(F)
        self.trend = Ridge(alpha=self.alpha).fit(Z, y)
        res = y - self.trend.predict(Z)
        zc = self.sc.transform(features(np.asarray(center)[None]))[0]
        d = np.abs(Z - zc).max(axis=1) + 1e-3 * np.linalg.norm(Z - zc, axis=1)
        idx = np.argsort(d)[: self.local_rows]
        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(np.ones(Z.shape[1]), (5e-2, 1e3)) \
            + WhiteKernel(0.1, (1e-4, 1.0))
        self.gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=0,
                                           random_state=self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.gp.fit(Z[idx], res[idx])
        self.fit_secs = time.time() - t0
        return self

    def predict(self, X):
        Z = self.sc.transform(features(X))
        m, s = self.gp.predict(Z, return_std=True)
        return self.trend.predict(Z) + m, s
