"""Gaussian process fitted only on the training rows nearest the best observed row."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from ..interfaces import Surrogate
from ..registry import register_model
from ..simplex import as_batch


@register_model("local-gp")
class LocalGaussianProcess(Surrogate):
    """Matern ARD Gaussian process on the rows nearest the best observed row; a local judge, not a global model."""

    @dataclass(frozen=True)
    class Params:
        neighbors: int = 800
        nu: float = 2.5
        length_start: float = 0.5
        length_min: float = 0.03
        length_max: float = 10.0
        seed: int = 0

    def fit(self, X, y):
        p = self.params
        X, y = as_batch(X), np.asarray(y, dtype=float)
        # One length scale per weight, no additive-by-building assumption: an independent check on the
        # additive ARD model in the only region the offline search cares about.
        self.anchor_ = X[np.argmin(y)]
        distance = np.linalg.norm(X - self.anchor_, axis=1)
        near = np.argsort(distance, kind="stable")[:min(p.neighbors, len(y))]
        kernel = (ConstantKernel(30.0, (1e-2, 1e4))
                  * Matern([p.length_start] * X.shape[1], (p.length_min, p.length_max), nu=p.nu)
                  + WhiteKernel(1.0, (1e-4, 1e2)))
        self.estimator_ = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=0,
                                                   random_state=p.seed).fit(X[near], y[near])
        fitted = self.estimator_.kernel_
        self.metrics = {"local_rows": float(len(near)),
                        "local_radius": float(distance[near].max()),
                        "fitted_noise": float(fitted.k2.noise_level),
                        "fitted_min_length": float(np.min(fitted.k1.k2.length_scale)),
                        "log_marginal_likelihood": float(self.estimator_.log_marginal_likelihood_value_)}
        return self

    def predict(self, X):
        mean, std = self.estimator_.predict(as_batch(X), return_std=True)
        return mean, np.maximum(std, 0.0)
