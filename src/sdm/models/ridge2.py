"""Quadratic ridge regression ported from baseline.py; tests/test_baseline_regression.py pins equivalence."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from ..interfaces import Surrogate
from ..registry import register_model
from ..simplex import DIM, as_batch


def _pipeline(alpha: float) -> Pipeline:
    return Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha, solver="cholesky")),
    ])


def _quadratic_form(pipeline: Pipeline) -> tuple[float, np.ndarray, np.ndarray]:
    """Rewrite the fitted pipeline as f(w) = c + a.w + 0.5 w.H.w."""
    poly, scale, ridge = (pipeline.named_steps[key] for key in ("poly", "scale", "ridge"))
    coefficients = ridge.coef_ / scale.scale_
    offset = float(ridge.intercept_ - coefficients @ scale.mean_)
    linear, hessian = np.zeros(DIM), np.zeros((DIM, DIM))
    for powers, coefficient in zip(poly.powers_, coefficients):
        positions = np.flatnonzero(powers)
        if powers.sum() == 1:
            linear[positions[0]] += coefficient
        elif len(positions) == 1:
            hessian[positions[0], positions[0]] += 2 * coefficient
        else:
            i, j = positions
            hessian[i, j] += coefficient
            hessian[j, i] += coefficient
    return offset, linear, hessian


@register_model("ridge2")
class QuadraticRidge(Surrogate):
    """Degree-2 polynomial features -> standardize -> ridge, alpha chosen by internal K-fold MAE."""

    @dataclass(frozen=True)
    class Params:
        alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0)
        folds: int = 5
        seed: int = 2026

    def fit(self, X, y):
        X, y = as_batch(X), np.asarray(y, dtype=float)
        folds, seed = self.params.folds, self.params.seed
        if len(y) < folds:
            raise ValueError(f"{folds}-fold CV needs at least {folds} rows")
        splits = list(KFold(n_splits=folds, shuffle=True, random_state=seed).split(X))
        best_error, best_alpha, best_oof = float("inf"), None, None
        for alpha in self.params.alphas:
            oof = np.empty(len(y))
            for fit_rows, held_out in splits:
                oof[held_out] = _pipeline(alpha).fit(X[fit_rows], y[fit_rows]).predict(X[held_out])
            error = float(mean_absolute_error(y, oof))
            if error < best_error:
                best_error, best_alpha, best_oof = error, alpha, oof
        pipeline = _pipeline(best_alpha).fit(X, y)
        self.offset, self.linear, self.hessian = _quadratic_form(pipeline)
        self.residual_std = float(np.sqrt(np.mean((best_oof - y) ** 2)))
        self.metrics = {"alpha": float(best_alpha), "cv_mae": best_error,
                        "cv_r2": float(r2_score(y, best_oof)), "cv_rmse": self.residual_std}
        return self

    def predict(self, X):
        X = as_batch(X)
        mean = self.offset + X @ self.linear + 0.5 * np.einsum("ni,ij,nj->n", X, self.hessian, X)
        # Ridge gives no per-point uncertainty; the out-of-fold RMSE is an honest constant stand-in.
        return mean, np.full(len(X), self.residual_std)

    def gradient(self, X):
        return as_batch(X) @ self.hessian + self.linear

    def smoothness(self):
        return float(np.max(np.abs(np.linalg.eigvalsh(self.hessian))))
