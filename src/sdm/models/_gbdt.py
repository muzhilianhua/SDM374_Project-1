"""Helpers shared by the gradient-boosted tree plugins (`histgb`, `xgboost`); not a plugin itself.

Boosted trees give no per-point uncertainty, so their std comes from out-of-fold residuals, stratified by
how far a point lies from the data (split conformal with distance bins). Points far from every measured row
get the wider residual spread observed for far points, which is what the optimizers' risk term needs.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.model_selection import KFold

from ..evaluate import nearest_distances
from ..interfaces import Surrogate
from ..simplex import as_batch


def low_target_weights(y: np.ndarray, fraction: float, weight: float) -> np.ndarray:
    """Sample weights: `weight` for the floor(fraction * n) smallest targets, 1 for the rest."""
    weights = np.ones(len(y))
    weights[np.argsort(y, kind="stable")[:int(np.floor(fraction * len(y)))]] = weight
    return weights


class DistanceConformal:
    """Per-point std: a quantile of |out-of-fold residual| within bins of distance to the training rows."""

    def fit(self, fit_predict: Callable[[np.ndarray, np.ndarray], np.ndarray], X: np.ndarray, y: np.ndarray, *,
            folds: int, bins: int, coverage: float, seed: int) -> DistanceConformal:
        """`fit_predict(train_rows, test_rows)` fits on train_rows and returns predictions for test_rows."""
        residual, distance = np.empty(len(y)), np.empty(len(y))
        for train, test in KFold(n_splits=folds, shuffle=True, random_state=seed).split(X):
            residual[test] = np.abs(y[test] - fit_predict(train, test))
            # Measured against this fold's training part (80% of the rows), whereas predict measures against
            # all rows; calibration points therefore look slightly farther away and std errs on the wide side.
            distance[test] = nearest_distances(X[test], X[train])
        self.edges_ = np.quantile(distance, np.linspace(0, 1, bins + 1)[1:-1])
        bin_of = np.searchsorted(self.edges_, distance, side="right")
        spread = [np.quantile(residual[bin_of == b], coverage) if np.any(bin_of == b) else 0.0 for b in range(bins)]
        # Never let a farther bin claim more confidence than a nearer one.
        self.std_ = np.maximum.accumulate(spread)
        self.X_ = X
        return self

    def std(self, points: np.ndarray) -> np.ndarray:
        # Points beyond the last edge fall into the farthest bin.
        return self.std_[np.searchsorted(self.edges_, nearest_distances(points, self.X_), side="right")]

    @property
    def metrics(self) -> dict[str, float]:
        return {**{f"conformal_edge_{i}": float(e) for i, e in enumerate(self.edges_)},
                **{f"conformal_std_{i}": float(s) for i, s in enumerate(self.std_)}}


class ConformalBoosted(Surrogate):
    """Base for boosted-tree plugins: optional low-target weighting plus DistanceConformal std.

    Subclasses define `Params` (with low_fraction, low_weight, calib_folds, calib_bins, coverage, seed)
    and `_estimator()`, which returns a fresh unfitted regressor accepting `sample_weight`.
    """

    def _estimator(self):
        raise NotImplementedError

    def fit(self, X, y):
        p = self.params
        X, y = as_batch(X), np.asarray(y, dtype=float)
        weights = low_target_weights(y, p.low_fraction, p.low_weight)

        def fit_predict(train, test):
            return self._estimator().fit(X[train], y[train], sample_weight=weights[train]).predict(X[test])

        self.conformal = DistanceConformal().fit(fit_predict, X, y, folds=p.calib_folds, bins=p.calib_bins,
                                                 coverage=p.coverage, seed=p.seed)
        self.estimator = self._estimator().fit(X, y, sample_weight=weights)
        self.metrics = {"weighted_rows": float(np.sum(weights != 1)), **self.conformal.metrics}
        return self

    def predict(self, X):
        X = as_batch(X)
        return np.asarray(self.estimator.predict(X), dtype=float), self.conformal.std(X)
