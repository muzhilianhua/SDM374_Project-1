"""The unchanged best stacking mean with label-excluded nested residual calibration."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import KFold

from ..interfaces import Surrogate
from ..registry import register_model
from ..simplex import as_batch
from ._gbdt import DistanceConformal
from .method_study import MethodStudy


@register_model("honest-stack")
class HonestStack(Surrogate):
    """Unchanged OOF-stack mean; each nested calibration label is excluded from its predictive fit."""

    @dataclass(frozen=True)
    class Params:
        iterations: int = 600
        calib_folds: int = 3
        calib_bins: int = 4
        coverage: float = .68
        fit_workers: int = 3
        seed: int = 2026

    def _core(self):
        return MethodStudy(MethodStudy.Params(family="ensemble", ensemble_mode="stack",
                           iterations=self.params.iterations, seed=self.params.seed))

    def fit(self, X, y):
        X, y = as_batch(X), np.asarray(y, dtype=float)
        p = self.params
        if (p.iterations < 1 or p.calib_folds < 2 or p.calib_bins < 1 or p.fit_workers < 1
                or not 0 < p.coverage < 1 or len(y) < max(12, 3*p.calib_folds)
                or y.shape != (len(X),) or not np.isfinite(y).all()):
            raise ValueError("invalid honest-stack settings or training observations")
        folds = list(KFold(p.calib_folds, shuffle=True, random_state=p.seed).split(X))

        def held_out(fold):
            train, test = fold
            return self._core().fit(X[train], y[train]).predict(X[test])[0]

        with ThreadPoolExecutor(max_workers=p.fit_workers) as pool:
            values = list(pool.map(held_out, folds))
        self.calibration_prediction_ = np.empty(len(y))
        lookup = {}
        for (train, test), predictions in zip(folds, values):
            self.calibration_prediction_[test] = predictions
            lookup[tuple(test)] = (train, predictions)

        def cached(train, test):
            recorded_train, predictions = lookup[tuple(test)]
            np.testing.assert_array_equal(train, recorded_train)
            return predictions

        self.conformal = DistanceConformal().fit(cached, X, y, folds=p.calib_folds,
            bins=p.calib_bins, coverage=p.coverage, seed=p.seed)
        self.predictor_ = self._core().fit(X, y)
        self.metrics = {**self.predictor_.metrics,
            **{f"honest_{key}": v for key, v in self.conformal.metrics.items()},
            "nested_calibration_mae": float(np.mean(abs(y-self.calibration_prediction_))),
            "mean_recipe_unchanged": True, "error_scale_is_heuristic": True,
            "calibration_labels_excluded_from_predictive_fit": True}
        return self

    def predict(self, X):
        X = as_batch(X)
        mean = self.predictor_.predict(X)[0]
        return mean, self.conformal.std(X)
