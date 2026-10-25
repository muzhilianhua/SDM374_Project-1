"""Extremely randomized trees; the uncertainty is the spread of the individual trees' predictions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from ..interfaces import Surrogate
from ..registry import register_model
from ..simplex import as_batch


@register_model("extratrees")
class ExtraTreesEnsemble(Surrogate):
    """Extra-trees ensemble; std is the spread across trees. Piecewise constant, so no gradient."""

    @dataclass(frozen=True)
    class Params:
        n_estimators: int = 300
        min_samples_leaf: int = 2
        max_features: float = 0.5
        seed: int = 2026

    def fit(self, X, y):
        p = self.params
        self.forest = ExtraTreesRegressor(
            n_estimators=p.n_estimators, min_samples_leaf=p.min_samples_leaf, max_features=p.max_features,
            random_state=p.seed, n_jobs=1,
        ).fit(as_batch(X), np.asarray(y, dtype=float))
        self.metrics = {"n_estimators": float(p.n_estimators)}
        return self

    def predict(self, X):
        X = as_batch(X)
        per_tree = np.stack([tree.predict(X) for tree in self.forest.estimators_])
        # Tree spread over-covers the out-of-fold error (std_coverage ~0.87 vs 0.68 ideal); left uncalibrated.
        return per_tree.mean(axis=0), per_tree.std(axis=0)
