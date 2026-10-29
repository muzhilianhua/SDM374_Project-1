"""Histogram gradient boosting (scikit-learn) with distance-stratified conformal std.

Defaults are member B's `histgb_low80_full` configuration (v3.2 72.35 on the frozen splits, as an
outside-framework artifact): 15 leaves, learning rate 0.05, 600 iterations, no weighting.
"""
from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import HistGradientBoostingRegressor

from ..registry import register_model
from ._gbdt import ConformalBoosted


@register_model("histgb")
class HistGradientBoosting(ConformalBoosted):
    """Histogram gradient boosting; std from distance-binned out-of-fold residuals. No gradient."""

    @dataclass(frozen=True)
    class Params:
        max_leaf_nodes: int = 15
        learning_rate: float = 0.05
        max_iter: int = 600
        l2_regularization: float = 0.0
        low_fraction: float = 0.2  # share of smallest-ramping rows that get low_weight
        low_weight: float = 1.0
        calib_folds: int = 5
        calib_bins: int = 4
        coverage: float = 0.68  # quantile of |residual| used as std; 0.68 matches one Gaussian sigma
        seed: int = 2026

    def _estimator(self):
        p = self.params
        return HistGradientBoostingRegressor(max_leaf_nodes=p.max_leaf_nodes, learning_rate=p.learning_rate,
                                             max_iter=p.max_iter, l2_regularization=p.l2_regularization,
                                             random_state=p.seed)
