"""XGBoost with low-target weighting and distance-stratified conformal std.

Defaults are member B's `xgb_w20x3_full` configuration (v3.2 72.34 on the frozen splits, as an
outside-framework artifact): depth 3, learning rate 0.1, 600 trees, subsample 0.9, and the 20% smallest
ramping rows weighted 3x. On macOS xgboost needs the OpenMP runtime: `brew install libomp`.
"""
from __future__ import annotations

from dataclasses import dataclass

from xgboost import XGBRegressor

from ..registry import register_model
from ._gbdt import ConformalBoosted


@register_model("xgboost")
class XGBoost(ConformalBoosted):
    """XGBoost, 20% lowest-ramping rows weighted 3x; std from distance-binned out-of-fold residuals."""

    @dataclass(frozen=True)
    class Params:
        max_depth: int = 3
        learning_rate: float = 0.1
        n_estimators: int = 600
        subsample: float = 0.9
        low_fraction: float = 0.2  # share of smallest-ramping rows that get low_weight
        low_weight: float = 3.0
        calib_folds: int = 5
        calib_bins: int = 4
        coverage: float = 0.68  # quantile of |residual| used as std; 0.68 matches one Gaussian sigma
        seed: int = 2026

    def _estimator(self):
        p = self.params
        # One thread keeps fits bit-identical across runs, which `sdm reproduce` relies on.
        return XGBRegressor(objective="reg:squarederror", tree_method="hist", max_depth=p.max_depth,
                            learning_rate=p.learning_rate, n_estimators=p.n_estimators, subsample=p.subsample,
                            random_state=p.seed, n_jobs=1)
