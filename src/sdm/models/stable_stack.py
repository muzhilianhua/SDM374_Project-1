"""Subsample-regularized convex blending with nested, label-excluded residual calibration."""
from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from catboost import CatBoostRegressor
import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from ..evaluate import median_spacing, nearest_distances
from ..interfaces import Surrogate
from ..registry import register_model
from ..simplex import as_batch
from ._gbdt import low_target_weights


@dataclass
class BlendCore:
    members: list
    weights: np.ndarray
    floor: float
    optimization: dict | None = None

    def member_predictions(self, X):
        return np.stack([np.column_stack([m.predict(X) for m in group]) @ self.weights
                         for group in self.members])

    def predict(self, X):
        values = self.member_predictions(X)
        return values.mean(axis=0), np.sqrt(values.var(axis=0, ddof=1) + self.floor**2)


def checked_blend(result, prior, objective):
    """Keep a usable SLSQP iterate, otherwise use the feasible prior and record why."""
    weights = np.asarray(result.x, dtype=float)
    feasible = (weights.shape == prior.shape and np.isfinite(weights).all()
                and np.all(weights >= -1e-9) and np.all(weights <= 1 + 1e-9)
                and abs(weights.sum() - 1) < 1e-8)
    prior_loss = float(objective(prior))
    loss = None
    if feasible:
        weights = np.maximum(weights, 0)
        weights /= weights.sum()
        loss = float(objective(weights))
        feasible = np.isfinite(loss) and loss <= prior_loss + 1e-10 * max(1., abs(prior_loss))
    accepted = bool(feasible)
    diagnostics = {"success": bool(result.success), "message": str(result.message),
        "iterations": int(getattr(result, "nit", 0)), "accepted_iterate": accepted,
        "used_prior": not accepted, "prior_objective": prior_loss,
        "returned_objective": loss if loss is not None and np.isfinite(loss) else None}
    return (weights if accepted else prior.copy()), diagnostics


@register_model("stable-stack")
class StableStack(Surrogate):
    """Five subsample blends; stability-regularized weights and nested distance-binned error scales."""

    @dataclass(frozen=True)
    class Params:
        profile: str = "mild"
        iterations: int = 600
        bags: int = 5
        subsample: float = .9
        inner_folds: int = 3
        calib_folds: int = 3
        calib_bins: int = 4
        coverage: float = .68
        low_fraction: float = .2
        low_weight: float = 3.0
        stability: float = .5
        shrinkage: float = .02
        fit_workers: int = 3
        seed: int = 2026

    def _fit_family(self, family, X, y, seed):
        p = self.params
        strong = p.profile == "strong"
        if family == 0:
            model = XGBRegressor(n_estimators=p.iterations, learning_rate=.1,
                max_depth=2 if strong else 3, min_child_weight=10 if strong else 5,
                reg_lambda=20 if strong else 10, subsample=.9, tree_method="hist",
                objective="reg:squarederror", n_jobs=1, random_state=seed)
        elif family == 1:
            model = CatBoostRegressor(iterations=p.iterations, learning_rate=.05,
                depth=4 if strong else 5, l2_leaf_reg=20 if strong else 10,
                loss_function="RMSE", border_count=128, random_seed=seed,
                bootstrap_type="Bayesian", bagging_temperature=1, random_strength=1,
                thread_count=1, verbose=False, allow_writing_files=False)
        else:
            model = HistGradientBoostingRegressor(max_iter=p.iterations, learning_rate=.05,
                max_leaf_nodes=7 if strong else 15, min_samples_leaf=40 if strong else 30,
                l2_regularization=5 if strong else 1, random_state=seed)
        weights = (np.ones(len(y)) if family == 2 else
                   low_target_weights(y, p.low_fraction, p.low_weight))
        return model.fit(X, y, sample_weight=weights)

    def _fit_core(self, X, y, seed):
        """Every choice here uses only this core's supplied training rows and labels."""
        p = self.params
        oof = np.empty((len(y), 3))
        groups, bag_predictions = [], []
        rng = np.random.default_rng(seed + 101)
        size = max(p.inner_folds + 1, int(np.floor(p.subsample * len(y))))
        # Each fit has its own estimator and seed. Parallelism changes scheduling only.
        with ThreadPoolExecutor(max_workers=p.fit_workers) as pool:
            def fit_group(rows, group_seed):
                return list(pool.map(lambda family: self._fit_family(
                    family, X[rows], y[rows], group_seed), range(3)))

            for fold, (train, test) in enumerate(KFold(p.inner_folds, shuffle=True,
                                                      random_state=seed).split(X)):
                group = fit_group(train, seed + 31*fold)
                oof[test] = np.column_stack([m.predict(X[test]) for m in group])
            for bag in range(p.bags):
                rows = np.sort(rng.choice(len(y), size=size, replace=False))
                group = fit_group(rows, seed + 1009*(bag+1))
                groups.append(group)
                bag_predictions.append(np.column_stack([m.predict(X) for m in group]))
        bag_predictions = np.stack(bag_predictions)
        centered = bag_predictions - bag_predictions.mean(axis=0)
        sample_weights = low_target_weights(y, p.low_fraction, p.low_weight)
        sample_weights /= sample_weights.sum()
        scale = max(float(y.std()), 1e-8)
        A, target = oof / scale, y / scale
        # Train-input variation is a regularizer, never presented as a held-out error estimate.
        variation = np.einsum("bnf,bng,n->fg", centered, centered, sample_weights)
        variation /= (p.bags - 1) * scale**2
        prior = np.array([.5, .3, .2])

        def objective(w):
            return float(sample_weights @ np.square(A @ w - target)
                         + p.stability * (w @ variation @ w)
                         + p.shrinkage * np.square(w - prior).sum())

        def gradient(w):
            return (2*A.T @ (sample_weights * (A @ w - target))
                    + 2*p.stability * variation @ w + 2*p.shrinkage * (w-prior))

        result = minimize(objective, prior, jac=gradient, method="SLSQP",
            bounds=[(0, 1)]*3,
            constraints={"type": "eq", "fun": lambda w: w.sum()-1,
                         "jac": lambda w: np.ones(3)},
            options={"ftol": 1e-12, "maxiter": 200})
        weights, diagnostics = checked_blend(result, prior, objective)
        return BlendCore(groups, weights, max(.02*scale, 1e-6), diagnostics)

    def fit(self, X, y):
        p = self.params
        X, y = as_batch(X), np.asarray(y, dtype=float)
        if (p.profile not in {"mild", "strong"} or p.iterations < 1 or p.bags < 2
                or not 0 < p.subsample <= 1 or p.inner_folds < 2 or p.calib_folds < 2
                or p.calib_bins < 1 or not 0 < p.coverage < 1
                or not 0 <= p.low_fraction <= 1 or p.low_weight <= 0
                or p.stability < 0 or p.shrinkage < 0 or p.fit_workers < 1):
            raise ValueError("invalid stable-stack parameters")
        if (y.shape != (len(X),) or not np.isfinite(y).all()
                or len(y) < max(12, 3*p.inner_folds, 3*p.calib_folds)):
            raise ValueError("stable-stack needs finite aligned labels and enough calibration rows")
        predicted, raw_scale, ratio = np.empty(len(y)), np.empty(len(y)), np.empty(len(y))
        self.calibration_folds_ = []
        for fold, (train, test) in enumerate(KFold(p.calib_folds, shuffle=True,
                                                  random_state=p.seed).split(X)):
            core = self._fit_core(X[train], y[train], p.seed + 7919*(fold+1))
            predicted[test], raw_scale[test] = core.predict(X[test])
            ratio[test] = nearest_distances(X[test], X[train]) / max(median_spacing(X[train]), 1e-12)
            self.calibration_folds_.append({"train": train, "test": test,
                                            "blend_weights": core.weights,
                                            "optimization": core.optimization})
        self.calibration_prediction_ = predicted
        self.calibration_raw_scale_ = raw_scale
        self.calibration_distance_ratio_ = ratio
        self.edges_ = np.quantile(ratio, np.linspace(0, 1, p.calib_bins+1)[1:-1])
        bins = np.searchsorted(self.edges_, ratio, side="right")
        normalized = np.abs(y-predicted) / raw_scale
        fallback = float(np.quantile(normalized, p.coverage))
        quantiles = [float(np.quantile(normalized[bins == b], p.coverage))
                     if np.any(bins == b) else fallback for b in range(p.calib_bins)]
        self.quantiles_ = np.maximum.accumulate(quantiles)
        self.max_calibration_ratio_ = max(float(ratio.max()), 1e-12)
        self.core_ = self._fit_core(X, y, p.seed)
        self.X_ = X.copy()
        self.spacing_ = max(median_spacing(X), 1e-12)
        calibrated = raw_scale * self.quantiles_[bins]
        self.metrics = {"training_rows": len(y), "bags": p.bags,
            "blend_optimization": self.core_.optimization,
            "calibration_blend_optimizations": [f["optimization"] for f in self.calibration_folds_],
            "calibration_mae": float(np.mean(abs(y-predicted))),
            "calibration_coverage": float(np.mean(abs(y-predicted) <= calibrated)),
            "calibration_labels_excluded_from_predictive_fit": True,
            "error_scale_is_heuristic": True,
            **{f"weight_{family}": float(w) for family, w in
               zip(("xgboost", "catboost", "histgb"), self.core_.weights)},
            **{f"calibration_multiplier_{i}": float(q) for i, q in enumerate(self.quantiles_)}}
        return self

    def predict_members(self, X):
        """Training-subsample predictions, for diagnostics; not independent model votes."""
        return self.core_.member_predictions(as_batch(X))

    def predict(self, X):
        X = as_batch(X)
        mean, raw_scale = self.core_.predict(X)
        ratio = nearest_distances(X, self.X_) / self.spacing_
        multiplier = self.quantiles_[np.searchsorted(self.edges_, ratio, side="right")]
        # Extend conservatism beyond calibration support without claiming interval validity there.
        extrapolation = np.maximum(1., ratio / self.max_calibration_ratio_)
        return mean, raw_scale * multiplier * extrapolation
