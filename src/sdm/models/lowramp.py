"""Full-support target weighting with training-only inner selection for boosted surrogates."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor

from ..frozen_v32 import load_scorer
from ..interfaces import Surrogate
from ..registry import register_model
from ..simplex import as_batch
from ._gbdt import DistanceConformal


def training_weights(y, strategy, alpha, fraction, temperature):
    """Return a training-row mask and weights, using only these rows' targets.

    Soft weights are 1 + alpha * sigmoid((quantile(y, fraction) - y) /
    (temperature * population_std(y))). Their range is [1, 1 + alpha].
    Neither soft nor step weighting removes observations. Trimmed-step retains exactly floor(.9*n).
    """
    y = np.asarray(y, dtype=float)
    if strategy not in {"unweighted", "step", "trimmed-step", "soft"}:
        raise ValueError("unknown target-weighting strategy")
    if not (0 < fraction < 1 and np.isfinite(alpha) and alpha >= 0
            and np.isfinite(temperature) and temperature > 0):
        raise ValueError("require 0 < fraction < 1, finite alpha >= 0 and temperature > 0")
    mask, weights = np.ones(len(y), dtype=bool), np.ones(len(y))
    order = np.argsort(y, kind="stable")
    if strategy == "soft":
        scale = max(float(np.std(y)), np.finfo(float).eps)
        weights += alpha * expit((np.quantile(y, fraction) - y) / (temperature * scale))
    elif strategy in {"step", "trimmed-step"}:
        weights[order[:int(np.floor(fraction * len(y)))]] += alpha
        if strategy == "trimmed-step":
            mask[order[int(np.floor(0.9 * len(y))):]] = False
    return mask, weights


def choose_trial(trials, global_limit=1.05, local_limit=1.10, tie_points=0.25):
    """Keep near-best inner score, prefer weaker weighting; unweighted is a selectable fallback."""
    baseline = trials[0]
    eligible = [row for row in trials
                if row["global_mae"] <= baseline["global_mae"] * global_limit
                and row["local_mae"] <= baseline["local_mae"] * local_limit]
    best = max(row["score"] for row in eligible)
    near_best = [row for row in eligible if row["score"] >= best - tie_points]
    return min(near_best, key=lambda row: (row["alpha"], -row["temperature"], -row["score"]))


@register_model("lowramp-boosted")
class LowRampingBoosted(Surrogate):
    """HistGB/XGBoost with full-support sigmoid target weights and optional nested inner selection."""

    @dataclass(frozen=True)
    class Params:
        family: str = "xgboost"
        strategy: str = "soft"
        tune: bool = False
        alpha: float = 1.0
        fraction: float = 0.2
        temperature: float = 0.3
        alpha_grid: tuple[float, ...] = (1.0, 3.0)
        temperature_grid: tuple[float, ...] = (0.1, 0.3)
        inner_folds: int = 3
        global_mae_limit: float = 1.05
        local_mae_limit: float = 1.10
        tie_points: float = 0.25
        max_depth: int = 3
        max_leaf_nodes: int = 15
        learning_rate: float = 0.1
        iterations: int = 600
        subsample: float = 0.9
        l2_regularization: float = 0.0
        calib_folds: int = 5
        calib_bins: int = 4
        coverage: float = 0.68
        seed: int = 2026

    def _estimator(self):
        p = self.params
        if p.family == "xgboost":
            return XGBRegressor(objective="reg:squarederror", tree_method="hist",
                                max_depth=p.max_depth, learning_rate=p.learning_rate,
                                n_estimators=p.iterations, subsample=p.subsample,
                                random_state=p.seed, n_jobs=1)
        if p.family == "histgb":
            return HistGradientBoostingRegressor(max_leaf_nodes=p.max_leaf_nodes,
                learning_rate=p.learning_rate, max_iter=p.iterations,
                l2_regularization=p.l2_regularization, random_state=p.seed)
        raise ValueError("family must be histgb or xgboost")

    def _fit_estimator(self, X, y, strategy, alpha, temperature):
        mask, weights = training_weights(y, strategy, alpha, self.params.fraction, temperature)
        return self._estimator().fit(X[mask], y[mask], sample_weight=weights[mask])

    def select(self, X, y):
        """Select only inside the supplied training data; outer labels are never passed here."""
        p = self.params
        scorer, policy = load_scorer()
        low = np.asarray(y) <= policy.fixed_cutoff
        if min(int(low.sum()), int((~low).sum())) < 2 * p.inner_folds:
            raise ValueError("inner v3.2 selection needs at least two low/high rows per fold")
        folds = list(StratifiedKFold(n_splits=p.inner_folds, shuffle=True,
                                     random_state=p.seed + 101).split(X, low))
        candidates = [(0.0, p.temperature)] + [(a, t) for a in p.alpha_grid for t in p.temperature_grid]
        self.trials_ = []
        for alpha, temperature in candidates:
            stats = []
            for train, test in folds:
                prediction = self._fit_estimator(X[train], y[train], "soft", alpha, temperature).predict(X[test])
                context = scorer.make_context(y[train], policy)
                result = scorer.evaluate(y[test], prediction, context, policy,
                                         np.array([f"inner-{i:06d}" for i in test]))
                stats.append([result[key] for key in ("score", "global_mae", "local_mae")])
            score, global_mae, local_mae = np.mean(stats, axis=0)
            self.trials_.append({"alpha": float(alpha), "temperature": float(temperature),
                "score": float(score), "global_mae": float(global_mae), "local_mae": float(local_mae)})
        return choose_trial(self.trials_, p.global_mae_limit, p.local_mae_limit, p.tie_points)

    def fit(self, X, y):
        p = self.params
        X, y = as_batch(X), np.asarray(y, dtype=float)
        if y.shape != (len(X),) or not np.isfinite(y).all():
            raise ValueError("expected one finite target per input row")
        if p.tune and p.strategy != "soft":
            raise ValueError("inner selection is only defined for soft weighting")
        if p.global_mae_limit < 1 or p.local_mae_limit < 1 or p.tie_points < 0:
            raise ValueError("MAE limits must be >=1 and tie_points >=0")
        self.trials_ = []
        chosen = self.select(X, y) if p.tune else {"alpha": p.alpha, "temperature": p.temperature}
        self.selected_ = chosen
        alpha, temperature = chosen["alpha"], chosen["temperature"]

        def fit_predict(train, test):
            # Recompute cutoff, scale, trimming and weights inside each calibration training fold.
            return self._fit_estimator(X[train], y[train], p.strategy, alpha, temperature).predict(X[test])

        self.conformal = DistanceConformal().fit(fit_predict, X, y, folds=p.calib_folds,
            bins=p.calib_bins, coverage=p.coverage, seed=p.seed)
        self.estimator = self._fit_estimator(X, y, p.strategy, alpha, temperature)
        mask, weights = training_weights(y, p.strategy, alpha, p.fraction, temperature)
        self.metrics = {"selected_alpha": float(alpha), "selected_temperature": float(temperature),
            "training_rows": int(mask.sum()), "dropped_rows": int((~mask).sum()),
            "weight_min": float(weights[mask].min()), "weight_max": float(weights[mask].max()),
            "effective_sample_size": float(weights[mask].sum() ** 2 / np.square(weights[mask]).sum()),
            "cutoff": float(np.quantile(y, p.fraction)), "target_std": float(np.std(y)),
            "inner_selected_score": float(chosen.get("score", 0.0)), **self.conformal.metrics}
        return self

    def predict(self, X):
        X = as_batch(X)
        return np.asarray(self.estimator.predict(X), dtype=float), self.conformal.std(X)
