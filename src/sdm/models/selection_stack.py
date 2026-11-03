"""Nested measured-pool tuning of a selection-conditioned residual correction to the OOF stack."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import warnings

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from ..evaluate import median_spacing, nearest_distances
from ..interfaces import Surrogate
from ..registry import register_model
from ..simplex import as_batch
from ._gbdt import DistanceConformal
from .method_study import MethodStudy


def trial_rows(n, seed, repeats):
    """Per-trial disjoint base/correction/validation indices, independent of labels."""
    rng = np.random.default_rng(seed + 1703)
    for _ in range(repeats):
        order = rng.permutation(n)
        yield order[:int(.6*n)], order[int(.6*n):int(.8*n)], order[int(.8*n):]


def measured_pools(n, seed, size=50):
    """Label-free disjoint pools, approximately size rows, with no omitted observations."""
    return np.array_split(np.random.default_rng(seed).permutation(n), max(1, n//size))


@dataclass
class ResidualCorrection:
    alpha: float = 10.

    def fit(self, features, residual, raw_mean, target_scale):
        self.location = features.mean(axis=0)
        self.scale = np.maximum(features.std(axis=0), 1e-8)
        weight = np.ones(len(residual))
        weight[np.argsort(raw_mean, kind="stable")[:max(1, len(residual)//5)]] = 4.
        self.estimator = Ridge(alpha=self.alpha).fit(
            (features-self.location)/self.scale, residual, sample_weight=weight)
        self.cap = .5*target_scale
        return self

    def predict(self, features):
        return np.clip(self.estimator.predict((features-self.location)/self.scale), -self.cap, self.cap)


@dataclass
class PredictionContext:
    X: np.ndarray
    location: float
    scale: float
    low: float
    spacing: float

    @classmethod
    def from_training(cls, X, y):
        return cls(X, float(np.median(y)), max(float(y.std()), 1e-8),
                   float(np.quantile(y, .2)), max(median_spacing(X), 1e-8))

    def features(self, model, X):
        members = np.column_stack([m.predict(X) for m in model.members])
        mean = members @ model.blend_weights
        feature = np.column_stack([(mean-self.location)/self.scale,
            np.maximum(self.low-mean, 0)/self.scale, members.std(axis=1)/self.scale,
            np.log1p(nearest_distances(X, self.X)/self.spacing), np.square(X).sum(axis=1)])
        return mean, feature


@register_model("selection-stack")
class SelectionStack(Surrogate):
    """OOF stack plus residual correction tuned on internally held-out measured candidate pools."""

    @dataclass(frozen=True)
    class Params:
        iterations: int = 600
        calib_folds: int = 3
        calib_bins: int = 4
        coverage: float = .68
        correction_alpha: float = 10.
        strengths: tuple[float, ...] = (0., .25, .5, 1.)
        selection_repeats: int = 3
        pool_size: int = 50
        fit_workers: int = 3
        seed: int = 2026

    def _core(self):
        return MethodStudy(MethodStudy.Params(family="ensemble", ensemble_mode="stack",
                           iterations=self.params.iterations, seed=self.params.seed))

    def _trial(self, X, y, rows):
        base, correction, validation = rows
        core = self._core().fit(X[base], y[base])
        context = PredictionContext.from_training(X[base], y[base])
        raw_b, feature_b = context.features(core, X[correction])
        adjuster = ResidualCorrection(self.params.correction_alpha).fit(
            feature_b, y[correction]-raw_b, raw_b, context.scale)
        raw_c, feature_c = context.features(core, X[validation])
        return {"rows": rows, "raw": raw_c, "delta": adjuster.predict(feature_c),
                "actual": y[validation].copy()}

    def _choose_strength(self, trials):
        records = []
        for strength in sorted(set(self.params.strengths)):
            absolute, regrets, actual, optimism = [], [], [], []
            for index, trial in enumerate(trials):
                pred = trial["raw"] + strength*trial["delta"]
                y = trial["actual"]
                absolute.extend(abs(y-pred))
                for pool in measured_pools(len(y), self.params.seed+1801+index, self.params.pool_size):
                    chosen = pool[np.argmin(pred[pool])]
                    regrets.append(y[chosen]-y[pool].min())
                    actual.append(y[chosen])
                    optimism.append(y[chosen]-pred[chosen])
            records.append({"strength": float(strength), "mae": float(np.mean(absolute)),
                "pool_regret": float(np.mean(regrets)), "selected_actual_R": float(np.mean(actual)),
                "selected_signed_optimism": float(np.mean(optimism)), "pools": len(regrets)})
        raw_mae = next(r["mae"] for r in records if r["strength"] == 0.)
        for row in records:
            row["mae_eligible"] = bool(row["mae"] <= 1.05*raw_mae)
        eligible = [r for r in records if r["mae_eligible"]]
        chosen = min(eligible, key=lambda r: (r["pool_regret"], r["strength"]))
        return chosen["strength"], records

    def fit(self, X, y):
        X, y = as_batch(X), np.asarray(y, dtype=float)
        p = self.params
        if (p.iterations < 1 or p.calib_folds < 2 or p.calib_bins < 1 or p.fit_workers < 1
                or p.selection_repeats < 1 or p.pool_size < 2 or p.correction_alpha <= 0
                or not 0 < p.coverage < 1 or 0. not in p.strengths
                or any(not np.isfinite(s) or s < 0 or s > 1 for s in p.strengths)
                or len(y) < max(30, 5*p.calib_folds)
                or y.shape != (len(X),) or not np.isfinite(y).all()):
            raise ValueError("invalid selection-stack settings or training observations")
        with ThreadPoolExecutor(max_workers=p.fit_workers) as pool:
            trials = list(pool.map(lambda rows: self._trial(X, y, rows),
                                   trial_rows(len(y), p.seed, p.selection_repeats)))
        self.strength_, selection = self._choose_strength(trials)
        self.selection_trials_ = trials
        folds = list(KFold(p.calib_folds, shuffle=True, random_state=p.seed).split(X))

        def held_out(fold):
            train, test = fold
            core = self._core().fit(X[train], y[train])
            return PredictionContext.from_training(X[train], y[train]).features(core, X[test])

        with ThreadPoolExecutor(max_workers=p.fit_workers) as pool:
            values = list(pool.map(held_out, folds))
        self.calibration_prediction_ = np.empty(len(y))
        calibration_features = np.empty((len(y), 5))
        lookup = {}
        for (train, test), (mean, feature) in zip(folds, values):
            self.calibration_prediction_[test], calibration_features[test] = mean, feature
            lookup[tuple(test)] = (train, mean)

        def cached(train, test):
            recorded_train, mean = lookup[tuple(test)]
            np.testing.assert_array_equal(train, recorded_train)
            return mean

        self.conformal = DistanceConformal().fit(cached, X, y, folds=p.calib_folds,
            bins=p.calib_bins, coverage=p.coverage, seed=p.seed)
        self.context_ = PredictionContext.from_training(X, y)
        self.correction_ = ResidualCorrection(p.correction_alpha).fit(calibration_features,
            y-self.calibration_prediction_, self.calibration_prediction_, self.context_.scale)
        self.predictor_ = self._core().fit(X, y)
        self.metrics = {**self.predictor_.metrics, "selected_correction_strength": self.strength_,
            "correction_feature_distribution_mismatch": self.strength_ > 0,
            "internal_selection_tuning": selection,
            "nested_raw_calibration_mae": float(np.mean(abs(y-self.calibration_prediction_))),
            "error_scale_is_heuristic": True,
            "internal_selection_is_not_independent_validation": True,
            "raw_calibration_labels_excluded_from_predictive_fit": True}
        return self

    def predict(self, X):
        X = as_batch(X)
        raw, feature = self.context_.features(self.predictor_, X)
        if self.strength_ == 0:
            return raw, self.conformal.std(X)
        warnings.warn(
            "selection-stack correction uses full-data features after training on fold features; "
            "nonzero correction is experimental and has not been validated for this shift",
            RuntimeWarning, stacklevel=2)
        correction = self.strength_*self.correction_.predict(feature)
        return raw+correction, self.conformal.std(X)+abs(correction)
