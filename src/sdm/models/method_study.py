"""Alternative surrogate structures and objectives, with training-only weighting and stacking."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from catboost import CatBoostRegressor
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from ..interfaces import Surrogate
from ..registry import register_model
from ..simplex import as_batch
from ._gbdt import DistanceConformal, low_target_weights


def features(X, transform):
    X = as_batch(X)
    if transform == "raw":
        return X
    if transform == "hellinger":
        return np.sqrt(np.maximum(X, 0))
    if transform == "interactions":
        within = [X[:, 7*b+i] * X[:, 7*b+j] for b in range(3) for i,j in combinations(range(7), 2)]
        between = [X[:, 7*b+i] * X[:, 7*c+i] for b,c in combinations(range(3), 2) for i in range(7)]
        return np.column_stack([X, *within, *between])
    raise ValueError("transform must be raw, hellinger or interactions")


@dataclass
class PairwiseSquaredLoss:
    """Squared error plus sampled low-target logistic pairs; diagonal Hessian approximation.

    Targets/predictions are standardized inside the estimator's current training subset. Pair
    endpoints and true gaps are derived solely from the labels supplied to that estimator.
    """
    strength: float = .1
    pair_count: int = 8192
    seed: int = 2026
    fraction: float = .3
    min_gap: float = .03

    def prepare(self, y):
        rng = np.random.default_rng(self.seed)
        pool = np.argsort(y, kind="stable")[:max(2, int(self.fraction * len(y))) ]
        left, right = rng.choice(pool, size=(2, self.pair_count))
        keep = np.abs(y[left] - y[right]) > self.min_gap
        left, right = left[keep], right[keep]
        self.better = np.where(y[left] < y[right], left, right)
        self.worse = np.where(y[left] < y[right], right, left)
        self.weights = np.minimum(np.abs(y[left] - y[right]), 1.0)

    def __call__(self, y_true, y_pred, sample_weight=None):
        w = np.ones(len(y_true)) if sample_weight is None else np.asarray(sample_weight)
        gradient, hessian = w * (y_pred - y_true), w.copy()
        if self.strength == 0:
            return gradient, hessian
        if not hasattr(self, "better"):
            self.prepare(y_true)
        if len(self.better):
            factor = self.strength * len(y_true) / len(self.better)
            probability = expit(y_pred[self.better] - y_pred[self.worse])
            g = factor * self.weights * probability
            h = factor * self.weights * probability * (1-probability)
            gradient += np.bincount(self.better, g, minlength=len(y_true))
            gradient -= np.bincount(self.worse, g, minlength=len(y_true))
            hessian += np.bincount(self.better, h, minlength=len(y_true))
            hessian += np.bincount(self.worse, h, minlength=len(y_true))
        return gradient, hessian

    def value(self, y, pred):
        if not hasattr(self, "better"):
            self.prepare(y)
        pair = 0.0 if not len(self.better) else float(np.mean(
            self.weights * np.logaddexp(0, pred[self.better]-pred[self.worse])))
        return float(.5 * np.square(pred-y).sum() + self.strength * len(y) * pair)


@dataclass
class Predictor:
    estimator: object
    transform: str = "raw"
    location: float = 0.0
    scale: float = 1.0

    def predict(self, X):
        return self.location + self.scale * self.estimator.predict(features(X, self.transform))


def convex_weights(predictions, y, sample_weight, prior):
    """Fit nonnegative sum-one stacking weights to training-only OOF predictions."""
    scale = max(float(np.std(y)), 1e-12)
    A, target = predictions / scale, y / scale
    weights = sample_weight / sample_weight.sum()
    def objective(w):
        return float(np.dot(weights, np.square(A @ w-target)) + .02*np.square(w-prior).sum())
    def gradient(w):
        return 2 * A.T @ (weights * (A @ w-target)) + .04*(w-prior)
    result = minimize(objective, prior, jac=gradient, method="SLSQP", bounds=[(0,1)]*len(prior),
        constraints={"type":"eq", "fun": lambda w: w.sum()-1, "jac": lambda w: np.ones(len(w))},
        options={"ftol":1e-12,"maxiter":200})
    if not result.success:
        raise ValueError(f"stacking optimization failed: {result.message}")
    w = np.maximum(result.x, 0)
    return w / w.sum()


@register_model("method-study")
class MethodStudy(Surrogate):
    """CatBoost, robust GBDT/XGBoost, pairwise regression, kernel models and OOF convex ensembles."""

    @dataclass(frozen=True)
    class Params:
        family: str = "catboost"
        loss: str = "RMSE"
        transform: str = "raw"
        iterations: int = 600
        learning_rate: float = .05
        depth: int = 5
        min_samples_leaf: int = 5
        low_fraction: float = .2
        low_weight: float = 3.0
        rank_strength: float = .1
        pair_count: int = 8192
        huber_slope: float = 5.0
        kernel_alpha: float = .1
        kernel_bandwidth: float = .5
        ensemble_mode: str = "fixed"
        calib_folds: int = 3
        calib_bins: int = 4
        coverage: float = .68
        seed: int = 2026

    def _core(self, X, y, family=None):
        p = self.params
        family = family or p.family
        transform = p.transform if p.family != "ensemble" else "raw"
        Z = features(X, transform)
        weight = low_target_weights(y, p.low_fraction, p.low_weight)
        location, scale = 0., 1.
        if family == "catboost":
            model = CatBoostRegressor(iterations=p.iterations, learning_rate=p.learning_rate,
                depth=p.depth, loss_function=p.loss if p.family != "ensemble" else "RMSE",
                l2_leaf_reg=5, border_count=128, random_seed=p.seed, thread_count=1,
                bootstrap_type="Bayesian", bagging_temperature=1, random_strength=1,
                verbose=False, allow_writing_files=False)
        elif family == "gbdt":
            model = GradientBoostingRegressor(n_estimators=p.iterations, learning_rate=p.learning_rate,
                max_depth=p.depth, min_samples_leaf=p.min_samples_leaf, subsample=.9,
                loss=p.loss, alpha=.5 if p.loss == "quantile" else .9, random_state=p.seed)
        elif family in {"xgboost", "pairwise"}:
            objective = "reg:squarederror"
            kwargs = {}
            if family == "pairwise":
                location, scale = float(y.mean()), max(float(y.std()), 1e-12)
                objective = PairwiseSquaredLoss(p.rank_strength, p.pair_count, p.seed,
                                                min_gap=.5/scale)
            elif p.loss == "pseudo-huber":
                objective = "reg:pseudohubererror"
                kwargs = {"huber_slope":p.huber_slope, "base_score":float(np.median(y))}
            model = XGBRegressor(objective=objective, tree_method="hist",
                n_estimators=p.iterations, learning_rate=.1 if p.family == "ensemble" else p.learning_rate,
                max_depth=3 if p.family == "ensemble" else p.depth, subsample=.9,
                n_jobs=1, random_state=p.seed, **kwargs)
        elif family == "histgb":
            weight = np.ones(len(y))
            model = HistGradientBoostingRegressor(max_iter=p.iterations, learning_rate=.05,
                                                  max_leaf_nodes=15, random_state=p.seed)
        elif family == "kernel":
            rng = np.random.default_rng(p.seed)
            pairs = rng.integers(0, len(Z), size=(2,2048))
            distances = np.linalg.norm(Z[pairs[0]]-Z[pairs[1]], axis=1)
            width = p.kernel_bandwidth * np.median(distances[distances > 0])
            model = KernelRidge(alpha=p.kernel_alpha, kernel="rbf", gamma=1/(2*width**2))
            location = float(y.mean())
        else:
            raise ValueError("unknown study family")
        model.fit(Z, (y-location)/scale, sample_weight=weight)
        return Predictor(model, transform, location, scale)

    def fit(self, X, y):
        p = self.params
        X, y = as_batch(X), np.asarray(y, dtype=float)
        if y.shape != (len(X),) or not np.isfinite(y).all():
            raise ValueError("expected one finite target per row")
        if (p.rank_strength < 0 or p.pair_count < 1 or p.kernel_alpha <= 0
                or p.kernel_bandwidth <= 0 or not 0 <= p.low_fraction <= 1 or p.low_weight <= 0):
            raise ValueError("invalid weighting, rank or kernel parameters")
        self.metrics = {}
        if p.family == "ensemble":
            if p.ensemble_mode not in {"fixed", "stack"}:
                raise ValueError("ensemble_mode must be fixed or stack")
            families = ("xgboost", "catboost", "histgb")
            oof = np.empty((len(y),3))
            folds = list(KFold(p.calib_folds, shuffle=True, random_state=p.seed).split(X))
            for train,test in folds:
                for column,family in enumerate(families):
                    oof[test,column] = self._core(X[train], y[train], family).predict(X[test])
            prior = np.array([.5,.3,.2])
            self.blend_weights = prior if p.ensemble_mode == "fixed" else convex_weights(
                oof,y,low_target_weights(y,p.low_fraction,p.low_weight),prior)
            self.members = [self._core(X,y,family) for family in families]
            # Stacking weights use these OOF labels, so its residual proxy is selection-aware.
            fit_predict = lambda train,test: oof[test] @ self.blend_weights
            self.metrics.update({"weight_"+f:float(w) for f,w in zip(families,self.blend_weights)})
        else:
            fit_predict = lambda train,test: self._core(X[train],y[train]).predict(X[test])
            self.predictor = self._core(X,y)
        self.conformal = DistanceConformal().fit(fit_predict,X,y,folds=p.calib_folds,bins=p.calib_bins,
                                                  coverage=p.coverage,seed=p.seed)
        self.metrics.update({"training_rows":len(y), **self.conformal.metrics})
        return self

    def predict(self, X):
        X = as_batch(X)
        mean = (np.column_stack([member.predict(X) for member in self.members]) @ self.blend_weights
                if self.params.family == "ensemble" else self.predictor.predict(X))
        return np.asarray(mean,dtype=float), self.conformal.std(X)
