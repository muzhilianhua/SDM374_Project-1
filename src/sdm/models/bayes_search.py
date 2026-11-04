"""Bayesian and smooth-kernel alternatives with training-only hyperparameter selection."""
from dataclasses import dataclass

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter, Matern, RBF, RationalQuadratic, WhiteKernel
from scipy.optimize import minimize
from sklearn.linear_model import BayesianRidge, ARDRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

from ..interfaces import Surrogate
from ..registry import register_model
from ..simplex import as_batch


class BuildingKernel(Kernel):
    """Additive per-building Matern covariance plus a whole-district interaction."""
    def __init__(self, length_scale=0.5, global_weight=0.25):
        self.length_scale = length_scale
        self.global_weight = global_weight

    def __call__(self, X, Y=None, eval_gradient=False):
        kernel = Matern(self.length_scale, length_scale_bounds="fixed", nu=2.5)
        result = sum(kernel(X[:, 7*b:7*b+7], None if Y is None else Y[:, 7*b:7*b+7])
                     for b in range(3)) * ((1-self.global_weight)/3)
        result += self.global_weight * kernel(X, Y)
        return (result, np.empty((*result.shape, 0))) if eval_gradient else result

    def diag(self, X):
        return np.ones(len(X))

    def is_stationary(self):
        return True


class BuildingARDKernel(Kernel):
    """Learn seven shared objective length scales across three building kernels."""
    def __init__(self, length_scale=(.5,)*7, global_weight=.1):
        self.length_scale = length_scale
        self.global_weight = global_weight

    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter('length_scale', 'numeric', (.03, 3.0), len(self.length_scale))

    def __call__(self, X, Y=None, eval_gradient=False):
        kernel = Matern(np.asarray(self.length_scale), length_scale_bounds=(.03, 3), nu=2.5)
        parts = [kernel(X[:,7*b:7*b+7], None if Y is None else Y[:,7*b:7*b+7],
                        eval_gradient=eval_gradient) for b in range(3)]
        full_kernel = Matern(np.tile(self.length_scale, 3), length_scale_bounds=(.03, 3), nu=2.5)
        full = full_kernel(X,Y,eval_gradient=eval_gradient)
        weight = self.global_weight
        if eval_gradient:
            matrix = sum(p[0] for p in parts)*(1-weight)/3+weight*full[0]
            grad = sum(p[1] for p in parts)*(1-weight)/3
            grad += weight*full[1].reshape(len(X),len(X),3,7).sum(axis=2)
            return matrix, grad
        return sum(parts)*(1-weight)/3+weight*full

    def diag(self, X):
        return np.ones(len(X))

    def is_stationary(self):
        return True


class LimitedOptimizer:
    """Keep the historical optimization budget and report termination honestly."""
    def __init__(self):
        self.diagnostics = {}

    def __call__(self, objective, theta, bounds):
        result = minimize(objective, theta, method='L-BFGS-B', jac=True, bounds=bounds,
                          options={'maxiter': 18, 'ftol': 1e-5, 'maxls': 10})
        self.diagnostics = {'optimizer_nit': int(result.nit),
            'optimizer_success': bool(result.success), 'optimizer_status': int(result.status),
            'optimizer_nfev': int(result.nfev),
            'optimizer_gradient_inf_norm': float(np.max(np.abs(result.jac)))}
        return result.x, result.fun


def limited_optimizer(objective, theta, bounds):
    """Compatibility with historical locally produced fitted models."""
    return LimitedOptimizer()(objective, theta, bounds)


@register_model("bayes-search")
class BayesianSearchModel(Surrogate):
    """GP, Bayesian polynomial regression or inner-selected SVR; raw or Hellinger geometry."""
    @dataclass(frozen=True)
    class Params:
        family: str = "gp"
        kernel: str = "matern"
        transform: str = "raw"
        target: str = "raw"
        trend: bool = False
        lengths: tuple = (0.25, 0.5, 1.0)
        noises: tuple = (0.005, 0.03)
        seed: int = 2026

    def _features(self, X):
        X = as_batch(X)
        return np.sqrt(np.maximum(X, 0)) if self.params.transform == "sqrt" else X

    @property
    def uncertainty_kind(self):
        p = self.params
        return ('gaussian-posterior' if p.family == 'gp' and p.target == 'raw' and not p.trend
                else 'heuristic')

    def fit(self, X, y):
        p = self.params
        X, y = self._features(X), np.asarray(y, dtype=float)
        if p.target not in ("raw", "log"):
            raise ValueError("target must be raw or log")
        # No arbitrary physical lower bound: log(R), with inverse median exp(mu).
        z = np.log(np.maximum(y, 1e-9)) if p.target == "log" else y.copy()
        self.location_, self.scale_ = float(z.mean()), max(float(z.std()), 1e-9)
        z = (z-self.location_)/self.scale_
        self.trend_ = None
        if p.trend:
            self.trend_ = make_pipeline(PolynomialFeatures(2, include_bias=False),
                                       StandardScaler(), BayesianRidge())
            self.trend_.fit(X, z)
            z = z-self.trend_.predict(X)
        if p.family == "gp":
            best = -np.inf
            lengths = tuple(dict.fromkeys(float(v) for v in p.lengths))
            noises = tuple(dict.fromkeys(float(v) for v in p.noises))
            if not lengths or not noises or any(not np.isfinite(v) or v <= 0 for v in lengths+noises):
                raise ValueError('lengths and noises must contain finite positive values')
            if p.kernel == 'additive-ard' and (any(not .03 <= v <= 3 for v in lengths)
                                               or any(not .001 <= v <= .3 for v in noises)):
                raise ValueError('ARD starts must lie within length/noise bounds')
            fits = 0
            for length in lengths:
                if p.kernel == "additive-ard":
                    signal = BuildingARDKernel((length,)*7)
                elif p.kernel == "additive":
                    signal = BuildingKernel(length)
                elif p.kernel == "rbf":
                    signal = RBF(length, length_scale_bounds="fixed")
                elif p.kernel == "rq":
                    signal = RationalQuadratic(length, alpha=1.0, length_scale_bounds="fixed", alpha_bounds="fixed")
                else:
                    signal = Matern(length, length_scale_bounds="fixed", nu=2.5)
                for noise in noises:
                    ard = p.kernel == "additive-ard"
                    kernel = signal+WhiteKernel(noise, noise_level_bounds=(.001,.3) if ard else "fixed")
                    optimizer = LimitedOptimizer() if ard else None
                    # No random restarts: seed does not vary this GP's fitted predictions.
                    model = GaussianProcessRegressor(kernel=kernel, optimizer=optimizer, normalize_y=False,
                                                     alpha=1e-8, n_restarts_optimizer=0).fit(X, z)
                    fits += 1
                    value = model.log_marginal_likelihood_value_
                    if value > best:
                        best, self.estimator_ = value, model
                        learned = np.atleast_1d(model.kernel_.k1.length_scale)
                        self.metrics = {'initial_length': length, 'initial_noise': noise,
                            'noise': float(model.kernel_.k2.noise_level),
                            'log_marginal_likelihood': float(value), 'gp_seed_affects_fit': False,
                            **{f'length_{i}': float(v) for i,v in enumerate(learned)}}
                        if optimizer is not None:
                            self.metrics.update(optimizer.diagnostics)
            self.metrics['kernel_fits'] = fits
        elif p.family in ("bayesian-ridge", "ard"):
            estimator = BayesianRidge() if p.family == "bayesian-ridge" else ARDRegression(n_iter=150)
            self.estimator_ = make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(), estimator)
            self.estimator_.fit(X, z)
        elif p.family == "svr":
            # Every outer training set gets its own 3-fold selection; outer labels are never used.
            folds = list(KFold(3, shuffle=True, random_state=p.seed).split(X))
            best = np.inf
            for C in (10.0, 100.0):
                for gamma in (0.5, 2.0, 8.0):
                    prediction = np.empty(len(z))
                    for train, test in folds:
                        # Recompute target scale on this inner training partition.
                        inner_mean = float(z[train].mean())
                        inner_scale = max(float(z[train].std()), 1e-9)
                        model = SVR(C=C, gamma=gamma, epsilon=0.03)
                        model.fit(X[train], (z[train]-inner_mean)/inner_scale)
                        prediction[test] = inner_mean+inner_scale*model.predict(X[test])
                    weights = 1+2*(y <= np.quantile(y, .25))
                    error = float(np.average(np.abs(prediction-z), weights=weights))
                    if error < best:
                        best, settings, residual = error, (C, gamma), prediction-z
            self.estimator_ = SVR(C=settings[0], gamma=settings[1], epsilon=.03).fit(X, z)
            self.residual_ = float(np.sqrt(np.mean(residual**2)))
            self.metrics = {"inner_mae": best, "C": settings[0], "gamma": settings[1]}
        else:
            raise ValueError("unknown family")
        self.metrics["train_rows"] = float(len(y))
        return self

    def predict(self, X):
        X = self._features(X)
        if self.params.family == "svr":
            mean = self.estimator_.predict(X)
            std = np.full(len(X), self.residual_)
        else:
            mean, std = self.estimator_.predict(X, return_std=True)
        if self.trend_ is not None:
            mean = mean+self.trend_.predict(X)
        mean, std = self.location_+self.scale_*mean, self.scale_*std
        if self.params.target == "log":
            # Median for point prediction; delta-method spread, not calibrated coverage.
            mean = np.exp(np.clip(mean, -20, 20))
            std = mean*std
        return np.asarray(mean), np.asarray(std)
