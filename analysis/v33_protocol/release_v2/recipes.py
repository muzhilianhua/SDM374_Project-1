"""Research-only registry plugin for the original three Runze model lineages."""
from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
from sdm.interfaces import Surrogate
from sdm.registry import register_model
from sdm.simplex import as_batch


@register_model('runze-v33')
class RunzeRecipe(Surrogate):
    """Fixed-parameter trimmed HistGB/XGB; zero std is an unused placeholder, not uncertainty."""

    @dataclass(frozen=True)
    class Params:
        learner: str = 'histgb'
        trim_fraction: float = 0.8
        low_fraction: float = 0.2
        low_weight: float = 1.0
        max_depth: int = 3
        max_leaf_nodes: int = 15
        learning_rate: float = 0.05
        iterations: int = 600
        subsample: float = 0.9
        l2_regularization: float = 0.0
        seed: int = 20263024

    def fit(self, X, y):
        X, y = as_batch(X), np.asarray(y, dtype=float)
        p = self.params
        if p.learner not in ('histgb', 'xgboost') or not 0 < p.trim_fraction <= 1:
            raise ValueError('Invalid learner or retained fraction')
        if not 0 <= p.low_fraction <= 1 or p.low_weight <= 0:
            raise ValueError('Invalid low-target weight policy')
        if y.shape != (len(X),) or not np.isfinite(y).all():
            raise ValueError('Invalid training labels')
        order = np.argsort(y, kind='stable')
        n = max(2, int(np.floor(p.trim_fraction * len(y))))
        self.retained_indices_ = np.sort(order[:n])
        weights = np.ones(len(y))
        weights[order[:int(np.floor(p.low_fraction * len(y)))]] = p.low_weight
        if p.learner == 'histgb':
            self.estimator = HistGradientBoostingRegressor(max_leaf_nodes=p.max_leaf_nodes,
                learning_rate=p.learning_rate, max_iter=p.iterations,
                l2_regularization=p.l2_regularization, random_state=p.seed)
        else:
            self.estimator = XGBRegressor(objective='reg:squarederror', tree_method='hist',
                max_depth=p.max_depth, learning_rate=p.learning_rate, n_estimators=p.iterations,
                subsample=p.subsample, random_state=p.seed, n_jobs=1)
        self.estimator.fit(X[self.retained_indices_], y[self.retained_indices_],
                           sample_weight=weights[self.retained_indices_])
        self.metrics = {'input_rows': len(y), 'retained_rows': len(self.retained_indices_),
                        'weighted_rows': int(np.sum(weights[self.retained_indices_] != 1)),
                        'uncertainty_available': False}
        return self

    def predict(self, X):
        predictions = np.asarray(self.estimator.predict(as_batch(X)), dtype=float)
        return predictions, np.zeros(len(predictions))
