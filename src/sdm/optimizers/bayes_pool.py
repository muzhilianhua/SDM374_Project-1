"""Finite-pool Bayesian acquisition on feasible, nearby simplex candidates."""
from dataclasses import dataclass
import numpy as np
from scipy.special import ndtr
from ..interfaces import Optimizer
from ..registry import register_optimizer
from ..simplex import as_batch, check_weights
from ..evaluate import median_spacing, nearest_distances
from .trust_dirichlet import _spread


@register_optimizer("bayes-pool")
class BayesianPool(Optimizer):
    """Expected improvement or conservative posterior acquisition, with locality and batch diversity."""
    @dataclass(frozen=True)
    class Params:
        acquisition: str = "ei"
        risk: float = 1.0
        samples: int = 6000
        max_distance_ratio: float = 0.5
        separation_ratio: float = 0.1
        seed: int = 2026

    def propose(self, model, X, y, n=1):
        X, y = as_batch(X), np.asarray(y)
        p, rng = self.params, np.random.default_rng(self.params.seed)
        spacing = median_spacing(X)
        anchors = X[np.argsort(y, kind="stable")[:32]]
        center = anchors[rng.integers(len(anchors), size=p.samples)].reshape(-1, 3, 7)
        concentration = rng.choice([100., 300., 1000., 3000.], size=(p.samples, 1, 1))
        gamma = rng.gamma(center*concentration+.03)
        points = (gamma/gamma.sum(axis=2, keepdims=True)).reshape(-1, 21)
        distance = nearest_distances(points, X)
        keep = (distance <= spacing*p.max_distance_ratio) & (distance > 1e-6)
        points, distance = points[keep], distance[keep]
        means, spreads = [], []
        for start in range(0, len(points), 256):
            mean, std = model.predict(points[start:start+256])
            means.extend(mean); spreads.extend(std)
        mean, std = np.asarray(means), np.maximum(np.asarray(spreads), 1e-12)
        if p.acquisition == "ei":
            gain = y.min()-mean
            z = gain/std
            objective = -(gain*ndtr(z)+std*np.exp(-z*z/2)/np.sqrt(2*np.pi))
        elif p.acquisition == "pessimistic":
            objective = mean+p.risk*std
        elif p.acquisition == "mean":
            objective = mean
        else:
            raise ValueError("unknown acquisition")
        if len(points) < n:
            raise ValueError("insufficient feasible pool points")
        ranked = points[np.argsort(objective, kind="stable")]
        separation = p.separation_ratio*spacing
        chosen = _spread(ranked, separation, n)
        while len(chosen) < n:
            separation /= 2
            chosen = _spread(ranked, separation, n)
        self.diagnostics = {"pool_size": len(points), "spacing": float(spacing),
                            "best_acquisition_objective": float(objective.min()), "separation": float(separation)}
        return check_weights(chosen)
