"""Multi-start projected gradient descent on the surrogate mean, ported from baseline.py."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..interfaces import Optimizer, Surrogate
from ..registry import register_optimizer
from ..simplex import DIM, N_METRICS, as_batch, project, sample


@register_optimizer("pgd")
class ProjectedGradientDescent(Optimizer):
    """Descends from the uniform weights, the best observed row and random starts, all updated together."""

    @dataclass(frozen=True)
    class Params:
        starts: int = 16
        max_iter: int = 5000
        tolerance: float = 1e-7
        seed: int = 2026

    def propose(self, model: Surrogate, X, y, n=1):
        starts, max_iter, tolerance = self.params.starts, self.params.max_iter, self.params.tolerance
        if starts < 2:
            raise ValueError("pgd needs at least two starts")
        if not 1 <= n <= starts:
            raise ValueError(f"pgd returns one point per start, so n must be in [1, {starts}]")
        if max_iter < 1 or tolerance <= 0:
            raise ValueError("max_iter and tolerance must be positive")
        X, y = as_batch(X), np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.params.seed)
        points = project(np.concatenate([
            np.full((1, DIM), 1 / N_METRICS), X[[int(np.argmin(y))]], sample(starts - 2, rng),
        ]))
        step = 1.0 / max(model.smoothness(), 1.0)
        for iteration in range(1, max_iter + 1):
            updated = project(points - step * model.gradient(points))
            residuals = np.max(np.abs(updated - points), axis=1) / step
            points = updated
            if residuals.max() <= tolerance:
                break
        self.diagnostics = {"iterations": iteration, "max_residual": float(residuals.max()),
                            "converged": bool(residuals.max() <= tolerance)}
        mean, _ = model.predict(points)
        return points[np.argsort(mean, kind="stable")[:n]]
