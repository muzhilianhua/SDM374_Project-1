"""Gradient-free local search on the simplex that stays inside a trust region around measured rows."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..evaluate import median_spacing, nearest_distances
from ..interfaces import Optimizer, Surrogate
from ..registry import register_optimizer
from ..simplex import N_BUILDINGS, N_METRICS, as_batch

# Candidates closer than this to an observed row count as re-measuring that row.
OBSERVED_TOLERANCE = 1e-6


def _spread(points: np.ndarray, separation: float, n: int) -> np.ndarray:
    """Greedily keep points (already sorted best first) at least `separation` apart, up to n of them."""
    chosen = []
    for point in points:
        if not chosen or np.min(np.linalg.norm(np.asarray(chosen) - point, axis=1)) >= separation:
            chosen.append(point)
            if len(chosen) == n:
                break
    return np.asarray(chosen)


@register_optimizer("trust-dirichlet")
class TrustRegionDirichlet(Optimizer):
    """Dirichlet local search from the best measured rows, inside a trust region, minimizing mean + risk * std."""

    @dataclass(frozen=True)
    class Params:
        centers: int = 16
        iterations: int = 30
        samples_per_center: int = 25
        concentration: float = 200.0
        growth: float = 1.15
        risk: float = 1.0
        max_distance_ratio: float = 1.5
        min_separation_ratio: float = 0.25
        exclude_observed: bool = True
        seed: int = 2026

    def _perturb(self, centers: np.ndarray, concentration: float, rng: np.random.Generator) -> np.ndarray:
        # Dirichlet(alpha) per building via normalized Gamma draws, vectorized over centers and samples.
        alpha = concentration * centers.reshape(-1, 1, N_BUILDINGS, N_METRICS) + 0.01
        alpha = np.broadcast_to(alpha, (len(centers), self.params.samples_per_center, N_BUILDINGS, N_METRICS))
        draws = rng.gamma(alpha)
        return (draws / draws.sum(axis=-1, keepdims=True)).reshape(-1, N_BUILDINGS * N_METRICS)

    def propose(self, model: Surrogate, X, y, n=1):
        p = self.params
        X, y = as_batch(X), np.asarray(y, dtype=float)
        rng = np.random.default_rng(p.seed)
        spacing = median_spacing(X)
        radius = p.max_distance_ratio * spacing
        centers = X[np.argsort(y, kind="stable")[:p.centers]]
        points, scores = np.empty((0, X.shape[1])), np.empty(0)
        concentration = p.concentration
        for _ in range(p.iterations):
            candidates = self._perturb(centers, concentration, rng)
            distance = nearest_distances(candidates, X)
            keep = distance <= radius
            if p.exclude_observed:
                keep &= distance > OBSERVED_TOLERANCE
            if keep.any():
                mean, std = model.predict(candidates[keep])
                points = np.concatenate([points, candidates[keep]])
                scores = np.concatenate([scores, mean + p.risk * std])
                centers = points[np.argsort(scores, kind="stable")[:p.centers]]
            concentration *= p.growth
        if len(points) < n:
            raise ValueError(f"only {len(points)} candidates fell inside the trust region; "
                             "raise max_distance_ratio or iterations")

        ranked = points[np.argsort(scores, kind="stable")]
        separation = p.min_separation_ratio * spacing
        chosen = _spread(ranked, separation, n)
        while len(chosen) < n:
            separation /= 2
            chosen = _spread(ranked, separation, n)
        best_mean, best_std = model.predict(chosen[:1])
        self.diagnostics = {"candidates_evaluated": int(len(points)), "best_objective": float(scores.min()),
                            "best_mean": float(best_mean[0]), "best_std": float(best_std[0]),
                            "separation": float(separation), "spacing": float(spacing)}
        return chosen
