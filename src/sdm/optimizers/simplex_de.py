"""Differential evolution with simplex projection and a distance-to-data trust region."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..evaluate import median_spacing, nearest_distances
from ..interfaces import Optimizer, Surrogate
from ..registry import register_optimizer
from ..simplex import DIM, N_BUILDINGS, N_METRICS, check_weights, project
from .trust_dirichlet import OBSERVED_TOLERANCE, _spread


@register_optimizer("simplex-de")
class SimplexDifferentialEvolution(Optimizer):
    """Gradient-free DE/rand/1/bin, simplex projection, trust region and mean + risk * std ranking."""

    @dataclass(frozen=True)
    class Params:
        population: int = 128
        generations: int = 92
        centers: int = 16
        concentration: float = 100.0
        mutation: float = 0.6
        crossover: float = 0.7
        risk: float = 1.0
        max_distance_ratio: float = 0.75
        min_separation_ratio: float = 0.25
        seed: int = 2026

    def propose(self, model: Surrogate, X, y, n=1):
        p = self.params
        numeric = (p.concentration, p.mutation, p.crossover, p.risk,
                   p.max_distance_ratio, p.min_separation_ratio)
        if (not np.isfinite(numeric).all() or p.population < 4 or p.generations < 1
                or p.centers < 1 or p.concentration <= 0 or not 0 < p.mutation <= 2
                or not 0 <= p.crossover <= 1 or p.risk < 0
                or p.max_distance_ratio <= 0 or p.min_separation_ratio <= 0 or n < 1):
            raise ValueError("invalid simplex-de parameters or requested batch size")
        X, y = check_weights(X), np.asarray(y, dtype=float)
        if y.shape != (len(X),) or len(X) < 2 or not np.isfinite(y).all():
            raise ValueError("simplex-de needs at least two observations and finite aligned labels")
        spacing = median_spacing(X)
        radius = p.max_distance_ratio * spacing
        if spacing <= 0:
            raise ValueError("trust region needs positive training spacing")
        rng = np.random.default_rng(p.seed)
        centers = X[np.argsort(y, kind="stable")[:p.centers]]
        anchors = centers[np.arange(p.population) % len(centers)]
        draws = rng.gamma(p.concentration * anchors.reshape(-1, N_BUILDINGS, N_METRICS) + 0.01)
        population = (draws / draws.sum(axis=-1, keepdims=True)).reshape(-1, DIM)
        # Convex contraction preserves the simplex and makes every initial point feasible.
        length = np.linalg.norm(population - anchors, axis=1)
        fraction = np.minimum(1.0, 0.99 * radius / np.maximum(length, 1e-15))
        population = anchors + fraction[:, None] * (population - anchors)
        archive, archived_scores = [], []
        evaluated = 0

        def objective(points):
            nonlocal evaluated
            distance = nearest_distances(points, X)
            valid = (distance <= radius) & (distance > OBSERVED_TOLERANCE)
            scores = np.full(len(points), np.inf)
            if valid.any():
                mean, std = model.predict(points[valid])
                value = np.asarray(mean) + p.risk * np.asarray(std)
                if not np.isfinite(value).all():
                    raise ValueError("surrogate returned non-finite acquisition scores")
                scores[valid] = value
                archive.append(points[valid].copy())
                archived_scores.append(value.copy())
                evaluated += int(valid.sum())
            return scores

        scores = objective(population)
        for _ in range(p.generations):
            # Each target uses three distinct donors, none equal to itself.
            donors = np.empty((p.population, 3), dtype=int)
            for target in range(p.population):
                eligible = np.delete(np.arange(p.population), target)
                donors[target] = rng.choice(eligible, size=3, replace=False)
            a, b, c = donors.T
            mutant = population[a] + p.mutation * (population[b] - population[c])
            mask = rng.random((p.population, DIM)) < p.crossover
            mask[np.arange(p.population), rng.integers(DIM, size=p.population)] = True
            trial = project(np.where(mask, mutant, population))
            trial_scores = objective(trial)
            # Equal-fitness replacement permits movement across a tree's flat leaves.
            replace = np.isfinite(trial_scores) & (trial_scores <= scores)
            population[replace], scores[replace] = trial[replace], trial_scores[replace]

        if not archive:
            raise ValueError("no new candidates inside the trust region")
        points, values = np.concatenate(archive), np.concatenate(archived_scores)
        ranked = points[np.argsort(values, kind="stable")]
        _, first = np.unique(np.round(ranked, 9), axis=0, return_index=True)
        ranked = ranked[np.sort(first)]
        if len(ranked) < n:
            raise ValueError(f"only {len(ranked)} distinct candidates inside the trust region; requested {n}")
        separation = p.min_separation_ratio * spacing
        chosen = _spread(ranked, separation, n)
        while len(chosen) < n and separation > 1e-10:
            separation /= 2
            chosen = _spread(ranked, separation, n)
        if len(chosen) < n:
            raise ValueError("not enough separated candidates; request a smaller batch")
        self.diagnostics = {"candidates_evaluated": evaluated,
                            "proposals_generated": p.population * (p.generations + 1),
                            "best_objective": float(values.min()),
                            "separation": float(separation), "spacing": spacing}
        return check_weights(chosen)
