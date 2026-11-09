"""Deterministic exploit/explore batches for the sequential online evaluator."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..evaluate import median_spacing, nearest_distances
from ..interfaces import Optimizer, Surrogate
from ..registry import register_optimizer
from ..simplex import DIM, N_BUILDINGS, N_METRICS, as_batch, check_weights


MAX_BATCH = 50
OBSERVED_TOLERANCE = 1e-8


def _unique_rows(points: np.ndarray) -> np.ndarray:
    """Remove numerical duplicates while preserving the first occurrence."""
    _, first = np.unique(np.round(points, 12), axis=0, return_index=True)
    return points[np.sort(first)]


def _diverse_selection(
    points: np.ndarray,
    order: np.ndarray,
    count: int,
    separation: float,
    already: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Greedily select ranked rows, relaxing separation only when necessary."""
    if count == 0:
        return np.empty(0, dtype=int), float(separation)
    already = np.empty((0, points.shape[1])) if already is None else np.asarray(already)
    current = max(float(separation), 0.0)
    for _ in range(32):
        selected: list[int] = []
        selected_points = already.copy()
        for index in order:
            point = points[index]
            if len(selected_points) and current > 0:
                if np.linalg.norm(selected_points - point, axis=1).min() < current:
                    continue
            selected.append(int(index))
            selected_points = np.vstack((selected_points, point))
            if len(selected) == count:
                return np.asarray(selected, dtype=int), current
        if current == 0:
            break
        current *= 0.5
    if len(selected) < count:
        raise ValueError(f"only {len(selected)} separated candidates available; requested {count}")
    return np.asarray(selected, dtype=int), current


@register_optimizer("online-batch")
class OnlineBatchOptimizer(Optimizer):
    """Deterministic trust-region batches split between model exploitation and uncertainty exploration."""

    @dataclass(frozen=True)
    class Params:
        exploit_fraction: float = 0.5
        centers: int = 32
        pool_size: int = 4000
        concentration: float = 100.0
        alpha_floor: float = 0.01
        risk: float = 1.0
        exploration_weight: float = 1.0
        diversity_weight: float = 0.25
        radius_ratio: float = 1.0
        min_radius_ratio: float = 0.25
        max_radius_ratio: float = 1.5
        min_separation_ratio: float = 0.1
        expand_factor: float = 1.5
        shrink_factor: float = 0.7
        success_patience: int = 1
        failure_patience: int = 1
        improvement_tolerance: float = 1e-12
        seed: int = 2026

    def __init__(self, params=None):
        super().__init__(params)
        self._radius_ratio = float(self.params.radius_ratio)
        self._proposal_index = 0
        self._success_streak = 0
        self._failure_streak = 0

    def _validate_params(self, n: int) -> None:
        p = self.params
        finite = (p.exploit_fraction, p.concentration, p.alpha_floor, p.risk,
                  p.exploration_weight, p.diversity_weight, p.radius_ratio,
                  p.min_radius_ratio, p.max_radius_ratio, p.min_separation_ratio,
                  p.expand_factor, p.shrink_factor, p.improvement_tolerance)
        if (n < 1 or n > MAX_BATCH or p.centers < 1 or p.pool_size < max(64, n)
                or not np.isfinite(finite).all() or not 0 <= p.exploit_fraction <= 1
                or p.concentration <= 0 or p.alpha_floor <= 0 or p.risk < 0
                or p.exploration_weight < 0 or p.diversity_weight < 0
                or p.min_radius_ratio <= 0 or p.max_radius_ratio < p.min_radius_ratio
                or not p.min_radius_ratio <= p.radius_ratio <= p.max_radius_ratio
                or p.min_separation_ratio <= 0 or p.expand_factor <= 1
                or not 0 < p.shrink_factor < 1 or p.success_patience < 1
                or p.failure_patience < 1 or p.improvement_tolerance < 0):
            raise ValueError("invalid online-batch parameters or requested batch size")

    def _pool(self, model: Surrogate, X: np.ndarray, y: np.ndarray,
              spacing: float, rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a feasible, unobserved pool inside the current trust region."""
        p = self.params
        centers = X[np.argsort(y, kind="stable")[:min(p.centers, len(X))]]
        radius = self._radius_ratio * spacing
        chunks: list[np.ndarray] = []
        # Lower concentrations widen the candidate cloud if the current radius is small.
        for divisor in (1.0, 2.0, 4.0, 8.0):
            anchors = centers[rng.integers(len(centers), size=p.pool_size)]
            alpha = (p.concentration / divisor) * anchors.reshape(-1, N_BUILDINGS, N_METRICS) + p.alpha_floor
            draws = rng.gamma(alpha)
            points = (draws / draws.sum(axis=-1, keepdims=True)).reshape(-1, DIM)
            distance = nearest_distances(points, X)
            keep = (distance > OBSERVED_TOLERANCE) & (distance <= radius + 1e-12)
            if keep.any():
                chunks.append(points[keep])
            pool = _unique_rows(np.concatenate(chunks)) if chunks else np.empty((0, DIM))
            if len(pool) >= max(n * 8, 128):
                break

        if not chunks:
            raise ValueError("no new candidates inside the current trust region")
        points = _unique_rows(np.concatenate(chunks))
        if len(points) < n:
            raise ValueError(
                f"only {len(points)} new candidates inside trust radius {self._radius_ratio:.3g}; "
                f"requested {n}"
            )
        mean, std = model.predict(points)
        mean, std = np.asarray(mean, dtype=float), np.asarray(std, dtype=float)
        if mean.shape != (len(points),) or std.shape != (len(points),):
            raise ValueError("surrogate returned invalid prediction shapes")
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std < 0).any():
            raise ValueError("surrogate returned non-finite predictions or negative uncertainty")
        return points, mean, std

    def propose(self, model: Surrogate, X, y, n: int = 1) -> np.ndarray:
        """Return a distinct batch, ordered by the pessimistic acquisition objective."""
        self._validate_params(n)
        X = check_weights(as_batch(X))
        y = np.asarray(y, dtype=float)
        if y.shape != (len(X),) or len(X) < 2 or not np.isfinite(y).all():
            raise ValueError("online-batch needs at least two observations and finite aligned labels")

        spacing = median_spacing(X)
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("online-batch needs positive training spacing")
        rng = np.random.default_rng(self.params.seed + self._proposal_index)
        points, mean, std = self._pool(model, X, y, spacing, rng, n)
        objective = mean + self.params.risk * std
        exploit_count = 1 if n == 1 else int(round(n * self.params.exploit_fraction))
        explore_count = n - exploit_count

        exploit_order = np.argsort(objective, kind="stable")
        separation = self.params.min_separation_ratio * spacing
        exploit_indices, used_separation = _diverse_selection(
            points, exploit_order, exploit_count, separation
        )

        if explore_count:
            mean_scale = max(float(np.ptp(mean)), 1e-12)
            std_scale = max(float(np.ptp(std)), 1e-12)
            distance_ratio = nearest_distances(points, X) / spacing
            explore_score = ((mean - mean.min()) / mean_scale
                             - self.params.exploration_weight * (std - std.min()) / std_scale
                             - self.params.diversity_weight * distance_ratio)
            explore_order = np.argsort(explore_score, kind="stable")
            explore_indices, used_separation = _diverse_selection(
                points, explore_order, explore_count, used_separation,
                already=points[exploit_indices],
            )
        else:
            explore_indices = np.empty(0, dtype=int)

        chosen_indices = np.concatenate((exploit_indices, explore_indices))
        # The public contract always returns best-first by the optimizer's objective.
        chosen_indices = chosen_indices[np.argsort(objective[chosen_indices], kind="stable")]
        chosen = check_weights(points[chosen_indices])
        if len(np.unique(np.round(chosen, 9), axis=0)) != len(chosen):
            raise ValueError("online-batch generated duplicate points")
        self._proposal_index += 1
        self.diagnostics = {
            "batch_size": int(n),
            "exploit_count": int(exploit_count),
            "explore_count": int(explore_count),
            "pool_size": int(len(points)),
            "radius_ratio": float(self._radius_ratio),
            "spacing": float(spacing),
            "separation": float(used_separation),
            "best_objective": float(objective[chosen_indices[0]]),
            "best_mean": float(mean[chosen_indices[0]]),
            "best_std": float(std[chosen_indices[0]]),
        }
        return chosen

    def update(self, previous_best: float, batch_best: float) -> dict[str, float | int]:
        """Update the trust radius after a completed evaluator batch."""
        previous_best, batch_best = float(previous_best), float(batch_best)
        if not np.isfinite((previous_best, batch_best)).all():
            raise ValueError("previous_best and batch_best must be finite")
        p = self.params
        improved = batch_best < previous_best - p.improvement_tolerance
        if improved:
            self._success_streak += 1
            self._failure_streak = 0
            if self._success_streak >= p.success_patience:
                self._radius_ratio = min(p.max_radius_ratio, self._radius_ratio * p.expand_factor)
                self._success_streak = 0
        else:
            self._failure_streak += 1
            self._success_streak = 0
            if self._failure_streak >= p.failure_patience:
                self._radius_ratio = max(p.min_radius_ratio, self._radius_ratio * p.shrink_factor)
                self._failure_streak = 0
        self.diagnostics.update({"radius_ratio": float(self._radius_ratio), "improved": bool(improved)})
        return self.state_dict()

    def state_dict(self) -> dict[str, float | int]:
        """Return JSON-serializable state for checkpointing and deterministic resume."""
        return {
            "radius_ratio": float(self._radius_ratio),
            "proposal_index": int(self._proposal_index),
            "success_streak": int(self._success_streak),
            "failure_streak": int(self._failure_streak),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore state written by :meth:`state_dict`."""
        required = {"radius_ratio", "proposal_index", "success_streak", "failure_streak"}
        if not isinstance(state, dict) or set(state) != required:
            raise ValueError(f"online-batch state must contain exactly {sorted(required)}")
        radius = float(state["radius_ratio"])

        integers = []
        for key in ("proposal_index", "success_streak", "failure_streak"):
            value = state[key]
            # JSON has no integer type guarantee: reject fractional and boolean values instead of
            # silently truncating them, otherwise a damaged checkpoint can change the RNG stream.
            if isinstance(value, bool):
                raise ValueError("invalid online-batch state")
            try:
                numeric = float(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("invalid online-batch state") from error
            if not np.isfinite(numeric) or numeric < 0 or numeric != np.floor(numeric):
                raise ValueError("invalid online-batch state")
            integers.append(int(numeric))

        if (not np.isfinite(radius) or not self.params.min_radius_ratio <= radius <= self.params.max_radius_ratio):
            raise ValueError("invalid online-batch state")
        self._radius_ratio = radius
        self._proposal_index, self._success_streak, self._failure_streak = integers
