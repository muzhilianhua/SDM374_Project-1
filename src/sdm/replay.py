"""Deterministic, evaluator-free replay of the sequential online loop.

A table lookup, a semi-synthetic surrogate, or a test double can be supplied as ``evaluate``.
The loop mirrors the in-memory part of an online run: fit, propose, validate, evaluate,
append, and update the optimizer state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.spatial.distance import cdist, pdist

from .interfaces import Optimizer, Surrogate
from .simplex import DIM, check_weights


MAX_CALLS = 20
MAX_BATCH = 50
REPLAY_TOLERANCE = 1e-8


@dataclass(frozen=True)
class ReplayConfig:
    """Budget for one replay trajectory."""

    calls: int = MAX_CALLS
    batch_size: int = MAX_BATCH

    def validate(self) -> None:
        if (isinstance(self.calls, bool) or isinstance(self.batch_size, bool)
                or not isinstance(self.calls, (int, np.integer))
                or not isinstance(self.batch_size, (int, np.integer))
                or not 1 <= self.calls <= MAX_CALLS
                or not 1 <= self.batch_size <= MAX_BATCH):
            raise ValueError(f"replay budget must be between 1 and {MAX_CALLS} calls × {MAX_BATCH} points")


@dataclass(frozen=True)
class ReplayRound:
    """One completed replay call, including predictions and truth values."""

    call: int
    points: np.ndarray
    predicted_mean: np.ndarray
    predicted_std: np.ndarray
    values: np.ndarray
    best_after: float
    optimizer_state: dict[str, float | int] | None = None

    @property
    def batch_best(self) -> float:
        """Return the best evaluation from this call alone."""
        return float(self.values.min())


@dataclass(frozen=True)
class ReplayResult:
    """All observations from a replay trajectory."""

    initial_best: float
    rounds: tuple[ReplayRound, ...]

    @property
    def best_curve(self) -> np.ndarray:
        """Return the initial best followed by the best value after each call."""
        return np.asarray([self.initial_best, *(round_.best_after for round_ in self.rounds)], dtype=float)

    @property
    def final_best(self) -> float:
        return self.rounds[-1].best_after if self.rounds else self.initial_best

    @property
    def best_evaluated(self) -> float:
        """Return the best evaluation across calls, excluding the initial observations."""
        return min(round_.batch_best for round_ in self.rounds)

    def to_dict(self) -> dict:
        """Return a JSON-serializable record for replay evidence."""
        return {
            "initial_best": float(self.initial_best),
            "best_curve": self.best_curve.tolist(),
            "best_evaluated": self.best_evaluated,
            "rounds": [
                {
                    "call": round_.call,
                    "points": round_.points.tolist(),
                    "predicted_mean": round_.predicted_mean.tolist(),
                    "predicted_std": round_.predicted_std.tolist(),
                    "values": round_.values.tolist(),
                    "batch_best": round_.batch_best,
                    "best_after": round_.best_after,
                    "optimizer_state": round_.optimizer_state,
                }
                for round_ in self.rounds
            ],
        }


def _validate_predictions(model: Surrogate, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean, std = model.predict(points)
    mean, std = np.asarray(mean, dtype=float), np.asarray(std, dtype=float)
    if mean.shape != (len(points),) or std.shape != (len(points),):
        raise ValueError("surrogate returned invalid prediction shapes during replay")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std < 0).any():
        raise ValueError("surrogate returned invalid predictions during replay")
    return mean.copy(), std.copy()


def _validate_truth(values, count: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape != (count,) or not np.isfinite(values).all():
        raise ValueError("replay evaluator must return one finite value per proposed point")
    return values


def run_replay(
    model_factory: Callable[[], Surrogate],
    optimizer: Optimizer,
    X,
    y,
    evaluate: Callable[[np.ndarray], Sequence[float] | np.ndarray],
    config: ReplayConfig | None = None,
) -> ReplayResult:
    """Run the sequential loop against a replacement evaluator.

    ``X`` and ``y`` are the observations available before the first call. ``optimizer`` persists
    across calls so a stateful trust-region strategy can update its radius. ``model_factory`` is
    called once per call and must return a fresh surrogate. The evaluator receives an ``(n, 21)``
    batch and returns ``n`` values in batch order. A lookup evaluator needs an optimizer that
    proposes rows from its holdout pool. If present, the optimizer's
    ``update(previous_best, batch_best)`` hook may return a JSON-ready state snapshot.
    """
    config = config or ReplayConfig()
    config.validate()
    if not isinstance(optimizer, Optimizer):
        raise TypeError("optimizer must be an sdm.interfaces.Optimizer")

    calls, batch_size = int(config.calls), int(config.batch_size)
    X = check_weights(X).copy()
    y = np.asarray(y, dtype=float).copy()
    if y.shape != (len(X),) or len(X) < 2 or not np.isfinite(y).all():
        raise ValueError("replay needs at least two observations and finite aligned labels")
    initial_best = float(y.min())
    rounds: list[ReplayRound] = []
    update = getattr(optimizer, "update", None)
    for call in range(1, calls + 1):
        model = model_factory()
        if not isinstance(model, Surrogate):
            raise TypeError("model_factory must return an sdm.interfaces.Surrogate")
        if model.fit(X.copy(), y.copy()) is not model:
            raise ValueError("surrogate.fit must return itself during replay")

        previous_best = float(y.min())
        proposed = optimizer.propose(model, X.copy(), y.copy(), n=batch_size)
        points = check_weights(proposed).copy()
        if points.shape != (batch_size, DIM):
            raise ValueError(f"optimizer returned {points.shape}; expected {(batch_size, DIM)}")
        # Compute from coordinate differences; the dot-product identity loses accuracy near 1e-8.
        if np.any(pdist(points, "sqeuclidean") <= REPLAY_TOLERANCE ** 2):
            raise ValueError("replay optimizer returned duplicate points within a batch")
        if np.any(cdist(points, X, "sqeuclidean") <= REPLAY_TOLERANCE ** 2):
            raise ValueError("replay optimizer proposed a previously observed point")

        predicted_mean, predicted_std = _validate_predictions(model, points)
        values = _validate_truth(evaluate(points.copy()), len(points)).copy()
        X = np.vstack((X, points))
        y = np.concatenate((y, values))
        batch_best = float(values.min())
        state = update(previous_best, batch_best) if update is not None else None
        if state is not None and not isinstance(state, dict):
            raise ValueError("stateful replay optimizer update must return a state dictionary")
        rounds.append(ReplayRound(call, points, predicted_mean, predicted_std, values,
                                  min(previous_best, batch_best), None if state is None else dict(state)))

    return ReplayResult(initial_best, tuple(rounds))
