"""The offline pipeline, once per call: refit the [model] on every observation, take the [optimizer]'s batch."""
from __future__ import annotations

import time

import numpy as np

from ..interfaces import BatchPolicy, OnlineState, Proposal
from ..registry import build_model, build_optimizer, register_policy


@register_policy("surrogate")
class SurrogatePolicy(BatchPolicy):
    """Refit the config's [model] on all observations each call, then propose with its [optimizer]."""

    def propose(self, state: OnlineState, n: int) -> Proposal:
        if self.model is None or self.optimizer is None:
            raise ValueError("policy 'surrogate' needs [model] and [optimizer] sections")
        X, y = state.X, state.y
        start = time.perf_counter()
        model = build_model(self.model.name, self.model.params).fit(X, y)
        fitted = time.perf_counter()
        optimizer = build_optimizer(self.optimizer.name, self.optimizer.params)
        weights = np.asarray(optimizer.propose(model, X, y, n), dtype=float)
        mean, std = model.predict(weights)
        diagnostics = {"fit_seconds": fitted - start, "propose_seconds": time.perf_counter() - fitted,
                       **{f"optimizer_{key}": value for key, value in optimizer.diagnostics.items()},
                       **{f"model_{key}": value for key, value in model.metrics.items()}}
        return Proposal(weights, np.asarray(mean, dtype=float), np.asarray(std, dtype=float),
                        diagnostics=diagnostics)
