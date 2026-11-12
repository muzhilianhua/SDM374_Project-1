"""Template for an online batch policy. Copy to src/sdm/policies/<name>.py (no leading underscore) and edit.

The registry skips modules whose name starts with "_", so this file itself is never registered.
Contract: docs/interfaces.md ("Online batch policy"); the online loop screens every proposal before sending.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..interfaces import BatchPolicy, OnlineState, Proposal
from ..registry import build_model, build_optimizer, register_policy


@register_policy("template")  # TODO: unique name, used as [policy] name in experiments/online-*.toml
class TemplatePolicy(BatchPolicy):
    """TODO: one-line summary shown by `sdm list`."""

    @dataclass(frozen=True)
    class Params:
        first_call_explore: float = 0.2   # e.g. a different exploit / explore split for the first batch
        explore: float = 0.4

    def propose(self, state: OnlineState, n: int) -> Proposal:
        # state.X / state.y: training rows plus every earlier online result; before call 1 only the 2048 rows.
        model = build_model(self.model.name, self.model.params).fit(state.X, state.y)
        optimizer = build_optimizer(self.optimizer.name, self.optimizer.params)
        share = self.params.first_call_explore if state.call == 1 else self.params.explore
        weights = optimizer.propose(model, state.X, state.y, n)   # TODO: split exploit / explore by `share`
        mean, std = model.predict(weights)
        tags = tuple("explore" if i >= round(n * (1 - share)) else "exploit" for i in range(len(weights)))
        return Proposal(np.asarray(weights), mean, std, tags=tags)  # best first; the loop keeps the first n that pass
