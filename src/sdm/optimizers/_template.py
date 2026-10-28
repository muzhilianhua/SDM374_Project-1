"""Template for an optimizer plugin. Copy to src/sdm/optimizers/<name>.py (no leading underscore) and edit.

The registry skips modules whose name starts with "_", so this file itself is never registered.
Contract: docs/interfaces.md; tests/test_plugin_contract.py checks every registered optimizer automatically.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..interfaces import Optimizer, Surrogate
from ..registry import register_optimizer
from ..simplex import sample


@register_optimizer("template")  # TODO: unique name, used as [optimizer] name in experiments/*.toml
class TemplateOptimizer(Optimizer):
    """TODO: one-line summary shown by `sdm list` and the GUI."""

    @dataclass(frozen=True)
    class Params:
        candidates: int = 20000
        seed: int = 0

    def propose(self, model: Surrogate, X, y, n=1):
        # X, y: every observation so far (training table, plus online results later).
        rng = np.random.default_rng(self.params.seed)
        # TODO: replace random sampling with a real search; keep every row on the simplex
        # (sdm.simplex.project helps) and return n distinct rows.
        points = sample(self.params.candidates, rng)
        mean, std = model.predict(points)
        self.diagnostics = {"best_predicted": float(mean.min())}  # optional, saved to metrics.json
        return points[np.argsort(mean, kind="stable")[:n]]  # (n, 21), best first
