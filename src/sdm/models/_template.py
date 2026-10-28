"""Template for a surrogate plugin. Copy to src/sdm/models/<name>.py (no leading underscore) and edit.

The registry skips modules whose name starts with "_", so this file itself is never registered.
Contract: docs/interfaces.md; tests/test_plugin_contract.py checks every registered model automatically.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..interfaces import Surrogate
from ..registry import register_model
from ..simplex import as_batch


@register_model("template")  # TODO: unique name, used as [model] name in experiments/*.toml
class TemplateModel(Surrogate):
    """TODO: one-line summary shown by `sdm list` and the GUI."""

    @dataclass(frozen=True)
    class Params:
        seed: int = 0  # TODO: every field needs a default; TOML lists arrive as tuples

    def fit(self, X, y):
        X, y = as_batch(X), np.asarray(y, dtype=float)
        # TODO: replace this constant predictor with a real model; build features from X here.
        self.mean_, self.std_ = float(y.mean()), float(y.std())
        self.metrics = {"train_rows": float(len(y))}  # optional scalars, saved to metrics.json
        return self

    def predict(self, X):
        X = as_batch(X)
        # Must return (mean, std), both shape (n,), finite, std >= 0.
        return np.full(len(X), self.mean_), np.full(len(X), self.std_)
