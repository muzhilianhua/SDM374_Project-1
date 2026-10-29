"""Two boosted models exposed as one predictor for direct pessimistic-consensus optimization."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..interfaces import Surrogate
from ..registry import build_model, register_model
from ..simplex import as_batch


@register_model("boosted-consensus")
class BoostedConsensus(Surrogate):
    """HistGB/XGBoost mean; std is a risk envelope so mean + std = max(member mean + std)."""

    @dataclass(frozen=True)
    class Params:
        seed: int = 2026

    def fit(self, X, y):
        X, y = as_batch(X), np.asarray(y, dtype=float)
        self.members = {name: build_model(name, {"seed": self.params.seed}).fit(X, y)
                        for name in ("histgb", "xgboost")}
        self.metrics = {"members": 2, "risk_envelope_not_standard_deviation": True}
        return self

    def predict(self, X):
        predictions = [model.predict(as_batch(X)) for model in self.members.values()]
        means = np.stack([pair[0] for pair in predictions])
        spreads = np.stack([pair[1] for pair in predictions])
        mean = means.mean(axis=0)
        # The interface calls this std, but this wrapper deliberately reports an acquisition
        # envelope, not a calibrated Gaussian sigma. Only risk=1 gives the minimax identity.
        envelope = np.maximum((means + spreads).max(axis=0) - mean, 0.0)
        return mean, envelope
