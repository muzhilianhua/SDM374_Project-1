"""Contracts that every surrogate model and optimizer plugin implements.

Each plugin declares its tunable settings as a nested frozen dataclass named `Params`.
The registry builds it from the `[model.params]` / `[optimizer.params]` tables of a config,
and the CLI (later the GUI) lists its fields, so adding a plugin needs no other code changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np


@dataclass(frozen=True)
class NoParams:
    """Settings for a plugin that has none."""


class Surrogate(ABC):
    """Predicts mean ramping and its uncertainty for batches of 21-dim weights."""

    name: ClassVar[str]  # set by @register_model
    Params: ClassVar[type] = NoParams

    def __init__(self, params=None):
        self.params = params if params is not None else self.Params()
        # Scalar diagnostics from fit (e.g. internal CV scores); recorded in each run's metrics.json.
        self.metrics: dict[str, float] = {}

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> Surrogate:
        """Fit on (n, 21) weights and (n,) ramping; return self."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, std), each of shape (n,), for an (n, 21) batch."""

    def gradient(self, X: np.ndarray) -> np.ndarray:
        """Gradient of the predicted mean w.r.t. the weights, shape (n, 21). Needed by gradient optimizers."""
        raise NotImplementedError(f"{type(self).__name__} does not provide gradients")

    def smoothness(self) -> float:
        """Upper bound on the Lipschitz constant of `gradient`; gradient optimizers derive a step size from it."""
        raise NotImplementedError(f"{type(self).__name__} does not provide a smoothness bound")


class Optimizer(ABC):
    """Proposes weights to evaluate next, given a fitted surrogate and all observations so far."""

    name: ClassVar[str]  # set by @register_optimizer
    Params: ClassVar[type] = NoParams

    def __init__(self, params=None):
        self.params = params if params is not None else self.Params()
        self.diagnostics: dict[str, float] = {}

    @abstractmethod
    def propose(self, model: Surrogate, X: np.ndarray, y: np.ndarray, n: int = 1) -> np.ndarray:
        """Return an (n, 21) batch, best first by the optimizer's own objective (e.g. mean or
        mean + k * std); each building's 7 weights non-negative and summing to 1.

        X, y hold every observation so far: the offline table plus any online evaluations.
        """
