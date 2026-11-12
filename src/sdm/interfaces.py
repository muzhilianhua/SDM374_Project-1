"""Contracts that every surrogate model, optimizer and online batch policy plugin implements.

Each plugin declares its tunable settings as a nested frozen dataclass named `Params`.
The registry builds it from the `[model.params]` / `[optimizer.params]` tables of a config,
and the CLI (later the GUI) lists its fields, so adding a plugin needs no other code changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class OnlineState:
    """Everything an online policy may use before call `call`: the training table and earlier results only.

    `X_online` / `y_online` hold the points of successful calls in the order they were sent, and
    `online_call[i]` is the call that measured row i. Before call 1 they are empty, so the first batch can
    only depend on the 2048 training rows (decision 19). `X_unmeasured` holds the points of failed attempts;
    `X_sent` is both together, which the pre-submit check excludes.
    """

    call: int                      # 1-based number of the successful call about to be made
    calls_total: int               # the run's call budget (20)
    calls_left: int                # attempts still allowed, after failed or lost attempts
    batch_size: int                # points per call (50)
    seed: int                      # the run's [online] seed
    X_train: np.ndarray
    y_train: np.ndarray
    X_online: np.ndarray
    y_online: np.ndarray
    online_call: np.ndarray
    # Points of failed or lost attempts: sent to the solver but without an R value. They count as sent and
    # are never proposed again (decision 28), but they are not observations.
    X_unmeasured: np.ndarray = field(default_factory=lambda: np.empty((0, 21)))

    @property
    def X_sent(self) -> np.ndarray:
        """Every point sent to the evaluator so far, measured or not: none of them may be sent again."""
        return np.vstack([self.X_online.reshape(-1, 21), self.X_unmeasured.reshape(-1, 21)])

    @property
    def X(self) -> np.ndarray:
        """Every observation so far, training rows first: what a surrogate is refitted on."""
        return np.vstack([self.X_train, self.X_online])

    @property
    def y(self) -> np.ndarray:
        return np.concatenate([self.y_train, self.y_online])

    @property
    def best(self) -> float:
        """Lowest measured R so far, training rows included."""
        return float(self.y.min())

    @property
    def best_online(self) -> float | None:
        return float(self.y_online.min()) if len(self.y_online) else None


@dataclass(frozen=True)
class Proposal:
    """Candidates from a policy, best first. The online loop screens them and sends the first that pass."""

    weights: np.ndarray                          # (m, 21); each row passes sdm.simplex.check_weights
    predicted_mean: np.ndarray | None = None     # (m,) the policy's prediction, saved for T13/T26
    predicted_std: np.ndarray | None = None      # (m,)
    tags: tuple[str, ...] | None = None          # optional per-row label, e.g. "exploit" / "explore"
    diagnostics: dict = field(default_factory=dict)   # scalars, saved with the call


class BatchPolicy(ABC):
    """Proposes the next online batch from the whole online state (T08). Plugins live in `sdm.policies`.

    A policy fits whatever model it needs inside `propose`; `self.model` and `self.optimizer` hold the
    config's [model] / [optimizer] components (name and params) for policies that use them. `propose`
    must be deterministic given the state and the params, so a resumed or re-run session sends the same
    batches (decision 22).
    """

    name: ClassVar[str]  # set by @register_policy
    Params: ClassVar[type] = NoParams

    def __init__(self, params=None, *, model=None, optimizer=None):
        self.params = params if params is not None else self.Params()
        self.model = model
        self.optimizer = optimizer

    @abstractmethod
    def propose(self, state: OnlineState, n: int) -> Proposal:
        """Return at least `n` candidates if possible, best first; the loop may ask again with a larger n."""
