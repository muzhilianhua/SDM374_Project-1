"""Recoverable exploit/explore policy for the T08 online runner (T15/T16)."""
from __future__ import annotations

from dataclasses import replace
import time

import numpy as np

from ..interfaces import BatchPolicy, OnlineState, Proposal
from ..optimizers.online_batch import OnlineBatchOptimizer
from ..registry import build_model, build_optimizer, register_policy


@register_policy("online-batch")
class OnlineBatchPolicy(BatchPolicy):
    """Refit on measured rows and reconstruct the trust radius from completed calls on every proposal."""

    def restored_optimizer(self, state: OnlineState) -> OnlineBatchOptimizer:
        """Derive all optimizer state from the logged observations and charged attempt count.

        Failed/lost evaluations consume RNG positions and exclude sent points, but have no measured
        objective with which to change the radius. Refill requests and dry runs consume no state.
        """
        if self.optimizer is None or self.optimizer.name != "online-batch":
            raise ValueError("policy 'online-batch' needs optimizer 'online-batch'")
        calls = np.asarray(state.online_call)
        if (calls.shape != (len(state.y_online),) or len(state.X_online) != len(calls)
                or not np.isfinite(calls).all() or (calls % 1 != 0).any()
                or (len(calls) and not np.array_equal(np.unique(calls), np.arange(1, state.call)))
                or (not len(calls) and state.call != 1)):
            raise ValueError("online-batch needs contiguous, aligned completed-call history")
        spent = state.calls_total - state.calls_left
        if not 0 <= spent <= state.calls_total or spent < state.call - 1:
            raise ValueError("online-batch received inconsistent attempt quota")
        optimizer = build_optimizer(self.optimizer.name, self.optimizer.params)
        # Both the experiment's optimizer seed and the run seed participate in the stream.
        seed = int(np.random.SeedSequence([optimizer.params.seed, state.seed]).generate_state(1)[0])
        optimizer.params = replace(optimizer.params, seed=seed)
        previous_best = float(np.min(state.y_train))
        for call in range(1, state.call):
            batch_best = float(np.min(state.y_online[calls == call]))
            optimizer.update(previous_best, batch_best)
            previous_best = min(previous_best, batch_best)
        restored = optimizer.state_dict()
        restored["proposal_index"] = int(spent)
        optimizer.load_state_dict(restored)
        return optimizer

    def propose(self, state: OnlineState, n: int) -> Proposal:
        if self.model is None:
            raise ValueError("policy 'online-batch' needs a [model] section")
        if state.calls_left <= 0:
            raise ValueError("online-batch cannot propose with no attempts left")
        optimizer = self.restored_optimizer(state)
        before = optimizer.state_dict()
        start = time.perf_counter()
        model = build_model(self.model.name, self.model.params).fit(state.X, state.y)
        fitted = time.perf_counter()
        weights = optimizer.propose_candidates(model, state.X, state.y, n,
                                               excluded=state.X_unmeasured,
                                               batch_size=state.batch_size)
        mean, std = model.predict(weights)
        diagnostics = {"fit_seconds": fitted - start,
                       "propose_seconds": time.perf_counter() - fitted,
                       **{f"state_{key}": value for key, value in before.items()},
                       **{f"optimizer_{key}": value for key, value in optimizer.diagnostics.items()},
                       **{f"model_{key}": value for key, value in model.metrics.items()}}
        return Proposal(weights, np.asarray(mean), np.asarray(std), optimizer.proposal_tags, diagnostics)
