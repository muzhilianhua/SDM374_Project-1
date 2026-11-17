"""v6: v4's fast single-form descent; a building opens alternative forms only once its own descent stalls.

v5 (lanes from call 1 plus vertex probes) narrowed the seed spread (106.17-106.44) but lost 0.67 on the mean
against v4: lanes compared unrefined forms, extreme probe blocks won early and refined poorly, and the main
form descended more slowly. v6 keeps v4 unchanged while a building improves and changes only this:

* Stall trigger. A building has stalled when none of its singles beat the base in each of its last
  `stall_calls` calls. Only a stalled building, and only up to call `lane_last`, gets lanes.
* Lanes of a stalled building: v5's lane refinement (diverse measured blocks, `lane_share` of its slots) plus
  `spawn` moves that open new forms when fewer than `n_lanes` heads exist: the base block shifted toward one
  objective (drawn from the training-derived favour) by a fraction in `spawn_f`.
* No vertex probes.

Everything uses the training rows and this run's own results only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from policy import ZERO
from policy_v4 import EQ
from policy_v5 import BlockDescentPolicyV5, ParamsV5


@dataclass
class ParamsV6(ParamsV5):
    probe_f: tuple = ()
    stall_calls: int = 2
    lane_last: int = 17
    spawn_share: float = 0.15
    spawn_f: tuple = (0.25, 0.5)


class BlockDescentPolicyV6(BlockDescentPolicyV5):
    def __init__(self, params: ParamsV6 | None = None, seed: int = 2026):
        super().__init__(params or ParamsV6(), seed)

    @staticmethod
    def stalls(st):
        """Per building: number of most recent calls in which no single of that building beat the call's base."""
        stall = [0, 0, 0]
        R_inc = float(np.min(st.y_train))
        for call in st.calls:
            base_e = next((e for e in call if e["meta"]["kind"] == "base"), None)
            R_base = R_inc if base_e is None else base_e["R"]
            for b in range(3):
                best = min((e["R"] for e in call if e["meta"]["kind"] == "single" and e["meta"]["building"] == b
                            and np.isfinite(e["R"])), default=np.inf)
                hit = np.isfinite(R_base) and best < R_base - EQ
                stall[b] = 0 if hit else stall[b] + 1
            R_inc = min([R_inc] + [e["R"] for e in call if np.isfinite(e["R"])])
        return stall

    def _lanes_active(self, st, call_no):
        if call_no > self.p.lane_last:
            return [False] * 3
        return [s >= self.p.stall_calls for s in self.stalls(st)]

    def _n_spawn(self, per_b, active_b, n_heads):
        if not active_b:
            return 0
        k = int(round(per_b * self.p.spawn_share))
        return max(k, 2) if n_heads < self.p.n_lanes else max(k // 2, 1)

    def _spawn(self, rng, blk, priors, b):
        fav = priors.favour[b] if priors.favour is not None else np.full(7, 1 / 7)
        k = rng.choice(7, p=fav)
        f = rng.uniform(*self.p.spawn_f)
        e = np.zeros(7); e[k] = 1.0
        w = (1 - f) * blk + f * e
        w[w < ZERO] = 0.0
        return w / w.sum()
