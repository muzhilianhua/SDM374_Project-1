"""v7 = v4 with the fixed-constant block-additive Matérn GP (`surrogate_hp.HPSurrogate`) and kappa = 0.

Both changes come from the offline call-by-call replay in `Project-1/surrogate_compare/` (2026-11-17): on the
unscreened points of the 16 v1-v6 runs, the block GP ranked a call's points better than the T37 surrogate
(within-call Spearman 0.52 vs 0.38; top-5 by mean -1.07 vs -0.65 below the call mean) and, for every model,
ranking by the mean (kappa 0) beat the LCB with kappa 1. The model uses only constants chosen on the training
rows (PR #24) and the run's own data. Choosing it on local rehearsal logs relies on decision 27.
"""
from __future__ import annotations

from dataclasses import dataclass

from policy_v4 import BlockDescentPolicyV4, ParamsV4
from surrogate_hp import HPSurrogate


@dataclass
class ParamsV7(ParamsV4):
    kappa: float = 0.0


class BlockDescentPolicyV7(BlockDescentPolicyV4):
    surrogate_cls = HPSurrogate
