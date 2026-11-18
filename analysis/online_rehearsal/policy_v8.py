"""v8 = v4 with the fixed-constant block Matérn GP (`surrogate_hp.HPSurrogate`) and v4's kappa 1.

Ablation of v7 (block GP + kappa 0, negative): separates the surrogate change from the kappa change, and is a
decision-31-compliant candidate (no run-time optimiser) for the frozen submission.
"""
from __future__ import annotations

from policy_v4 import BlockDescentPolicyV4, ParamsV4
from surrogate_hp import HPSurrogate


class ParamsV8(ParamsV4):
    """Identical to ParamsV4 (kappa 1.0); a separate name so the run config records the variant."""


class BlockDescentPolicyV8(BlockDescentPolicyV4):
    surrogate_cls = HPSurrogate
