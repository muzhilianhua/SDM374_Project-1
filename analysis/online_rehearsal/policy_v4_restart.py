"""A conservative, training-derived restart variant of V4.

V4 is good at refining the block form it has selected, but an early bad form can
make one seed spend the remaining calls in the wrong basin.  This variant keeps
V4's proposal batch and replaces only a few single-building proposals when that
building has stalled for several consecutive calls.

The restart heads are selected at runtime from the lowest training rows and the
run's own measured blocks.  No research log, known low point, or fixed weight is
used.  This file is deliberately kept under ``analysis`` until a rehearsal shows
that the extra exploration does not regress V4.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from policy import bkey, norm_blocks, ZERO
from policy_v4 import BlockDescentPolicyV4, CLASS_OF, EQ, ParamsV4


CLASS_OF["restart"] = "random"


@dataclass
class ParamsV4Restart(ParamsV4):
    """Small generic restart budget; values are policy controls, not point data."""

    restart_after: int = 3
    restart_call: int = 7
    restart_slots: int = 3
    restart_heads: int = 4
    restart_train_quantile: float = 0.20
    restart_jitter: float = 0.06


class BlockDescentPolicyV4Restart(BlockDescentPolicyV4):
    """V4 with a per-building restart only after repeated lack of progress."""

    def __init__(self, params: ParamsV4Restart | None = None, seed: int = 2026):
        super().__init__(params or ParamsV4Restart(), seed=seed)

    @staticmethod
    def _stalls(st, threshold=EQ):
        """Count the current consecutive calls without a single-building hit."""
        stall = [0, 0, 0]
        incumbent = float(np.min(st.y_train))
        for call in st.calls:
            base = next((e for e in call if e["meta"].get("kind") == "base"), None)
            base_r = incumbent if base is None else float(base["R"])
            for b in range(3):
                best = min((float(e["R"]) for e in call
                            if e["meta"].get("kind") == "single"
                            and e["meta"].get("building") == b
                            and np.isfinite(e["R"])), default=np.inf)
                if np.isfinite(base_r) and best < base_r - threshold:
                    stall[b] = 0
                else:
                    stall[b] += 1
            measured = [float(e["R"]) for e in call if np.isfinite(e["R"])]
            if measured:
                incumbent = min(incumbent, min(measured))
        return stall

    @staticmethod
    def _unique_blocks(blocks):
        out = []
        seen = set()
        for block in blocks:
            block = np.asarray(block, float).copy()
            if not np.all(np.isfinite(block)) or block.sum() <= 0:
                continue
            block = block / block.sum()
            block[block < ZERO] = 0.0
            if block.sum() <= 0:
                continue
            block = block / block.sum()
            key = bkey(block)
            if key not in seen:
                seen.add(key)
                out.append(block)
        return out

    def _restart_heads(self, st, b, base_block, rng):
        """Choose diverse alternative blocks using only training/current-run data."""
        p = self.p
        train = norm_blocks(st.X_train)
        order = np.argsort(st.y_train, kind="stable")
        limit = max(32, int(np.ceil(len(order) * p.restart_train_quantile)))

        # Good training rows provide plausible forms; measured blocks provide
        # forms that have already been tested in this run.  Both are dynamic.
        observed = []
        candidates = []
        for call in st.calls:
            for e in call:
                meta = e.get("meta", {})
                if (meta.get("kind") == "single" and meta.get("building") == b
                        and np.isfinite(e["R"])):
                    observed.append((float(e["R"]), e["W"][b]))
        for i in order[:limit]:
            candidates.append((0, float(st.y_train[i]), train[i, b]))
        # A measured single-building block has evaluator evidence in this run,
        # so it is the safer first restart head.  Training heads remain useful
        # as fallbacks and for later diversity selection.
        observed_blocks = self._unique_blocks([x[1] for x in sorted(observed, key=lambda x: x[0])])
        training_blocks = self._unique_blocks(
            [x[2] for x in sorted(candidates, key=lambda x: (x[0], x[1]))])
        pool = observed_blocks + [x for x in training_blocks
                                  if bkey(x) not in {bkey(y) for y in observed_blocks}]
        if not pool:
            return []

        # Farthest-point selection keeps the restart budget spread across
        # different block forms instead of copying one nearby elite repeatedly.
        chosen = []
        remaining = pool[:]
        while remaining and len(chosen) < p.restart_heads:
            if not chosen:
                # Prefer the best observed block when one is available.  This
                # avoids sending a large, unsupported training extrapolation
                # merely because it is far from the incumbent.
                if observed_blocks:
                    chosen.append(remaining.pop(0))
                    continue
                scores = [np.sum(np.abs(x - base_block)) for x in remaining]
            else:
                scores = [min(np.sum(np.abs(x - y)) for y in [base_block] + chosen)
                          for x in remaining]
            # Stable random tie breaking keeps the path deterministic while
            # avoiding a preference for the order of the CSV alone.
            best_score = max(scores)
            tied = [i for i, score in enumerate(scores)
                    if np.isclose(score, best_score, rtol=0.0, atol=1e-12)]
            pick = tied[int(rng.integers(len(tied)))]
            chosen.append(remaining.pop(pick))
        return chosen

    def _make_restarts(self, st, b, base, seen, rng):
        p = self.p
        heads = self._restart_heads(st, b, base[b], rng)
        out = []
        for head_i, head in enumerate(heads):
            variants = [head]
            # A direct head tests the alternative form.  The two local variants
            # give the head a small chance to land on the useful side of a thin
            # basin without spending a full V5-sized lane budget.
            for _ in range(2):
                moved = self._move(rng, head, p.restart_jitter, "multi",
                                   self._priors(st), b=b)
                if moved is not None:
                    variants.append(moved)
            for variant_i, block in enumerate(variants):
                W = base.copy()
                W[b] = block
                W = norm_blocks(W)[0]
                key = tuple(bkey(W[j]) for j in range(3))
                if key in seen:
                    continue
                seen.add(key)
                out.append((W, {"kind": "single", "building": b,
                                "move": "restart", "head": head_i,
                                "variant": variant_i}))
                if len(out) >= p.restart_slots:
                    return out
        return out

    def _priors(self, st):
        # Keep this helper local so restart generation follows the same
        # training-only prior as V4's normal moves.
        from policy import TrainPriors
        return TrainPriors.from_training(st.X_train,
                                         st.y_train if self.p.shift_bias else None)

    def propose(self, st, n=50):
        props, info = super().propose(st, n)
        call_no = len(st.calls) + 1
        if call_no < self.p.restart_call or not st.calls:
            return props, info

        stalls = self._stalls(st)
        active = [b for b, count in enumerate(stalls)
                  if count >= self.p.restart_after]
        if not active:
            info = dict(info, restart_buildings=[], restart_points=0, stalls=stalls)
            return props, info

        # Reconstruct the exact V4 base when it is present in the batch.  If
        # there is no base, V4 uses the current incumbent as its base too.
        base_entry = next((x for x in props if x[1].get("kind") == "base"), None)
        if base_entry is not None:
            base = base_entry[0].copy()
        else:
            measured = [(e["R"], e["W"]) for c in st.calls for e in c
                        if np.isfinite(e["R"])]
            base = min(measured, key=lambda x: x[0])[1].copy() if measured else norm_blocks(st.X_train[np.argmin(st.y_train)])[0]

        seen = {tuple(bkey(W[b]) for b in range(3)) for W, _ in props}
        seen.update(tuple(bkey(e["W"][b]) for b in range(3))
                   for c in st.calls for e in c)
        seen.update(tuple(bkey(W[b]) for b in range(3))
                    for W in norm_blocks(st.X_train))

        replacements = {}
        restart_meta = []
        for b in active:
            rng = np.random.default_rng([self.seed, call_no, 4400 + b])
            candidates = self._make_restarts(st, b, base, seen, rng)
            target = [i for i, (_, meta) in enumerate(props)
                      if meta.get("kind") == "single"
                      and meta.get("building") == b]
            # Replace the tail of this building's V4 allocation.  Earlier
            # V4 choices retain priority, so restart cannot erase the descent.
            for i, candidate in zip(reversed(target), candidates):
                replacements[i] = candidate
                restart_meta.append(b)

        for i, candidate in replacements.items():
            props[i] = candidate
        info = dict(info, restart_buildings=active,
                    restart_points=len(replacements), stalls=stalls)
        return props, info

