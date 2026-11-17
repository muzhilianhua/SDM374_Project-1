"""v5: v4 plus per-building lanes, so a run does not commit to one block form per building in its first calls.

Diagnosis of v1-v4 (own rehearsal logs only): every run fixed the dominant objectives ("form") of each
building's block by call 3-8 and kept it to call 20; the descent then only refined that form. Runs that locked a
poor form for one building ended 2-5 above the best run. The policy cannot know which form is good, so v5 keeps
several and lets measured values decide, using only the run's own results:

1. Vertex probes (calls 1-2). Per building, the base block is mixed toward each of the 7 objective vertices
   (fraction 0.5 in call 1, 0.8 in call 2). This is a uniform design over directions, not a prior on location.
2. Lanes (calls 1..lane_calls). Per building, the lowest-valued measured blocks that differ from each other by an
   L1 distance >= lane_dist are lane heads; head 0 is the base block. A share of each building's slots refines
   heads 1.. with local moves, and the surrogate pool draws its non-base parents from the heads instead of the
   nearest elites. A lane that beats the base becomes the base through the usual combination step, because all
   singles of a call are measured against the same base.
3. After lane_calls the policy is v4.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools

import numpy as np

from policy import ZERO, bkey, norm_blocks, State, TrainPriors
from policy_v4 import BlockDescentPolicyV4, ParamsV4, CLASS_OF
from surrogate import Surrogate

CLASS_OF.setdefault("lane", "lane")       # not in CLASSES: excluded from the adaptive shares
CLASS_OF.setdefault("probe", "probe")
CLASS_OF.setdefault("spawn", "spawn")


@dataclass
class ParamsV5(ParamsV4):
    n_lanes: int = 3
    lane_dist: float = 0.4
    lane_calls: int = 12
    lane_share: float = 0.3
    probe_f: tuple = (0.5, 0.8)       # vertex-probe fraction per probe call (call 1, call 2, ...)


class BlockDescentPolicyV5(BlockDescentPolicyV4):
    def __init__(self, params: ParamsV5 | None = None, seed: int = 2026):
        super().__init__(params or ParamsV5(), seed)

    @staticmethod
    def _heads(val_b, blocks_b, base_blk, n, dist):
        heads = [base_blk]
        for k, _ in sorted(val_b.items(), key=lambda kv: kv[1]):
            if len(heads) >= n:
                break
            blk = blocks_b[k]
            if all(np.abs(blk - h).sum() >= dist for h in heads):
                heads.append(blk)
        return heads

    def propose(self, st: State, n: int = 50):
        p = self.p
        call_no = len(st.calls) + 1
        rng = np.random.default_rng([self.seed, call_no, 5])
        priors = TrainPriors.from_training(st.X_train, st.y_train if p.shift_bias else None)
        inc, R_inc, val, blocks, scale = self._replay(st)
        null, hist, improvers = self._learn(st)

        Xo = [e["W"].reshape(21) for c in st.calls for e in c if np.isfinite(e["R"])]
        yo = [e["R"] for c in st.calls for e in c if np.isfinite(e["R"])]
        X = np.vstack([st.X_train] + ([np.array(Xo)] if Xo else []))
        y = np.concatenate([st.y_train, np.array(yo)]) if yo else st.y_train.copy()
        model = Surrogate(local_rows=p.local_rows, seed=self.seed).fit(X, y, center=inc.reshape(21))

        out, seen = [], set()
        sent = {tuple(bkey(e["W"][b]) for b in range(3)) for c in st.calls for e in c}
        seen |= sent
        seen.add(tuple(bkey(inc[b]) for b in range(3)))
        train_keys = getattr(self, "_train_keys", None)
        if train_keys is None:
            Wt = norm_blocks(st.X_train)
            self._train_keys = train_keys = {tuple(bkey(w[b]) for b in range(3)) for w in Wt}
        seen |= train_keys

        def add(W, meta):
            W = norm_blocks(W)[0]
            k = tuple(bkey(W[b]) for b in range(3))
            if k in seen:
                return False
            seen.add(k)
            out.append((W, meta))
            return True

        def in_null(b, base_blk, blk):
            from policy_v4 import one_coord
            oc = one_coord(base_blk, blk)
            if oc is None:
                return False
            k, kind, v = oc
            nl = null.get((b, bkey(base_blk), k))
            if nl is None:
                return False
            if kind == "f":
                return nl["lo"] <= v <= nl["hi"]
            return v <= nl["tmax"]

        # ---- base and cross-building combinations (as v4)
        v_inc = [val[b][bkey(inc[b])] for b in range(3)]
        better = []
        for b in range(3):
            items = sorted(val[b].items(), key=lambda kv: kv[1])
            better.append([(blocks[b][k], v - v_inc[b]) for k, v in items if v < v_inc[b] - 1e-9][:2])
        base = inc.copy()
        combos = []
        for choice in itertools.product(*[[(inc[b], 0.0)] + better[b] for b in range(3)]):
            W = np.array([c[0] for c in choice])
            if all(bkey(W[b]) == bkey(inc[b]) for b in range(3)):
                continue
            combos.append((R_inc + sum(c[1] for c in choice), W))
        combos.sort(key=lambda t: t[0])
        combos = [(pr, W) for pr, W in combos if tuple(bkey(W[b]) for b in range(3)) not in sent]
        if combos:
            base = combos[0][1]
            add(base, {"kind": "base", "pred": combos[0][0]})
            for pr, W in combos[1: 1 + p.n_combo]:
                add(W, {"kind": "combo", "pred": pr})

        active = self._lanes_active(st, call_no)
        lanes_on = any(active)
        heads = [self._heads(val[b], blocks[b], base[b], p.n_lanes if active[b] else 1, p.lane_dist)
                 for b in range(3)]

        # ---- vertex probes
        if call_no <= len(p.probe_f):
            f = p.probe_f[call_no - 1]
            for b in range(3):
                for k in range(7):
                    e = np.zeros(7); e[k] = 1.0
                    W = base.copy(); W[b] = (1 - f) * base[b] + f * e
                    add(W, {"kind": "single", "building": b, "move": "probe", "vertex": k, "f": f})

        # ---- surrogate pools: non-base parents are the lane heads while lanes are on, else v4's elites
        def elites(b):
            items = sorted(val[b].items(), key=lambda kv: kv[1])[: p.n_elite]
            return [blocks[b][k] for k, _ in items]

        pools = {}
        for b in range(3):
            others = heads[b][1:] if active[b] and len(heads[b]) > 1 else elites(b)
            parents = [base[b]] + [e for e in others if bkey(e) != bkey(base[b])]
            cand = []
            kinds = ["scale1", "scale1", "multi", "shift", "zero", "tiny"]
            for _ in range(p.pool):
                par = parents[0] if rng.random() < 0.6 or len(parents) == 1 else parents[rng.integers(1, len(parents))]
                blk = self._move(rng, par, scale[b], kinds[rng.integers(len(kinds))], priors, b=b)
                if blk is not None and not in_null(b, base[b], blk):
                    cand.append(blk)
            cand = np.array(cand)
            Wfull = np.repeat(base[None], len(cand), 0)
            Wfull[:, b] = cand
            mu, sd = model.predict(Wfull.reshape(-1, 21))
            pools[b] = (cand, mu, sd)

        # ---- v2 navigation moves
        if p.n_nav and call_no <= p.nav_calls:
            for b in range(3):
                got, tries = 0, 0
                while got < p.n_nav and tries < 100:
                    tries += 1
                    blk = self._move(rng, base[b], max(scale[b], 0.5), "shift", priors, b=b)
                    if blk is None:
                        continue
                    W = base.copy(); W[b] = blk
                    got += add(W, {"kind": "single", "building": b, "move": "nav"})

        # ---- per-building slots: lane refinement first, the rest by v4's adaptive class shares
        shares = self._shares(hist, call_no)
        slots = n - len(out)
        per = [slots // 3 + (1 if b < slots % 3 else 0) for b in range(3)]
        for b in range(3):
            k_lane = int(round(per[b] * p.lane_share)) if active[b] and len(heads[b]) > 1 else 0
            k_spawn = self._n_spawn(per[b], active[b], len(heads[b]))
            if k_spawn:
                got, tries = 0, 0
                while got < k_spawn and tries < 200:
                    tries += 1
                    blk = self._spawn(rng, base[b], priors, b)
                    if blk is None:
                        continue
                    W = base.copy(); W[b] = blk
                    got += add(W, {"kind": "single", "building": b, "move": "spawn"})
            if k_lane:
                lane_heads = heads[b][1:]
                w_rank = np.array([0.6, 0.4, 0.3, 0.2][: len(lane_heads)])
                q = np.floor(k_lane * w_rank / w_rank.sum()).astype(int)
                q[0] += k_lane - q.sum()
                lsc = min(max(scale[b], 0.15), 0.5)
                for h, qh in zip(lane_heads, q):
                    got, tries = 0, 0
                    while got < qh and tries < 200:
                        tries += 1
                        kind = ["scale1", "scale1", "multi", "fine", "zero"][rng.integers(5)]
                        blk = self._move(rng, h, lsc, kind, priors, b=b)
                        if blk is None:
                            continue
                        W = base.copy(); W[b] = blk
                        got += add(W, {"kind": "single", "building": b, "move": "lane"})
            rest = per[b] - k_lane - k_spawn
            quota = {c: int(np.floor(rest * shares[c])) for c in shares}
            for c in sorted(shares, key=lambda c: -shares[c])[: rest - sum(quota.values())]:
                quota[c] += 1
            self._fill_model(b, base, pools[b], quota.get("model", 0), add)
            if "scan" in quota:
                self._fill_scan(b, base, quota["scan"], null, improvers, priors, add, rng, call_no)
            self._fill_moves(b, base, scale[b], quota.get("random", 0), ["scale1", "multi", "shift"], "random",
                             priors, add, rng, in_null)
            self._fill_moves(b, base, scale[b], quota.get("fine", 0), ["fine", "fine", "zero", "tiny"], None,
                             priors, add, rng, in_null)
        tries = 0
        while len(out) < n and tries < 2000:
            tries += 1
            b = int(rng.integers(3))
            blk = self._move(rng, base[b], scale[b], "scale1", priors)
            if blk is None or in_null(b, base[b], blk):
                continue
            W = base.copy(); W[b] = blk
            add(W, {"kind": "single", "building": b, "move": "fill"})
        info = {"call": call_no, "R_inc": R_inc, "scale": [round(s, 4) for s in scale],
                "model_secs": model.fit_secs, "shares": {c: round(v, 3) for c, v in shares.items()},
                "lanes": [len(h) for h in heads], "active": [bool(a) for a in active]}
        return out[:n], info

    # ------------------------------------------------------------------ hooks (v6 overrides them)
    def _lanes_active(self, st, call_no):
        return [call_no <= self.p.lane_calls] * 3

    def _n_spawn(self, per_b, active_b, n_heads):
        return 0

    def _spawn(self, rng, blk, priors, b):
        return None
