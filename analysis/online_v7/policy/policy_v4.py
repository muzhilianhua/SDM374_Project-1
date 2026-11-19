"""v4: v2 plus four changes that use only the run's own results (no locally measured data enters the policy).

1. Null-step filter. A single-building move that changes one coordinate by factor f (or sets a zero
   coordinate to a tiny value t) and returns exactly the base's R lies on the base's plateau. Per building and
   base block, the policy keeps the widest such interval per coordinate and no longer proposes moves inside it.
   In the v1-v3 rehearsals 30 % of the points in calls 11-20 repeated an R already seen in the run.
2. Late coordinate scans (from `scan_call`): for each building, one-coordinate moves just outside the known
   null interval (smallest effective step first), plus moves that apply the previous call's improving
   coordinates of the same building to the current base (E15-style combinations).
3. Adaptive slot shares: per building, slots go to the move classes (model, random, fine, scan) in proportion
   to their posterior hit rate (hits + 1) / (n + 5) over the last `share_window` calls, above floors
   (model 0.25, others 0.10).
   A hit is a point below its call's base R.
4. No point that repeats an already sent point is ever proposed (the v1-v3 prototype lost 5-37 points per run
   to the pre-submit screen); call-1 multi-building jumps are dropped (0 hits in 56 points in v1-v3).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import itertools

import numpy as np

from policy import (BlockDescentPolicy, MINOR, Params, State, TrainPriors, ZERO, bkey, norm_blocks)
from surrogate import Surrogate

EQ = 1e-9
CLASSES = ("model", "random", "fine", "scan")
CLASS_OF = {"model": "model", "random": "random", "fine": "fine", "zero": "fine", "tiny": "fine",
            "scan": "scan", "icombo": "scan", "grid": "scan", "nav": "nav", "fill": "random"}
SCAN_F = [0.0, 0.5, 0.7, 0.85, 0.93, 0.97, 0.99, 1.01, 1.03, 1.07, 1.15, 1.4, 2.0, 3.0]


@dataclass
class ParamsV4(Params):
    scan_call: int = 9
    share_window: int = 4
    share_floor: float = 0.10
    model_floor: float = 0.25   # v3 (fewer surrogate picks) was slower: keep at least half of v2's model share
    n_icombo: int = 3


def one_coord(base_blk, blk):
    """If blk differs from base_blk in exactly one coordinate (up to renormalisation): (k, kind, value).
    kind 'f': positive coordinate multiplied by value; kind 't': zero coordinate set to value (relative to
    degradation + ramping of blk)."""
    pos = base_blk > 0
    if np.any((blk > 0) & ~pos):                       # a zero coordinate became positive
        new = np.flatnonzero((blk > 0) & ~pos)
        if len(new) != 1 or not np.allclose(blk[pos] / base_blk[pos], (blk[pos] / base_blk[pos])[0], rtol=1e-9):
            return None
        k = int(new[0]); s = blk[4] + blk[5]
        return (k, "t", blk[k] / s) if s > 0 else None
    q = blk[pos] / base_blk[pos]
    med = np.median(q)
    off = np.flatnonzero(np.abs(q / med - 1) > 1e-9)
    if len(off) != 1:
        return None
    k = int(np.flatnonzero(pos)[off[0]])
    return k, "f", float(q[off[0]] / med)


class BlockDescentPolicyV4(BlockDescentPolicy):
    def __init__(self, params: ParamsV4 | None = None, seed: int = 2026):
        super().__init__(params or ParamsV4(), seed)

    # ------------------------------------------------------------------ what the run has learned
    def _learn(self, st: State):
        """Null intervals per (building, base block, coordinate), class hit statistics, last improvers."""
        i0 = int(np.argmin(st.y_train))
        inc, R_inc = norm_blocks(st.X_train[i0])[0], float(st.y_train[i0])
        null = defaultdict(lambda: {"lo": 1.0, "hi": 1.0, "tmax": 0.0})
        hist = []                                   # per call: {class: [hits, n]}
        improvers = {}                              # building -> [(delta, base block key, k, kind, value)]
        for call in st.calls:
            base_e = next((e for e in call if e["meta"]["kind"] == "base"), None)
            if base_e is None:
                base, R_base = inc, R_inc
            elif np.isfinite(base_e["R"]):
                base, R_base = base_e["W"], base_e["R"]
            else:
                base, R_base = None, None
            stats = defaultdict(lambda: [0, 0])
            imp = defaultdict(list)
            for e in call:
                m = e["meta"]
                if m["kind"] != "single" or base is None or not np.isfinite(e["R"]):
                    continue
                b = m["building"]
                cls = CLASS_OF.get(m.get("move"), "random")
                d = e["R"] - R_base
                stats[cls][0] += d < -EQ
                stats[cls][1] += 1
                oc = one_coord(base[b], e["W"][b])
                if oc is None:
                    continue
                k, kind, v = oc
                key = (b, bkey(base[b]), k)
                if abs(d) <= EQ:
                    if kind == "f":
                        null[key]["lo"] = min(null[key]["lo"], v)
                        null[key]["hi"] = max(null[key]["hi"], v)
                    else:
                        null[key]["tmax"] = max(null[key]["tmax"], v)
                elif d < -EQ:
                    imp[b].append((d, bkey(base[b]), k, kind, v))
            hist.append(stats)
            improvers = {b: sorted(v) for b, v in imp.items()}
            for e in call:
                if np.isfinite(e["R"]) and e["R"] < R_inc - 1e-12:
                    inc, R_inc = e["W"].copy(), float(e["R"])
        return null, hist, improvers

    def _shares(self, hist, call_no):
        p = self.p
        recent = hist[-p.share_window:]
        classes = [c for c in CLASSES if c != "scan" or call_no >= p.scan_call]
        rate = {}
        for c in classes:
            h = sum(s[c][0] for s in recent if c in s)
            n = sum(s[c][1] for s in recent if c in s)
            rate[c] = (h + 1) / (n + 5)          # Beta(1, 4) prior: a class with no history starts at 0.2
        tot = sum(rate.values())
        raw = {c: rate[c] / tot for c in classes}
        floors = {c: (p.model_floor if c == "model" else p.share_floor) for c in classes}
        free = 1 - sum(floors.values())
        return {c: floors[c] + free * raw[c] for c in classes}

    # ------------------------------------------------------------------ proposal
    def propose(self, st: State, n: int = 50):
        p = self.p
        call_no = len(st.calls) + 1
        rng = np.random.default_rng([self.seed, call_no, 4])
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

        def elites(b):
            items = sorted(val[b].items(), key=lambda kv: kv[1])[: p.n_elite]
            return [blocks[b][k] for k, _ in items]

        # ---- base and cross-building combinations (as v1-v3)
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

        # ---- surrogate pools (null-filtered)
        pools = {}
        for b in range(3):
            parents = [base[b]] + [e for e in elites(b) if bkey(e) != bkey(base[b])]
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

        # ---- per-building slots by adaptive class shares
        shares = self._shares(hist, call_no)
        slots = n - len(out)
        per = [slots // 3 + (1 if b < slots % 3 else 0) for b in range(3)]
        for b in range(3):
            quota = {c: int(np.floor(per[b] * shares[c])) for c in shares}
            for c in sorted(shares, key=lambda c: -shares[c])[: per[b] - sum(quota.values())]:
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
                "model_secs": model.fit_secs, "shares": {c: round(v, 3) for c, v in shares.items()}}
        return out[:n], info

    # ------------------------------------------------------------------ slot fillers
    def _fill_model(self, b, base, pool, k_model, add):
        cand, mu, sd = pool
        if k_model <= 0 or len(cand) == 0:
            return
        lcb = mu - self.p.kappa * sd
        F = self._full(base, b, cand)
        taken = []
        for i in np.argsort(lcb):
            if len(taken) >= k_model:
                break
            if taken and np.min(np.abs(F[taken] - F[i]).max(axis=1)) < 0.02:
                continue
            W = base.copy(); W[b] = cand[i]
            if add(W, {"kind": "single", "building": b, "move": "model", "pred": float(mu[i])}):
                taken.append(i)

    def _fill_moves(self, b, base, scale, k, kinds, label, priors, add, rng, in_null):
        got, tries = 0, 0
        while got < k and tries < 300:
            tries += 1
            kind = kinds[rng.integers(len(kinds))]
            blk = self._move(rng, base[b], scale, kind, priors, b=b)
            if blk is None or in_null(b, base[b], blk):
                continue
            W = base.copy(); W[b] = blk
            got += add(W, {"kind": "single", "building": b, "move": label or kind})

    def _fill_scan(self, b, base, k, null, improvers, priors, add, rng, call_no):
        if k <= 0:
            return
        blk0 = base[b]
        got = 0
        # (a) previous call's improving coordinates of this building, applied to the current base block
        for d, key, kk, kind, v in improvers.get(b, [])[: self.p.n_icombo]:
            if got >= k:
                break
            w = blk0.copy()
            if kind == "f":
                if w[kk] == 0:
                    continue
                w[kk] *= v
            else:
                if w[kk] > 0:
                    continue
                w[kk] = v * (w[4] + w[5])
            if w.sum() <= 0:
                continue
            W = base.copy(); W[b] = w / w.sum()
            got += add(W, {"kind": "single", "building": b, "move": "icombo"})
        # (b) one-coordinate steps just outside the null interval, coordinates in rotating order
        order = [(call_no * 3 + b + j) % 7 for j in range(7)]
        cands = []
        for kk in order:
            nl = null.get((b, bkey(blk0), kk), {"lo": 1.0, "hi": 1.0, "tmax": 0.0})
            if blk0[kk] > 0:
                ups = [f for f in SCAN_F if f > nl["hi"]]
                downs = [f for f in sorted(SCAN_F, reverse=True) if f < nl["lo"]]
                steps = [("f", f) for pair in itertools.zip_longest(ups, downs) for f in pair if f is not None]
            else:
                ts = np.geomspace(priors.tiny_lo, max(priors.tiny_hi, 3 * priors.tiny_lo), 8)
                steps = [("t", t) for t in ts if t > nl["tmax"]]
            cands.append((kk, steps))
        depth = 0
        while got < k and depth < 14:
            for kk, steps in cands:
                if got >= k:
                    break
                if depth >= len(steps):
                    continue
                kind, v = steps[depth]
                w = blk0.copy()
                if kind == "f":
                    w[kk] *= v
                else:
                    w[kk] = v * (w[4] + w[5])
                if w.sum() <= 0:
                    continue
                w = w / w.sum()
                w[w < ZERO] = 0.0
                if w.sum() <= 0:
                    continue
                W = base.copy(); W[b] = w / w.sum()
                got += add(W, {"kind": "single", "building": b, "move": "scan"})
            depth += 1
