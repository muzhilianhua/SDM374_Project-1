"""Additive block-descent batch policy for the online phase (rehearsal prototype).

Data boundary (decisions 19, 27, 29): the policy sees only the 2048 training rows and the results of
the calls it made itself in this run. It reads no research log, embeds no locally measured point or R
value, and every structural constant below is either generic (step sizes, batch split) or computed at run
time from the training rows (`TrainPriors`).

Idea
----
* Navigation model (`Surrogate`): per-building ratio features, a linear trend fitted on all data (it
  carries the search out of the training cloud) plus a GP on the residuals near the incumbent.
* Buildings are treated as additive (checked on the training rows only, see README). Each call changes
  one building per candidate while the other two stay at a *base* point, so R - R(base) is that block's
  delta. Improving blocks of different buildings are combined into the next base, whose R is then
  predicted almost exactly and measured in the same batch.
* Per-building trust scale (log10 sigma of multiplicative moves): grows on success, shrinks on failure,
  never below a floor; a fixed share of every building's slots is spent on small unscreened moves, zero
  toggles, tiny weights and (late) one-coordinate log-grid scans, because only many samples help at the
  finest scale (batch minimum is the score).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, asdict
import itertools

import numpy as np

from surrogate import Surrogate, features

MINOR = [0, 1, 2, 3, 6]   # cost, carbon, peak, discomfort, resilience
ZERO = 1e-6               # weights below this (relative to the building) are sent as exact zeros


def norm_blocks(W):
    W = np.asarray(W, float).reshape(-1, 3, 7).copy()
    W = W / W.sum(axis=2, keepdims=True)
    W[W < ZERO] = 0.0
    return W / W.sum(axis=2, keepdims=True)


def bkey(block) -> tuple:
    return tuple(np.round(np.asarray(block, float) / np.sum(block), 12))


@dataclass
class TrainPriors:
    """Constants derived from the 2048 training rows only."""
    tiny_lo: float     # empirical absolute-share quantile used as a minor/deg+ramp ratio
    tiny_hi: float
    favour: np.ndarray | None = None   # (3, 7) shift-target probabilities from training Spearman

    @classmethod
    def from_training(cls, X, y=None):
        W = np.asarray(X, float).reshape(-1, 3, 7)
        W = W / W.sum(axis=2, keepdims=True)
        pos = W[W >= ZERO]
        pr = cls(tiny_lo=float(np.quantile(pos, 0.001)), tiny_hi=float(np.quantile(pos, 0.10)))
        if y is not None:
            # per building, how strongly a larger share of each objective goes with lower R in the training rows
            from scipy.stats import spearmanr
            fav = np.zeros((3, 7))
            for b in range(3):
                for k in range(7):
                    rho = spearmanr(W[:, b, k], y).correlation
                    fav[b, k] = max(0.0, -rho) + 0.02
            pr.favour = fav / fav.sum(axis=1, keepdims=True)
        return pr


@dataclass
class Params:
    n_combo: int = 6            # combination points per call (besides the base)
    n_jump_first: int = 8       # call 1: multi-building jumps built from model-predicted best blocks
    scale0: float = 0.5         # initial log10 sigma of a multiplicative move
    scale_up: float = 1.5
    scale_down: float = 0.6
    scale_min: float = 0.01
    scale_max: float = 1.0
    share_model: float = 0.5    # building slots chosen by the surrogate from moves at the trust scale
    share_random: float = 0.25  # unscreened moves at the trust scale
    # the rest: fine lottery (small multiplicative moves, zero toggles, tiny weights, coordinate scans)
    late_call: int = 13         # from this call on, part of the fine slots are coordinate log-grid scans
    pool: int = 1500            # candidates per building and call for surrogate screening
    kappa: float = 1.0          # lower confidence bound mean - kappa * std
    n_elite: int = 6
    local_rows: int = 600
    # v2: navigation biased by the training rows only
    shift_bias: bool = False    # shift-move target drawn from TrainPriors.favour instead of uniformly
    n_nav: int = 0              # per building and call: unscreened biased shift moves while nav is on
    nav_calls: int = 6
    # v3: after the navigation calls, fewer surrogate picks (they were optimistic and hit less often)
    share_model_late: float | None = None
    share_random_late: float | None = None
    # streams drawn per call: v4's own proposals use v4_stream; v7 adds replace_stream for the multiscale slots.
    v4_stream: int = 4


@dataclass
class State:
    """What the policy knows: training rows and its own online results."""
    X_train: np.ndarray
    y_train: np.ndarray
    calls: list = field(default_factory=list)   # per call: list of dict(W=(3,7), R=float|nan, meta=dict)


class BlockDescentPolicy:
    def __init__(self, params: Params | None = None, seed: int = 2026):
        self.p = params or Params()
        self.seed = seed

    # ------------------------------------------------------------------ bookkeeping
    def _replay(self, st: State):
        """Rebuild incumbent, per-building block values and trust scales from the call history."""
        p = self.p
        i0 = int(np.argmin(st.y_train))
        inc = norm_blocks(st.X_train[i0])[0]
        R_inc = float(st.y_train[i0])
        val = [{bkey(inc[b]): 0.0} for b in range(3)]           # per-building additive value v_b(block)
        blocks = [{bkey(inc[b]): inc[b].copy()} for b in range(3)]
        scale = [p.scale0] * 3
        for call in st.calls:
            # 1) base of this call: measured (or known) R of the base point
            base_e = next((e for e in call if e["meta"]["kind"] == "base"), None)
            if base_e is not None and np.isfinite(base_e["R"]):
                base, R_base = base_e["W"], base_e["R"]
            elif base_e is None:
                base, R_base = inc, R_inc
            else:                                               # base failed: singles cannot be attributed
                base, R_base = None, None
            improved = [False] * 3
            v_base = [val[b].get(bkey(base[b])) if base is not None else None for b in range(3)]
            if base is not None:
                # base blocks without a value get the measured surplus split over them
                unknown = [b for b in range(3) if v_base[b] is None]
                if unknown:
                    known = sum(v_base[b] for b in range(3) if v_base[b] is not None)
                    v_inc_sum = sum(val[b][bkey(inc[b])] for b in range(3))
                    surplus = (R_base - R_inc) + v_inc_sum - known
                    for b in unknown:
                        v_base[b] = surplus / len(unknown)
                        val[b][bkey(base[b])] = v_base[b]
                        blocks[b][bkey(base[b])] = base[b].copy()
            for e in call:
                m = e["meta"]
                if m["kind"] != "single" or base is None or not np.isfinite(e["R"]):
                    continue
                b = m["building"]
                k = bkey(e["W"][b])
                v = v_base[b] + (e["R"] - R_base)
                val[b][k] = v
                blocks[b][k] = e["W"][b].copy()
            # 2) new incumbent: best measured point so far
            for e in call:
                if np.isfinite(e["R"]) and e["R"] < R_inc - 1e-12:
                    old = inc
                    inc, R_inc = e["W"].copy(), float(e["R"])
                    for b in range(3):
                        if bkey(inc[b]) != bkey(old[b]):
                            improved[b] = True
                        if bkey(inc[b]) not in val[b]:   # multi-building point with an unvalued block
                            val[b][bkey(inc[b])] = val[b][bkey(old[b])]
                            blocks[b][bkey(inc[b])] = inc[b].copy()
            # 3) a building "succeeds" if its best single beat the base
            for b in range(3):
                best_single = min((e["R"] for e in call if e["meta"]["kind"] == "single"
                                   and e["meta"]["building"] == b and np.isfinite(e["R"])), default=np.inf)
                if improved[b] or (R_base is not None and best_single < R_base - 1e-9):
                    scale[b] = min(p.scale_max, scale[b] * p.scale_up)
                elif R_base is not None:
                    scale[b] = max(p.scale_min, scale[b] * p.scale_down)
        return inc, R_inc, val, blocks, scale

    # ------------------------------------------------------------------ moves
    def _move(self, rng, parent, scale, kind, priors, grid=None, b=None):
        w = parent.copy()
        pos = np.flatnonzero(w > 0)
        if kind == "scale1":
            k = rng.integers(7)
            if w[k] == 0:
                # A zero coordinate is revived with an absolute normalized share.
                w[k] = np.exp(rng.uniform(np.log(priors.tiny_lo), np.log(priors.tiny_hi)))
            else:
                w[k] *= 10 ** rng.normal(0, scale)
        elif kind == "multi":
            w[pos] *= 10 ** rng.normal(0, scale / 3, len(pos))
        elif kind == "shift":                     # move mass toward one objective (vertex direction)
            if self.p.shift_bias and b is not None and priors.favour is not None:
                k = rng.choice(7, p=priors.favour[b])
            else:
                k = rng.integers(7)
            f = min(0.9, abs(rng.normal(0, scale / 2)))
            e = np.zeros(7); e[k] = 1
            w = (1 - f) * w + f * e
        elif kind == "zero":
            cand = [k for k in pos if k in MINOR]
            if not cand or len(pos) < 2:
                return None
            w[rng.choice(cand)] = 0.0
        elif kind == "tiny":
            k = rng.choice(MINOR)
            # Historical rehearsal semantics use the empirical quantiles as ratios to deg+ramp mass.
            w[k] = np.exp(rng.uniform(np.log(priors.tiny_lo), np.log(priors.tiny_hi))) * w[[4, 5]].sum()
        elif kind == "grid":
            k, value = grid
            w[k] = value * w[[4, 5]].sum()
        elif kind == "fine":
            k = rng.integers(7)
            if w[k] == 0:
                return None
            w[k] *= 10 ** rng.normal(0, rng.choice([0.002, 0.006, 0.02]))
        if w.sum() <= 0 or not np.all(np.isfinite(w)):
            return None
        w = w / w.sum()
        w[w < ZERO] = 0.0
        if w.sum() <= 0:
            return None
        return w / w.sum()

    # ------------------------------------------------------------------ proposal
    def propose(self, st: State, n: int = 50):
        p = self.p
        call_no = len(st.calls) + 1
        rng = np.random.default_rng([self.seed, call_no])
        priors = TrainPriors.from_training(st.X_train, st.y_train if self.p.shift_bias else None)
        inc, R_inc, val, blocks, scale = self._replay(st)

        # data for the surrogate: training rows + every measured online point
        Xo = [e["W"].reshape(21) for c in st.calls for e in c if np.isfinite(e["R"])]
        yo = [e["R"] for c in st.calls for e in c if np.isfinite(e["R"])]
        X = np.vstack([st.X_train] + ([np.array(Xo)] if Xo else []))
        y = np.concatenate([st.y_train, np.array(yo)]) if yo else st.y_train.copy()
        model = Surrogate(local_rows=p.local_rows, seed=self.seed).fit(X, y, center=inc.reshape(21))

        out = []          # list of (W(3,7), meta)
        seen = set()

        def add(W, meta):
            W = norm_blocks(W)[0]
            k = tuple(bkey(W[b]) for b in range(3))
            if k in seen:
                return False
            seen.add(k)
            out.append((W, meta))
            return True

        seen.add(tuple(bkey(inc[b]) for b in range(3)))     # the incumbent is already measured

        # elites per building: lowest additive value
        def elites(b):
            items = sorted(val[b].items(), key=lambda kv: kv[1])[: p.n_elite]
            return [blocks[b][k] for k, _ in items]

        # ---- base and combinations
        v_inc = [val[b][bkey(inc[b])] for b in range(3)]
        better = []
        for b in range(3):
            items = sorted(val[b].items(), key=lambda kv: kv[1])
            better.append([(blocks[b][k], v - v_inc[b]) for k, v in items if v < v_inc[b] - 1e-9][:2])
        base = inc.copy()
        combos = []
        opts = [[(inc[b], 0.0)] + better[b] for b in range(3)]
        for choice in itertools.product(*opts):
            W = np.array([c[0] for c in choice])
            if all(bkey(W[b]) == bkey(inc[b]) for b in range(3)):
                continue
            combos.append((R_inc + sum(c[1] for c in choice), W))
        combos.sort(key=lambda t: t[0])
        # skip combos already measured (a single that improved is its own combo)
        measured = {tuple(bkey(e["W"][b]) for b in range(3)) for c in st.calls for e in c}
        combos = [(pr, W) for pr, W in combos if tuple(bkey(W[b]) for b in range(3)) not in measured]
        if combos:
            base = combos[0][1]
            add(base, {"kind": "base", "pred": combos[0][0]})
            for pr, W in combos[1: 1 + p.n_combo]:
                add(W, {"kind": "combo", "pred": pr})

        # ---- call 1: navigation jumps from model-predicted best blocks (multi-building)
        pools = {}
        for b in range(3):
            parents = [base[b]] + [e for e in elites(b) if bkey(e) != bkey(base[b])]
            cand = []
            kinds = ["scale1", "scale1", "multi", "shift", "zero", "tiny"]
            for _ in range(p.pool):
                par = parents[0] if rng.random() < 0.6 or len(parents) == 1 else parents[rng.integers(1, len(parents))]
                blk = self._move(rng, par, scale[b], kinds[rng.integers(len(kinds))], priors, b=b)
                if blk is not None:
                    cand.append(blk)
            cand = np.array(cand)
            Wfull = np.repeat(base[None], len(cand), 0)
            Wfull[:, b] = cand
            mu, sd = model.predict(Wfull.reshape(-1, 21))
            pools[b] = (cand, mu, sd)
        if call_no == 1:
            mu_base = model.predict(base.reshape(1, 21))[0][0]
            gains = []
            for b in range(3):
                cand, mu, sd = pools[b]
                order = np.argsort(mu)
                gains.append([(cand[i], mu[i] - mu_base) for i in order[:6]])
            jumps = []
            for idx in itertools.product(range(6), repeat=3):
                W = np.array([gains[b][idx[b]][0] for b in range(3)])
                jumps.append((sum(gains[b][idx[b]][1] for b in range(3)), W))
            jumps.sort(key=lambda t: t[0])
            for i in range(0, len(jumps), max(1, len(jumps) // (4 * p.n_jump_first))):
                if sum(1 for _, m in out if m["kind"] == "jump") >= p.n_jump_first:
                    break
                add(jumps[i][1], {"kind": "jump", "pred": float(mu_base + jumps[i][0])})

        # ---- singles: one building changed, others at base
        if p.n_nav and call_no <= p.nav_calls:      # v2: unscreened biased navigation moves
            for b in range(3):
                got, tries = 0, 0
                while got < p.n_nav and tries < 100:
                    tries += 1
                    blk = self._move(rng, base[b], max(scale[b], 0.5), "shift", priors, b=b)
                    if blk is None:
                        continue
                    W = base.copy(); W[b] = blk
                    got += add(W, {"kind": "single", "building": b, "move": "nav"})
        slots = n - len(out)
        per = [slots // 3 + (1 if b < slots % 3 else 0) for b in range(3)]
        for b in range(3):
            cand, mu, sd = pools[b]
            late = p.share_model_late is not None and call_no > p.nav_calls
            k_model = int(round(per[b] * (p.share_model_late if late else p.share_model)))
            k_rand = int(round(per[b] * (p.share_random_late if late else p.share_random)))
            k_fine = per[b] - k_model - k_rand
            # model picks: lower confidence bound, with a minimum feature distance
            lcb = mu - p.kappa * sd
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
            # unscreened moves at the trust scale
            tries = 0
            got = 0
            while got < k_rand and tries < 200:
                tries += 1
                blk = self._move(rng, base[b], scale[b], ["scale1", "multi", "shift"][rng.integers(3)], priors, b=b)
                if blk is None:
                    continue
                W = base.copy(); W[b] = blk
                got += add(W, {"kind": "single", "building": b, "move": "random"})
            # fine lottery: small moves, zero toggles, tiny weights, late coordinate scans
            got = 0
            tries = 0
            grid_items = []
            if call_no >= p.late_call:
                ks = MINOR
                k = ks[(call_no + b) % len(ks)]
                vals = np.geomspace(priors.tiny_lo, max(priors.tiny_hi, 3 * priors.tiny_lo), 8)
                grid_items = [(k, v) for v in vals]
            while got < k_fine and tries < 300:
                tries += 1
                if grid_items and got < k_fine // 2:
                    blk = self._move(rng, base[b], scale[b], "grid", priors, grid=grid_items.pop(0))
                    kind = "grid"
                else:
                    kind = ["fine", "fine", "zero", "tiny"][rng.integers(4)]
                    blk = self._move(rng, base[b], scale[b], kind, priors)
                if blk is None:
                    continue
                W = base.copy(); W[b] = blk
                got += add(W, {"kind": "single", "building": b, "move": kind})
        # top up if anything was skipped
        while len(out) < n:
            b = int(rng.integers(3))
            blk = self._move(rng, base[b], scale[b], "scale1", priors)
            if blk is None:
                continue
            W = base.copy(); W[b] = blk
            add(W, {"kind": "single", "building": b, "move": "fill"})
        info = {"call": call_no, "R_inc": R_inc, "scale": [round(s, 4) for s in scale],
                "model_secs": model.fit_secs}
        return out[:n], info

    @staticmethod
    def _full(base, b, cand):
        Wf = np.repeat(base[None], len(cand), 0)
        Wf[:, b] = cand
        return features(Wf.reshape(-1, 21))


# ==================================================================================================
# v4 (formerly policy_v4.py, unchanged)
#
# v4: v2 plus four changes that use only the run's own results (no locally measured data enters the policy).
#
# 1. Null-step filter. A single-building move that changes one coordinate by factor f (or sets a zero
#    coordinate to a tiny value t) and returns exactly the base's R lies on the base's plateau. Per building and
#    base block, the policy keeps the widest such interval per coordinate and no longer proposes moves inside it.
#    In the v1-v3 rehearsals 30 % of the points in calls 11-20 repeated an R already seen in the run.
# 2. Late coordinate scans (from `scan_call`): for each building, one-coordinate moves just outside the known
#    null interval (smallest effective step first), plus moves that apply the previous call's improving
#    coordinates of the same building to the current base (E15-style combinations).
# 3. Adaptive slot shares: per building, slots go to the move classes (model, random, fine, scan) in proportion
#    to their posterior hit rate (hits + 1) / (n + 5) over the last `share_window` calls, above floors
#    (model 0.25, others 0.10).
#    A hit is a point below its call's base R.
# 4. No point that repeats an already sent point is ever proposed (the v1-v3 prototype lost 5-37 points per run
#    to the pre-submit screen); call-1 multi-building jumps are dropped (0 hits in 56 points in v1-v3).
# ==================================================================================================
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


# ==================================================================================================
# v6 / v7 (formerly policy_v6.py and policy_v7.py, merged here for the single-file submission)
#
# v6: calls 1-11 stay exactly v4; from call 12 the basin is searched with several step scales at once
# (coarse single-coordinate jumps, 2-3 coordinate block moves, block restarts by blending with training
# and elite blocks, plus v4's fine moves). A building whose singles stopped improving for
# `stagnation_window` calls uses the wider `wide_sigma`.
# v7: keeps v4's complete batch of 50 points from call 12 on and only replaces slots that produced no
# new best in the v4 seed-7 rehearsal (fine -> zero -> combo, at most `max_replace`) with multiscale
# candidates (coarse / block / restart), so every model, scan and random pick v4 made is preserved.
# All randomness still derives from (seed, call_no); no locally measured point or R is embedded.
# ==================================================================================================
COARSE_F = [0.2, 0.3, 0.45, 0.6, 0.75, 0.9, 1.15, 1.35, 1.7, 2.2, 3.0]
FINE_F = [0.94, 0.96, 0.975, 0.985, 0.992, 0.996, 1.004, 1.008, 1.015, 1.025, 1.04, 1.06]


@dataclass
class ParamsV6(ParamsV4):
    freeze_calls: int = 11        # calls 1-11 are point-identical to v4
    stagnation_window: int = 3    # consecutive calls without a single-building improvement
    base_sigma: float = 0.18      # log10 step of a block move
    wide_sigma: float = 0.32      # log10 step of a block move while a building is stagnant
    share_model: float = 0.38
    share_coarse: float = 0.22
    share_fine: float = 0.22
    n_combo: int = 6
    pool: int = 1500


class BlockDescentPolicyV6(BlockDescentPolicyV4):
    def __init__(self, params: ParamsV6 | None = None, seed: int = 2026):
        super().__init__(params or ParamsV6(), seed)
        # The frozen phase delegates to a plain v4 instance, so calls 1-11 match it point by point.
        self._frozen = BlockDescentPolicyV4(ParamsV4(shift_bias=True, n_nav=3), seed=seed)
        self._train_top_blocks = None

    # ------------------------------------------------------------------ entry
    def propose(self, st, n: int = 50):
        if len(st.calls) + 1 <= self.p.freeze_calls:
            return self._frozen.propose(st, n)
        return self._propose_in_basin(st, n)

    # ------------------------------------------------------------------ stagnation
    def _stagnant(self, st):
        """Per building: no single-building improvement within the last `stagnation_window` calls."""
        window = self.p.stagnation_window
        hit = [False] * 3
        seen = 0
        for call in reversed(st.calls):
            if seen >= window:
                break
            base = next((e["R"] for e in call if e["meta"].get("kind") == "base"), None)
            if base is None:
                continue
            for e in call:
                m = e["meta"]
                if m.get("kind") == "single" and np.isfinite(e["R"]) and e["R"] < base - 1e-9:
                    hit[m["building"]] = True
            seen += 1
        return [not h for h in hit]

    # ------------------------------------------------------------------ low-R training blocks (2048 rows only)
    def _top_train_blocks(self, st):
        if self._train_top_blocks is not None:
            return self._train_top_blocks
        W = norm_blocks(st.X_train)
        order = np.argsort(st.y_train)[:10]
        self._train_top_blocks = [W[order][:, b, :].copy() for b in range(3)]
        return self._train_top_blocks

    # ------------------------------------------------------------------ new moves
    def _move_block(self, rng, base_blk, sigma):
        """Move 2-3 coordinates at once and optionally zero one minor objective."""
        w = base_blk.copy()
        pos = np.flatnonzero(w > 0)
        if len(pos) == 0:
            return None
        nch = 2 if rng.random() < 0.7 else 3
        ks = rng.choice(pos, size=min(nch, len(pos)), replace=False)
        factors = np.clip(10 ** rng.normal(0, sigma, len(ks)), 0.2, 4.0)
        w[ks] *= factors
        if rng.random() < 0.30 and len(pos) > 2:
            minor = [k for k in pos if k in MINOR]
            if minor:
                w[rng.choice(minor)] = 0.0
        if w.sum() <= 0 or not np.all(np.isfinite(w)):
            return None
        w = w / w.sum()
        w[w < ZERO] = 0.0
        return w / w.sum() if w.sum() > 0 else None

    def _blend_block(self, rng, base_blk, seeds):
        """Convex blend with a training/elite block: restart the whole block."""
        if not seeds:
            return None
        a = float(rng.choice([0.3, 0.5, 0.7]))
        src = seeds[rng.integers(len(seeds))]
        w = (1 - a) * base_blk + a * src
        w[w < ZERO] = 0.0
        if w.sum() <= 0:
            return None
        w = w / w.sum()
        return w

    # ------------------------------------------------------------------ in-basin proposal
    def _propose_in_basin(self, st, n):
        p = self.p
        call_no = len(st.calls) + 1
        rng = np.random.default_rng([self.seed, call_no, 6])
        priors = TrainPriors.from_training(st.X_train, st.y_train if p.shift_bias else None)
        inc, R_inc, val, blocks, scale = self._replay(st)
        null, _, improvers = self._learn(st)
        stagnant = self._stagnant(st)
        train_top = self._top_train_blocks(st)

        Xo = [e["W"].reshape(21) for c in st.calls for e in c if np.isfinite(e["R"])]
        yo = [e["R"] for c in st.calls for e in c if np.isfinite(e["R"])]
        X = np.vstack([st.X_train] + ([np.array(Xo)] if Xo else []))
        y = np.concatenate([st.y_train, np.array(yo)]) if yo else st.y_train.copy()
        model = Surrogate(local_rows=p.local_rows, seed=self.seed).fit(X, y, center=inc.reshape(21))

        out, seen = [], set()
        sent = {tuple(bkey(e["W"][b]) for b in range(3)) for c in st.calls for e in c}
        seen |= sent
        seen.add(tuple(bkey(inc[b]) for b in range(3)))
        Wt = norm_blocks(st.X_train)
        seen |= {tuple(bkey(w[b]) for b in range(3)) for w in Wt}

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
            return (nl["lo"] <= v <= nl["hi"]) if kind == "f" else (v <= nl["tmax"])

        def elites(b):
            items = sorted(val[b].items(), key=lambda kv: kv[1])[: p.n_elite]
            return [blocks[b][k] for k, _ in items]

        # ---- base and cross-building combinations (as in v4)
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

        # ---- surrogate pool with coarse and block moves
        pools, sigma_use = {}, {}
        for b in range(3):
            sigma_use[b] = p.wide_sigma if stagnant[b] else p.base_sigma
            parents = [base[b]] + [e for e in elites(b) if bkey(e) != bkey(base[b])]
            cand = []
            kinds = ["scale1", "scale1", "multi", "shift", "block", "block", "tiny"]
            for _ in range(p.pool):
                par = parents[0] if rng.random() < 0.7 or len(parents) == 1 else parents[rng.integers(1, len(parents))]
                kind = kinds[rng.integers(len(kinds))]
                if kind == "block":
                    blk = self._move_block(rng, par, sigma_use[b])
                else:
                    blk = self._move(rng, par, max(scale[b], 0.35), kind, priors, b=b)
                if blk is not None:
                    cand.append(blk)
            if not cand:
                continue
            cand = np.array(cand)
            Wfull = np.repeat(base[None], len(cand), 0)
            Wfull[:, b] = cand
            mu, sd = model.predict(Wfull.reshape(-1, 21))
            pools[b] = (cand, mu, sd)

        # ---- slot split: model / coarse / fine / block
        slots = n - len(out)
        per = [slots // 3 + (1 if b < slots % 3 else 0) for b in range(3)]
        for b in range(3):
            if b not in pools:
                continue
            k_model = max(2, int(round(per[b] * p.share_model)))
            k_coarse = max(2, int(round(per[b] * p.share_coarse)))
            k_fine = max(2, int(round(per[b] * p.share_fine)))
            k_block = max(1, per[b] - k_model - k_coarse - k_fine)
            self._fill_model(b, base, pools[b], k_model, add)
            self._fill_coarse(b, base, k_coarse, rng, call_no, sigma_use[b], in_null, add)
            self._fill_fine(b, base, k_fine, null, priors, rng, call_no, in_null, add)
            self._fill_blocks(b, base, k_block, rng, call_no, sigma_use[b],
                              train_top[b], elites(b), add)

        # ---- top up
        tries = 0
        while len(out) < n and tries < 3000:
            tries += 1
            b = int(rng.integers(3))
            blk = self._move_block(rng, base[b], p.base_sigma)
            if blk is None or in_null(b, base[b], blk):
                continue
            W = base.copy()
            W[b] = blk
            add(W, {"kind": "single", "building": b, "move": "fill"})

        info = {"call": call_no, "R_inc": R_inc, "scale": [round(s, 4) for s in scale],
                "model_secs": model.fit_secs, "stagnant": stagnant,
                "sigma": [round(sigma_use[b], 3) for b in range(3)],
                "phase": "in_basin_multiscale"}
        return out[:n], info

    # ------------------------------------------------------------------ three fill channels
    def _fill_coarse(self, b, base, k, rng, call_no, sigma, in_null, add):
        """Single-coordinate coarse jumps that deliberately cross known null intervals."""
        blk0 = base[b]
        got, depth = 0, 0
        order = [(call_no * 3 + b + j) % 7 for j in range(7)]
        while got < k and depth < 4:
            for kk in order:
                if got >= k:
                    break
                if blk0[kk] <= 0:
                    continue
                idx = (depth * 3 + kk) % len(COARSE_F)
                w = blk0.copy()
                w[kk] *= COARSE_F[idx]
                w = w / w.sum()
                if in_null(b, blk0, w):
                    continue
                W = base.copy()
                W[b] = w
                got += add(W, {"kind": "single", "building": b, "move": "coarse"})
            depth += 1

    def _fill_fine(self, b, base, k, null, priors, rng, call_no, in_null, add):
        """0.4%-6% coordinate moves plus tiny-weight revival, outside known null intervals."""
        blk0 = base[b]
        got, depth = 0, 0
        order = [(call_no + b + j) % 7 for j in range(7)]
        tiny_grid = np.geomspace(priors.tiny_lo, max(priors.tiny_hi, 3 * priors.tiny_lo), 14)
        while got < k and depth < 10:
            for kk in order:
                if got >= k:
                    break
                nl = null.get((b, bkey(blk0), kk), {"lo": 1.0, "hi": 1.0, "tmax": 0.0})
                w = blk0.copy()
                if w[kk] > 0:
                    ups = [f for f in FINE_F if f > nl["hi"]]
                    downs = [f for f in sorted(FINE_F, reverse=True) if f < nl["lo"]]
                    steps = [v for pair in itertools.zip_longest(ups, downs) for v in pair if v is not None]
                    if depth >= len(steps):
                        continue
                    w[kk] *= steps[depth]
                else:
                    ts = [t for t in tiny_grid if t > nl["tmax"]]
                    if depth >= len(ts):
                        continue
                    w[kk] = ts[depth] * (w[4] + w[5])
                if w.sum() <= 0:
                    continue
                w = w / w.sum()
                w[w < ZERO] = 0.0
                if w.sum() <= 0 or in_null(b, blk0, w):
                    continue
                W = base.copy()
                W[b] = w / w.sum()
                got += add(W, {"kind": "single", "building": b, "move": "fine"})
            depth += 1

    def _fill_blocks(self, b, base, k, rng, call_no, sigma, train_blocks, elite_blocks, add):
        """2-3 coordinate scaling; the last slot blends with a training/elite block (restart)."""
        got, tries = 0, 0
        seeds = list(elite_blocks) + [train_blocks[i] for i in range(len(train_blocks))]
        while got < k and tries < 200:
            tries += 1
            if got == k - 1 and seeds:
                w = self._blend_block(rng, base[b], seeds)
                move = "restart"
            else:
                w = self._move_block(rng, base[b], sigma)
                move = "block"
            if w is None:
                continue
            W = base.copy()
            W[b] = w
            got += add(W, {"kind": "single", "building": b, "move": move})


# --------------------------------------------------------------------------------------------------
# v7: keep v4's full batch from call 12 on, replace only the zero-yield slots
# --------------------------------------------------------------------------------------------------
REPLACE_ORDER = ("fine", "zero")
COARSE_STEPS = [0.45, 0.6, 0.75, 1.35, 1.7, 2.2, 3.0]


@dataclass
class ParamsV7(ParamsV6):
    freeze_calls: int = 11
    base_sigma: float = 0.22
    wide_sigma: float = 0.35
    max_replace: int = 12
    replace_stream: int = 7


class BlockDescentPolicyV7(BlockDescentPolicyV6):
    def __init__(self, params: ParamsV7 | None = None, seed: int = 2026):
        super().__init__(params or ParamsV7(), seed)
        # The inner v4 that runs for calls 1-freeze_calls is built from this very same Params, filtered down to
        # the fields ParamsV4 declares (so config.json's V4 constants are honoured and nothing is hard-coded).
        v4_kwargs = {k: v for k, v in asdict(self.p).items()
                     if k in ParamsV4.__dataclass_fields__}
        self._v4 = BlockDescentPolicyV4(ParamsV4(**v4_kwargs), seed=seed)

    # ------------------------------------------------------------------ entry
    def propose(self, st, n: int = 50):
        if len(st.calls) + 1 <= self.p.freeze_calls:
            return self._v4.propose(st, n)
        return self._propose_hybrid(st, n)

    # ------------------------------------------------------------------ multiscale candidates
    def _coarse_block(self, rng, base_blk, k):
        w = base_blk.copy()
        w[k] *= COARSE_STEPS[rng.integers(len(COARSE_STEPS))]
        if w.sum() <= 0:
            return None
        w = w / w.sum()
        w[w < ZERO] = 0.0
        return w / w.sum() if w.sum() > 0 else None

    def _restart_block(self, rng, base_blk, seeds):
        return self._blend_block(rng, base_blk, seeds)

    # ------------------------------------------------------------------ main path
    def _propose_hybrid(self, st, n):
        p = self.p
        call_no = len(st.calls) + 1
        rng = np.random.default_rng([self.seed, call_no, p.replace_stream])
        inc, _, val, blocks, scale = self._replay(st)
        stagnant = self._stagnant(st)
        train_top = self._top_train_blocks(st)

        # 1) v4's complete batch: every model/scan/random/icombo pick is kept
        props, info = self._v4.propose(st, n)
        props = [(np.asarray(W, float), dict(m)) for W, m in props]
        metas = [m for _, m in props]

        # 2) replaceable slots: fine -> zero -> combo (base and every other channel stay)
        replaceable = []
        for move in REPLACE_ORDER:
            replaceable += [i for i, m in enumerate(metas)
                            if m.get("kind") == "single" and m.get("move") == move]
        replaceable += [i for i, m in enumerate(metas) if m.get("kind") == "combo"]
        replaceable = replaceable[: p.max_replace]
        if not replaceable:
            info.update({"phase": "v4_batch", "replaced": 0})
            return props[:n], info

        # 3) multiscale candidates, rotating over the three buildings
        W = np.array([norm_blocks(W)[0] for W, _ in props])
        seen = {tuple(bkey(w[b]) for b in range(3)) for w in W}
        seen |= {tuple(bkey(w[b]) for b in range(3)) for w in norm_blocks(st.X_train)}
        for c in st.calls:
            for e in c:
                seen.add(tuple(bkey(e["W"][b]) for b in range(3)))

        def seeds_for(b):
            items = sorted(val[b].items(), key=lambda kv: kv[1])[: p.n_elite]
            return [blocks[b][k] for k, _ in items] + [train_top[b][i] for i in range(len(train_top[b]))]

        plans = [(0, "coarse1"), (1, "coarse1"), (2, "coarse1"),
                 (0, "block"), (1, "block"), (2, "block"),
                 (0, "restart"), (1, "restart"), (2, "restart"),
                 (0, "coarse2"), (1, "coarse2"), (2, "coarse2")]
        counts = {"coarse": 0, "block": 0, "restart": 0}
        used, new_rows = set(), []
        for b, kind in plans:
            if len(new_rows) >= len(replaceable):
                break
            sigma = p.wide_sigma if stagnant[b] else p.base_sigma
            if kind.startswith("coarse"):
                pos = np.flatnonzero(inc[b] > 0)
                if len(pos) == 0:
                    continue
                k = int(pos[rng.integers(len(pos))]) if kind == "coarse1" else int(
                    pos[np.argmax(inc[b][pos])])
                blk = self._coarse_block(rng, inc[b], k)
                move = "coarse"
            elif kind == "block":
                blk = self._move_block(rng, inc[b], sigma)
                move = "block"
            else:
                blk = self._restart_block(rng, inc[b], seeds_for(b))
                move = "restart"
            if blk is None or np.allclose(blk, inc[b], atol=1e-12):
                continue
            Wnew = inc.copy()
            Wnew[b] = blk
            key = tuple(bkey(Wnew[j]) for j in range(3))
            if key in seen or key in used:
                continue
            used.add(key)
            new_rows.append((Wnew, {"kind": "single", "building": b, "move": move}))
            counts[move] += 1

        # 4) replace by priority; slots that stay empty keep the v4 point
        replaced = 0
        for idx, row in zip(replaceable, new_rows):
            props[idx] = row
            metas[idx] = row[1]
            replaced += 1

        info.update({"phase": "v4_batch_plus_multiscale", "replaced": replaced,
                     "coarse": counts["coarse"], "block": counts["block"],
                     "restart": counts["restart"], "stagnant": stagnant,
                     "sigma": [round(p.wide_sigma if s else p.base_sigma, 3) for s in stagnant]})
        return props[:n], info
