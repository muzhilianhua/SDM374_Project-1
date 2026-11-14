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

from dataclasses import dataclass, field
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
    tiny_lo: float     # smallest non-negligible weight magnitude seen in the training rows
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
            meas = {}
            for e in call:
                if np.isfinite(e["R"]):
                    meas[e["meta"].get("id")] = e
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
                ref = R_base if R_base is not None else np.inf
                if best_single < ref - 1e-9 or improved[b]:
                    scale[b] = min(p.scale_max, scale[b] * p.scale_up)
                else:
                    scale[b] = max(p.scale_min, scale[b] * p.scale_down)
        return inc, R_inc, val, blocks, scale

    # ------------------------------------------------------------------ moves
    def _move(self, rng, parent, scale, kind, priors, grid=None, b=None):
        w = parent.copy()
        pos = np.flatnonzero(w > 0)
        if kind == "scale1":
            k = rng.integers(7)
            if w[k] == 0:
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
