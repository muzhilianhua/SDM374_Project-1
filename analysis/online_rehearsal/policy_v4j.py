"""v4j: ablation of the block structure of v4 (report experiment, 2026-11-22).

Everything that does not depend on blocks is v4: the navigation surrogate and its lower-confidence-bound picks,
the move kinds and their step sizes, the training-derived shift bias and navigation calls, the trust-scale rule,
the adaptive class shares with the same floors and window, late scans, and the no-repeat rule. What is removed is
exactly what the block structure provides:

* every candidate moves all three buildings at once (each building draws its own move of the call's class) from
  the incumbent or a measured elite, instead of one building from the call's base;
* there are no base or combination points, because without single-building results there are no per-building
  effects to combine;
* the null-step filter and the improving-coordinate re-application are off, since both need a single-building,
  single-coordinate result to attribute; scans step one coordinate in every building at once;
* one joint trust scale and one hit statistic per class, where a hit is a point below the incumbent.

The policy still sees only the training rows and this run's own results.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from policy import State, TrainPriors, ZERO, bkey, norm_blocks
from policy_v4 import CLASS_OF, EQ, SCAN_F, BlockDescentPolicyV4, ParamsV4
from surrogate import Surrogate, features


class JointPolicyV4(BlockDescentPolicyV4):
    def __init__(self, params: ParamsV4 | None = None, seed: int = 2026):
        super().__init__(params or ParamsV4(), seed)

    def _joint_learn(self, st: State):
        """Incumbent, joint trust scale and per-class hit statistics from the run's own results."""
        p = self.p
        i0 = int(np.argmin(st.y_train))
        inc, R_inc = norm_blocks(st.X_train[i0])[0], float(st.y_train[i0])
        scale = p.scale0
        hist = []
        for call in st.calls:
            stats = defaultdict(lambda: [0, 0])
            for e in call:
                if not np.isfinite(e["R"]):
                    continue
                cls = CLASS_OF.get(e["meta"].get("move"), "random")
                stats[cls][0] += e["R"] < R_inc - EQ
                stats[cls][1] += 1
            finite = [e for e in call if np.isfinite(e["R"])]
            best = min(finite, key=lambda e: e["R"]) if finite else None
            if best is not None and best["R"] < R_inc - 1e-12:
                inc, R_inc = best["W"].copy(), float(best["R"])
                scale = min(p.scale_max, scale * p.scale_up)
            else:
                scale = max(p.scale_min, scale * p.scale_down)
            hist.append(stats)
        return inc, R_inc, scale, hist

    def _joint_move(self, rng, parent, scale, kinds, priors):
        """Move every building of `parent` with its own move drawn from `kinds`; None if a building fails."""
        W = parent.copy()
        for b in range(3):
            blk = None
            for _ in range(20):
                blk = self._move(rng, parent[b], scale, kinds[rng.integers(len(kinds))], priors, b=b)
                if blk is not None:
                    break
            if blk is None:
                return None
            W[b] = blk
        return W

    def propose(self, st: State, n: int = 50):
        p = self.p
        call_no = len(st.calls) + 1
        rng = np.random.default_rng([self.seed, call_no, 5])
        priors = TrainPriors.from_training(st.X_train, st.y_train if p.shift_bias else None)
        inc, R_inc, scale, hist = self._joint_learn(st)

        Xo = [e["W"].reshape(21) for c in st.calls for e in c if np.isfinite(e["R"])]
        yo = [e["R"] for c in st.calls for e in c if np.isfinite(e["R"])]
        X = np.vstack([st.X_train] + ([np.array(Xo)] if Xo else []))
        y = np.concatenate([st.y_train, np.array(yo)]) if yo else st.y_train.copy()
        model = Surrogate(local_rows=p.local_rows, seed=self.seed).fit(X, y, center=inc.reshape(21))

        out, seen = [], set()
        seen |= {tuple(bkey(e["W"][b]) for b in range(3)) for c in st.calls for e in c}
        seen.add(tuple(bkey(inc[b]) for b in range(3)))
        if getattr(self, "_train_keys", None) is None:
            self._train_keys = {tuple(bkey(w[b]) for b in range(3)) for w in norm_blocks(st.X_train)}
        seen |= self._train_keys

        def add(W, meta):
            W = norm_blocks(W)[0]
            k = tuple(bkey(W[b]) for b in range(3))
            if k in seen:
                return False
            seen.add(k)
            out.append((W, dict(meta, kind="joint")))
            return True

        # parents: the incumbent, or one of the best measured points (the joint analogue of block elites)
        measured = sorted(((e["R"], e["W"]) for c in st.calls for e in c if np.isfinite(e["R"])),
                          key=lambda t: t[0])
        elites = [W for _, W in measured[: p.n_elite] if tuple(bkey(W[b]) for b in range(3))
                  != tuple(bkey(inc[b]) for b in range(3))]

        def parent():
            return inc if rng.random() < 0.6 or not elites else elites[rng.integers(len(elites))]

        # ---- surrogate pool of joint moves
        cand = []
        for _ in range(p.pool):
            W = self._joint_move(rng, parent(), scale, ["scale1", "scale1", "multi", "shift", "zero", "tiny"],
                                 priors)
            if W is not None:
                cand.append(W)
        cand = np.array(cand)
        mu, sd = model.predict(cand.reshape(-1, 21))

        # ---- v2 navigation moves (same number of points as v4: n_nav per building)
        if p.n_nav and call_no <= p.nav_calls:
            got, tries = 0, 0
            while got < 3 * p.n_nav and tries < 300:
                tries += 1
                W = self._joint_move(rng, inc, max(scale, 0.5), ["shift"], priors)
                if W is not None:
                    got += add(W, {"move": "nav"})

        # ---- slots by adaptive class shares, over the whole batch
        shares = self._shares(hist, call_no)
        slots = n - len(out)
        quota = {c: int(np.floor(slots * shares[c])) for c in shares}
        for c in sorted(shares, key=lambda c: -shares[c])[: slots - sum(quota.values())]:
            quota[c] += 1

        # model picks: lowest LCB with a minimum feature distance, as in v4
        lcb = mu - p.kappa * sd
        F = features(cand.reshape(-1, 21))
        taken = []
        for i in np.argsort(lcb):
            if len(taken) >= quota.get("model", 0):
                break
            if taken and np.min(np.abs(F[taken] - F[i]).max(axis=1)) < 0.02:
                continue
            if add(cand[i], {"move": "model", "pred": float(mu[i])}):
                taken.append(i)

        if "scan" in quota:
            self._joint_scan(inc, quota["scan"], priors, add, call_no)
        for k, kinds, label in ((quota.get("random", 0), ["scale1", "multi", "shift"], "random"),
                                (quota.get("fine", 0), ["fine", "fine", "zero", "tiny"], None)):
            got, tries = 0, 0
            while got < k and tries < 300:
                tries += 1
                kind = kinds[rng.integers(len(kinds))]
                W = self._joint_move(rng, inc, scale, [kind], priors)
                if W is not None:
                    got += add(W, {"move": label or kind})
        tries = 0
        while len(out) < n and tries < 2000:
            tries += 1
            W = self._joint_move(rng, inc, scale, ["scale1"], priors)
            if W is not None:
                add(W, {"move": "fill"})
        info = {"call": call_no, "R_inc": R_inc, "scale": [round(scale, 4)] * 3,
                "model_secs": model.fit_secs, "shares": {c: round(v, 3) for c, v in shares.items()}}
        return out[:n], info

    def _joint_scan(self, inc, k, priors, add, call_no):
        """One-coordinate steps in every building at once: the j-th point steps coordinate (call*3 + b + j) % 7
        of building b by the next factor outward from 1 (a zero coordinate gets the next tiny share)."""
        steps = [f for f in sorted(SCAN_F, key=lambda f: abs(np.log(f)) if f > 0 else np.inf)]
        tiny = np.geomspace(priors.tiny_lo, max(priors.tiny_hi, 3 * priors.tiny_lo), 8)
        got, j = 0, 0
        while got < k and j < 7 * len(steps):
            W = inc.copy()
            for b in range(3):
                kk = (call_no * 3 + b + j) % 7
                depth = j // 7
                w = W[b].copy()
                if w[kk] > 0:
                    w[kk] *= steps[depth % len(steps)]
                else:
                    w[kk] = tiny[depth % len(tiny)] * (w[4] + w[5])
                if w.sum() <= 0:
                    continue
                w = w / w.sum()
                w[w < ZERO] = 0.0
                if w.sum() > 0:
                    W[b] = w / w.sum()
            got += add(W, {"move": "scan"})
            j += 1
