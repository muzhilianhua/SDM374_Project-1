"""FMS: factorised multi-scale search for low R on the course evaluator (research use).

Why this shape (see README.md for the evidence):
  * each building's trajectory depends only on its own 7 weights and R is close to
    additive over buildings in the low region, so the 18-dof problem is searched as
    three 6-dof blocks with the other two blocks held at the incumbent, plus an
    exact recombination step over the best blocks;
  * R is piecewise constant on cones (per-hour argmin over a finite action set), with
    steps at every scale, so moves are multiplicative on single weights at scales from
    0.001 to 1 decade, plus exact-zero toggles (a zero weight is a different tie-break
    regime, not the limit of a small weight);
  * selection is rank based (no gradients, no smooth surrogate), and the operator mix
    adapts to which scale is currently producing improvements.

Run:  python fms.py --minutes 40 [--workers 10] [--per-building 56]
Resumes from fms_state.json; every evaluation is in eval_log.jsonl.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from pool import SolverPool, load_log, normalize

HERE = Path(__file__).resolve().parent
STATE = HERE / "fms_state.json"
PROGRESS = HERE / "fms_progress.jsonl"
TEAM_LOG = HERE.parent / "replay" / "evidence" / "eval_log.jsonl"

SCALES = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]          # decades, single-weight moves
MULTI = [0.002, 0.01, 0.05, 0.2]                             # decades, all positive weights
LATTICE = [40, 100, 400, 1000]                               # integer-ratio weights: exact score ties
OPS = ([f"scale{s:g}" for s in SCALES] + ["toggle"] + [f"multi{s:g}" for s in MULTI]
       + ["cross", "transplant", "global"] + [f"snap{d}" for d in LATTICE])
N_ELITE = 10
N_NICHE, NICHE_FAR, NICHE_STALL = 3, 0.04, 3                 # side funnels per building; L1 distance; patience
LOCAL_OPS = [f"scale{s:g}" for s in SCALES[:6]] + [f"multi{s:g}" for s in MULTI] + ["toggle", "cross"]
MID_OPS = ["scale0.03", "scale0.1", "scale0.3", "scale1", "multi0.05", "multi0.2", "toggle", "cross", "transplant", "global"]


def sensitivity() -> np.ndarray:
    """Per building and weight: how much R moved along that weight's E2 profile (capped at 1).
    Building 1's resilience weight, for example, changes nothing between 0 and 0.3, so two blocks
    that differ only there are the same block for the search."""
    sens = np.ones((3, 7))
    path = HERE / "E2_profiles.json"
    if path.exists():
        prof = json.loads(path.read_text(encoding="utf-8"))
        names = ["cost", "carbon", "peak", "discomfort", "degradation", "ramping", "resilience"]
        for key, pts in prof.items():
            b, name = int(key[1]) - 1, key.split(":")[1]
            if name in names:
                vals = [r for _, r in pts]
                sens[b, names.index(name)] = min(1.0, max(vals) - min(vals))
    return sens


SENS = sensitivity()


def features(w) -> np.ndarray:
    """Per-building features for the trend surrogate: shares, log shares, r = F5/(F5+F6), F5+F6."""
    w = np.asarray(w, float).reshape(-1, 7)
    s = w[:, 4] + w[:, 5]
    return np.hstack([w, np.log10(w + 1e-4), (w[:, 4] / (s + 1e-12))[:, None], s[:, None]])


def bkey(w) -> str:
    return ",".join(repr(float(v)) for v in w)


def norm_block(w) -> np.ndarray:
    w = np.clip(np.asarray(w, float), 0.0, None)
    return w / w.sum()


class Search:
    def __init__(self, pool: SolverPool, seed: int = 20261112, run: str = "fms"):
        self.pool = pool
        self.run = run
        self.rng = np.random.default_rng(seed)
        self.gen = 0
        self.ctx = 0                                   # increases whenever the incumbent changes
        self.arch = [dict(), dict(), dict()]           # per building: key -> {w, delta, ctx, op}
        self.op_reward = {op: [4.0, 0.4] for op in OPS}  # discounted [trials, credit] per operator
        self.inc = None
        self.inc_R = None
        self.niches = [[], [], []]                     # per building: side funnels being descended
        self.retired = [[], [], []]                    # centres of side funnels that stalled
        self.next_niche = 1

    # ------------------------------------------------------------ persistence
    def save(self):
        STATE.write_text(json.dumps({
            "gen": self.gen, "ctx": self.ctx, "inc": self.inc.tolist(), "inc_R": self.inc_R,
            "op_reward": self.op_reward, "rng": self.rng.bit_generator.state,
            "niches": [[{**z, "center": z["center"].tolist()} for z in zs] for zs in self.niches],
            "retired": [[c.tolist() for c in cs] for cs in self.retired], "next_niche": self.next_niche,
            "arch": [[{"w": e["w"].tolist(), "delta": e["delta"], "ctx": e["ctx"], "op": e["op"], "niche": e.get("niche", 0)}
                      for e in a.values()] for a in self.arch],
        }), encoding="utf-8")

    def load(self):
        s = json.loads(STATE.read_text(encoding="utf-8"))
        self.gen, self.ctx, self.inc_R = s["gen"], s["ctx"], s["inc_R"]
        self.inc = np.array(s["inc"]); self.op_reward.update(s["op_reward"])
        self.rng.bit_generator.state = s["rng"]
        for b in range(3):
            for e in s["arch"][b]:
                w = np.array(e["w"])
                self.arch[b][bkey(w)] = {"w": w, "delta": e["delta"], "ctx": e["ctx"], "op": e["op"], "niche": e.get("niche", 0)}
        self.niches = [[{**z, "center": np.array(z["center"])} for z in zs] for zs in s.get("niches", [[], [], []])]
        self.retired = [[np.array(c) for c in cs] for cs in s.get("retired", [[], [], []])]
        self.next_niche = s.get("next_niche", 1)

    # ------------------------------------------------------------------ init
    def init_from_logs(self, window: float = 8.0):
        """Seed the archives with main effects from an additive least-squares fit of every
        logged point (team log + this harness) within `window` of the best R."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import lsqr
        Ws, Rs = [], []
        if TEAM_LOG.exists():
            for line in TEAM_LOG.open(encoding="utf-8"):
                rec = json.loads(line); Ws.append(rec["weights"]); Rs.append(rec["R"])
        W2, R2, _ = load_log()
        W = normalize(np.vstack([np.array(Ws).reshape(-1, 21), W2])).reshape(-1, 3, 7)
        R = np.concatenate([np.array(Rs), R2])
        i0 = int(np.argmin(R))
        self.inc, self.inc_R = W[i0].copy(), float(R[i0])
        m = R < self.inc_R + window
        W, R = W[m], R[m]
        idx, blocks = [dict(), dict(), dict()], [[], [], []]
        rows, cols = [], []
        off = 0
        col = np.zeros((len(R), 3), int)
        for b in range(3):
            for i in range(len(R)):
                k = bkey(W[i, b])
                if k not in idx[b]:
                    idx[b][k] = len(blocks[b]); blocks[b].append(W[i, b])
                col[i, b] = off + idx[b][k]
            off += len(blocks[b])
        n = len(R)
        A = coo_matrix((np.ones(3 * n), (np.repeat(np.arange(n), 3), col.ravel())), shape=(n, off)).tocsr()
        sol = lsqr(A, R - R.mean(), damp=0.05)[0]
        off = 0
        for b in range(3):
            eff = sol[off:off + len(blocks[b])]; off += len(blocks[b])
            ref = eff[idx[b][bkey(self.inc[b])]]
            for w, e in zip(blocks[b], eff):
                self.arch[b][bkey(w)] = {"w": np.array(w), "delta": float(e - ref), "ctx": -1, "op": "seed"}
            self.arch[b][bkey(self.inc[b])].update(delta=0.0, ctx=0)
        print(f"[init] incumbent R={self.inc_R:.6f}; seeded blocks per building: {[len(a) for a in self.arch]}", flush=True)

    # --------------------------------------------------------------- elites
    def elites(self, b: int, n: int = N_ELITE, niche: int = 0):
        """Best blocks of building b by estimated delta: one per plateau (equal delta means the
        same trajectory) and at most half of them with the same support pattern. Mutants of a
        needle are near-needles (gen 1: median delta +0.12 against +1.7..2.8 for random blocks
        of the basin), so several near-optimal parents reach more distinct cells than one."""
        ordered = sorted((e for e in self.arch[b].values() if e.get("niche", 0) == niche), key=lambda e: e["delta"])
        out, deltas, pats = [], [], {}
        for e in ordered:
            if any(abs(e["delta"] - d) < 1e-9 for d in deltas):
                continue
            pat = tuple((e["w"] > 0).tolist())
            if pats.get(pat, 0) >= max(1, n // 2):
                continue
            pats[pat] = pats.get(pat, 0) + 1
            out.append(e); deltas.append(e["delta"])
            if len(out) == n:
                break
        return out

    # ------------------------------------------------------------- side funnels
    def refresh_niches(self, b: int):
        """Funnel-aware search. A block far from the incumbent is judged by where a local descent
        from it ends, not by its first sample: mutants of a good block are much closer to it in R
        (median +0.12) than random blocks of the basin are (+1.7..2.8), so every good block sits in
        a funnel whose bottom is unknown until it is descended. Each building keeps N_NICHE side
        funnels; one that fails to improve for NICHE_STALL generations is retired and replaced by
        the best block not within NICHE_FAR (L1) of any current or retired centre."""
        for z in list(self.niches[b]):
            if z["stall"] >= NICHE_STALL:
                self.retired[b].append(z["center"]); self.niches[b].remove(z)
        while len(self.niches[b]) < N_NICHE:
            centers = np.array([self.inc[b]] + [z["center"] for z in self.niches[b]] + self.retired[b])
            start = None
            for e in sorted(self.arch[b].values(), key=lambda e: e["delta"]):
                if e["delta"] < 1e-9 or e.get("niche", 0) != 0:      # the incumbent's own cell, or taken
                    continue
                if (np.abs(centers - e["w"]) * SENS[b]).sum(1).min() > NICHE_FAR:
                    start = e
                    break
            if start is None:
                break
            start["niche"] = self.next_niche
            self.niches[b].append({"id": self.next_niche, "center": start["w"].copy(), "stall": 0,
                                   "best_abs": float("inf"), "start_delta": start["delta"], "born": self.gen})
            self.next_niche += 1

    # ------------------------------------------------------------ operators
    def mutate(self, b: int, op: str, parents=None) -> np.ndarray | None:
        rng = self.rng
        el = self._el[b]
        if parents is None:
            p = np.full(len(el), 0.6 / max(len(el) - 1, 1)); p[0] = 0.4; p /= p.sum()   # 40 % incumbent block
            w = el[rng.choice(len(el), p=p)]["w"].copy()
        else:                                         # wider parent set for the surrogate-screened pool
            w = parents[rng.integers(0, len(parents))]["w"].copy()
        pos = np.where(w > 0)[0]
        if op.startswith("scale"):
            k = rng.choice(pos)
            w[k] *= 10 ** rng.normal(0, float(op[5:]))
        elif op.startswith("multi"):
            w[pos] *= 10 ** rng.normal(0, float(op[5:]), size=len(pos))
        elif op == "toggle":
            zero = np.where(w == 0)[0]
            if len(zero) and (len(pos) <= 2 or rng.random() < 0.6):
                w[rng.choice(zero)] = w.max() * 10 ** rng.uniform(-3.7, -0.5)   # below ~1e-4 a weight acts as exact 0 (E2)
            elif len(pos) > 2:
                w[rng.choice(pos)] = 0.0
            else:
                return None
        elif op.startswith("snap"):
            # Weights in small-integer ratios make candidate scores tie exactly, so the controller's
            # tie-break (least battery use first) decides: a regime generic real-valued weights never reach.
            w[pos] *= 10 ** rng.normal(0, 0.02, size=len(pos))
            w = np.round(w / w.sum() * int(op[4:]))
        elif op == "cross":
            other = el[rng.choice(len(el))]["w"]
            t = rng.uniform(-0.25, 1.25)
            w = np.clip(t * w + (1 - t) * other, 0, None)
        elif op == "transplant":                     # another building's elite block as the second parent
            b2 = rng.choice([x for x in range(3) if x != b])
            other = self._el[b2][rng.integers(0, min(3, len(self._el[b2])))]["w"]
            mask = rng.random(7) < 0.5
            w = np.where(mask, other, w)
        elif op == "global":                          # fresh sparse block anywhere on the simplex
            s = rng.integers(1, 6)
            sup = rng.choice(7, size=s, replace=False)
            w = np.zeros(7); w[sup] = 10 ** rng.uniform(-3, 0, size=s)
        if w.sum() <= 0 or not np.all(np.isfinite(w)):
            return None
        return norm_block(w)

    def op_probs(self):
        r = np.array([(self.op_reward[o][1] + 0.3) / (self.op_reward[o][0] + 6.0) for o in OPS])
        return 0.5 * r / r.sum() + 0.5 / len(OPS)      # half adaptive, half uniform: success rates are noisy

    def screened(self, b: int, n: int, n_pool: int = 3000, kappa: float = 1.0):
        """Mid- and large-scale moves pre-screened by a tree-ensemble model of this building's
        trend (lower confidence bound). The fine-scale roughness is not predictable, the trend is."""
        from sklearn.ensemble import ExtraTreesRegressor
        ent = list(self.arch[b].values())
        model = ExtraTreesRegressor(n_estimators=150, min_samples_leaf=2, max_features=0.5, random_state=self.gen, n_jobs=8)
        model.fit(features([e["w"] for e in ent]), np.clip([e["delta"] for e in ent], -2.0, 6.0))
        parents = sorted(ent, key=lambda e: e["delta"])[:30]
        cand, keys = [], set()
        tries = 0
        while len(cand) < n_pool and tries < n_pool * 5:
            tries += 1
            w = self.mutate(b, MID_OPS[self.rng.integers(0, len(MID_OPS))], parents)
            if w is None:
                continue
            k = bkey(w)
            if k in self.arch[b] or k in keys:
                continue
            keys.add(k); cand.append(w)
        F = features(cand)
        pred = np.stack([t.predict(F) for t in model.estimators_])
        score = pred.mean(0) - kappa * pred.std(0)
        out = []
        for i in np.argsort(score):
            if all(np.abs(cand[i] - o).sum() > 0.004 for o in out):   # spread the picks
                out.append(cand[i])
                if len(out) == n:
                    break
        return out

    # ---------------------------------------------------------- one generation
    def point(self, b: int, w: np.ndarray) -> np.ndarray:
        W = self.inc.copy(); W[b] = w
        return W.ravel()

    def adopt(self, W_new: np.ndarray, R_new: float, why: str):
        for b in range(3):
            k = bkey(W_new[b])
            shift = self.arch[b][k]["delta"] if k in self.arch[b] else 0.0
            if k not in self.arch[b]:
                self.arch[b][k] = {"w": W_new[b].copy(), "delta": 0.0, "ctx": 0, "op": why}
            nid = self.arch[b][k].get("niche", 0)
            if nid:                                   # a side funnel won: it becomes the main one
                print(f"  [funnel] building {b + 1}: side funnel {nid} took over", flush=True)
                for e in self.arch[b].values():
                    if e.get("niche", 0) == nid:
                        e["niche"] = 0
                self.niches[b] = [z for z in self.niches[b] if z["id"] != nid]
            for e in self.arch[b].values():
                e["delta"] -= shift
        print(f"  [adopt] {self.inc_R:.6f} -> {R_new:.6f} ({why})", flush=True)
        self.inc, self.inc_R = W_new.copy(), float(R_new)
        self.ctx += 1
        for b in range(3):
            self.arch[b][bkey(self.inc[b])].update(delta=0.0, ctx=self.ctx)

    def combos(self, sets, tag: str):
        """Evaluate the cross product of per-building block lists; adopt the best if it improves."""
        X = np.array([np.concatenate(c) for c in itertools.product(*sets)])
        y = self.pool.evaluate(X, tag=tag)
        i = int(np.nanargmin(y))
        if y[i] < self.inc_R - 1e-12:
            self.adopt(X[i].reshape(3, 7), float(y[i]), tag)
            return True
        return False

    def generation(self, per_building: int, n_sur: int = 0, n_niche: int = 0):
        t0 = time.time(); n0 = self.pool.n_solver_evals; R_start = self.inc_R
        probs = self.op_probs()
        X, meta, seen = [], [], set()
        self._el = [self.elites(b) for b in range(3)]
        for b in range(3):
            # re-measure stale elites in the current context (exact deltas for the parents)
            for e in self._el[b][:3]:
                if e["ctx"] != self.ctx:
                    X.append(self.point(b, e["w"])); meta.append((b, bkey(e["w"]), "recheck", 0))
            made = 0
            if n_niche:                                # side funnels: local moves around their own best blocks
                self.refresh_niches(b)
                local_p = np.array([probs[OPS.index(o)] for o in LOCAL_OPS]); local_p /= local_p.sum()
                for z in self.niches[b]:
                    parents = self.elites(b, 4, niche=z["id"])
                    for e in parents[:1]:
                        if e["ctx"] != self.ctx:
                            X.append(self.point(b, e["w"])); meta.append((b, bkey(e["w"]), "recheck", z["id"]))
                    got = tries = 0
                    while got < n_niche and tries < n_niche * 20:
                        tries += 1
                        w = self.mutate(b, LOCAL_OPS[self.rng.choice(len(LOCAL_OPS), p=local_p)], parents)
                        if w is None:
                            continue
                        k = bkey(w)
                        if k in self.arch[b] or (b, k) in seen:
                            continue
                        seen.add((b, k)); X.append(self.point(b, w)); meta.append((b, k, "funnel", z["id"])); got += 1; made += 1
            for w in (self.screened(b, n_sur) if n_sur else []):
                k = bkey(w)
                if (b, k) in seen:
                    continue
                seen.add((b, k)); X.append(self.point(b, w)); meta.append((b, k, "sur", 0)); made += 1
            tries = 0
            while made < per_building and tries < per_building * 20:
                tries += 1
                op = OPS[self.rng.choice(len(OPS), p=probs)]
                w = self.mutate(b, op)
                if w is None:
                    continue
                k = bkey(w)
                if k in self.arch[b] or (b, k) in seen:
                    continue
                seen.add((b, k)); X.append(self.point(b, w)); meta.append((b, k, op, 0)); made += 1
        y = self.pool.evaluate(np.array(X), tag=f"{self.run}_g{self.gen}")
        improved = [[], [], []]
        stats = {}
        for (b, k, op, nid), x, r in zip(meta, X, y):
            if r != r:
                continue
            w = x.reshape(3, 7)[b]
            d = float(r - self.inc_R)
            self.arch[b][k] = {"w": w.copy(), "delta": d, "ctx": self.ctx, "niche": nid,
                               "op": op if op != "recheck" else self.arch[b].get(k, {}).get("op", "seed")}
            if d < -1e-12:
                improved[b].append((d, w.copy()))
            if op != "recheck":
                s = stats.setdefault(op, [0, 0, 0.0, 0.0]); s[0] += 1; s[1] += d < -1e-12; s[2] = min(s[2], d)
                s[3] += 1.0 if d < -1e-12 else (0.2 if d < 0.15 else 0.0)   # near-ties keep useful parents coming
        for op in OPS:                                             # discounted success credit per operator
            n, _, _, c = stats.get(op, [0, 0, 0.0, 0.0])
            self.op_reward[op] = [0.85 * self.op_reward[op][0] + n, 0.85 * self.op_reward[op][1] + c]
        n_imp = [len(v) for v in improved]
        funnels = []
        for b in range(3):                                         # did each side funnel get deeper?
            for z in self.niches[b]:
                best = min((e for e in self.arch[b].values() if e.get("niche", 0) == z["id"]), key=lambda e: e["delta"])
                best_abs = R_start + best["delta"]
                if best_abs < z["best_abs"] - 1e-9:
                    z["best_abs"], z["stall"], z["center"] = best_abs, 0, best["w"].copy()
                else:
                    z["stall"] += 1
                if best["delta"] > 1.2 or abs(best["delta"]) < 1e-9:   # optimistic entrance, or the incumbent's own cell
                    z["stall"] = NICHE_STALL
                funnels.append([b + 1, z["id"], round(z["start_delta"], 3), round(best["delta"], 4), z["stall"]])
        if sum(n_imp):
            sets = []
            for b in range(3):
                top = [w for _, w in sorted(improved[b], key=lambda t: t[0])[:1]]
                sets.append([self.inc[b].copy()] + top)
            self.combos(sets, f"{self.run}_g{self.gen}_adopt")
        # No elite recombination step: E4 (exp_recomb.py) measured the 7x7x7 cross product of
        # near-elite blocks; single-block deltas add up to within 0.03 and no combination beat
        # the incumbent, so the best point is simply the best block of each building.
        rec = {"run": self.run, "gen": self.gen, "R_best": self.inc_R, "gain": R_start - self.inc_R, "improving": n_imp,
               "new_evals": self.pool.n_solver_evals - n0, "secs": round(time.time() - t0, 1),
               "ops": {op: [n, int(k), round(dm, 4)] for op, (n, k, dm, _) in sorted(stats.items())},
               "funnels": funnels,
               "best_W": self.inc.tolist()}
        with PROGRESS.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"gen {self.gen:3d}  R*={self.inc_R:.6f}  gain={rec['gain']:.4f}  improving b1/b2/b3={n_imp}  "
              f"evals={rec['new_evals']}  {rec['secs']:.0f}s", flush=True)
        self.gen += 1
        self.save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--per-building", type=int, default=56)
    ap.add_argument("--surrogate", type=int, default=24, help="surrogate-screened candidates per building per generation")
    ap.add_argument("--niche", type=int, default=0, help="candidates per side funnel per generation (3 funnels per building)")
    ap.add_argument("--fresh", action="store_true", help="ignore fms_state.json and re-seed from the logs")
    ap.add_argument("--run-name", default="fms", help="prefix of the evaluation tags and of the progress records")
    a = ap.parse_args()
    pool = SolverPool(n_workers=a.workers)
    s = Search(pool, run=a.run_name)
    if STATE.exists() and not a.fresh:
        s.load(); print(f"[resume] gen {s.gen}, R*={s.inc_R:.6f}", flush=True)
    else:
        s.init_from_logs(); s.save()
    t_end = time.time() + a.minutes * 60
    while time.time() < t_end and not (HERE / "STOP").exists():   # create a file named STOP to end after this generation
        s.generation(a.per_building, a.surrogate, a.niche)
    np.set_printoptions(precision=6, suppress=True, linewidth=200)
    print("FINAL R*=%.9f" % s.inc_R); print(s.inc)
    print("operator selection probabilities:", dict(zip(OPS, np.round(s.op_probs(), 3).tolist())))


if __name__ == "__main__":
    main()
