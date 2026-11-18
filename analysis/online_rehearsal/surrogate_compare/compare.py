"""Prequential (call-by-call) offline comparison of online surrogates on the T37 rehearsal logs.

For every rehearsal run and every call k >= 2: fit each candidate surrogate on the 2048 training rows plus
that run's points from calls < k (exactly the data the policy had), centre = best point so far, then
predict the points the run actually measured in call k. No solver calls. Writes one row per
(run, call, point, model) to preds.csv.

Run from the repository root (about 40 minutes with 8 workers; the T37 model dominates):
    uv run python analysis/online_rehearsal/surrogate_compare/compare.py --workers 8

Without --runs it compares the 16 v1-v6 runs it was written for (later runs are not part of this study).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("SDM_REPO", HERE.parents[2])).resolve()   # repository root
REH = REPO / "analysis" / "online_rehearsal"
OM = REPO / "analysis" / "online_models"


def _setup_threads(n: int):
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = str(n)


# ---------------------------------------------------------------------------------------- models
def build_models():
    import numpy as np
    from scipy.linalg import cho_factor, cho_solve
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    sys.path.insert(0, str(REH))
    sys.path.insert(0, str(OM))
    from surrogate import Surrogate, features  # T37
    from modellib import block_ard_cov

    spec = json.loads((OM / "hp_noise001_untrained" / "train_config.json").read_text(encoding="utf-8"))
    LS = np.asarray(spec["kernel"]["length_scale"], float)
    GW = float(spec["kernel"]["global_weight"])
    NOISE = float(spec["kernel"]["noise"])
    JIT = float(spec["kernel"].get("jitter", 1e-8))

    def norm(X):
        W = np.asarray(X, float).reshape(-1, 3, 7)
        return (W / W.sum(axis=2, keepdims=True)).reshape(-1, 21)

    def local_idx(Z, zc, n):
        d = np.abs(Z - zc).max(axis=1) + 1e-3 * np.linalg.norm(Z - zc, axis=1)
        return np.argsort(d)[:n]

    class T37:
        name = "t37"

        def fit(self, X, y, c):
            self.m = Surrogate(local_rows=600, seed=0).fit(X, y, center=c)
            return self

        def predict(self, X):
            return self.m.predict(X)

    class T37Fixed:
        """T37 architecture with a fixed kernel. kernel=None -> azq's submission constants."""

        def __init__(self, name, kernel=None, gpr_alpha=1e-10):
            self.name, self.kernel, self.gpr_alpha = name, kernel, gpr_alpha

        def fit(self, X, y, c):
            F = features(X)
            self.sc = StandardScaler().fit(F)
            Z = self.sc.transform(F)
            self.trend = Ridge(alpha=1.0).fit(Z, y)
            res = y - self.trend.predict(Z)
            idx = local_idx(Z, self.sc.transform(features(np.asarray(c)[None]))[0], 600)
            self.gp = GaussianProcessRegressor(self.kernel, normalize_y=True, optimizer=None,
                                               alpha=self.gpr_alpha)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.gp.fit(Z[idx], res[idx])
            return self

        def predict(self, X):
            Z = self.sc.transform(features(X))
            m, s = self.gp.predict(Z, return_std=True)
            return self.trend.predict(Z) + m, s

    class BlockGP:
        """hp_noise001 kernel with fixed constants on raw normalised weights.

        trend=False: global GP on all rows with constant mean (PR #24 model, refitted with online rows).
        trend=True:  T37 ridge trend on ratio features + this GP on the residuals of the 600 local rows."""

        def __init__(self, name, trend):
            self.name, self.trend_on = name, trend

        def fit(self, X, y, c):
            Xn = norm(X)
            if self.trend_on:
                F = features(X)
                self.sc = StandardScaler().fit(F)
                Z = self.sc.transform(F)
                self.trend = Ridge(alpha=1.0).fit(Z, y)
                t = y - self.trend.predict(Z)
                idx = local_idx(Z, self.sc.transform(features(np.asarray(c)[None]))[0], 600)
                Xn, t = Xn[idx], t[idx]
            else:
                t = y
            self.loc, self.scale = float(t.mean()), float(t.std()) or 1.0
            K = block_ard_cov(Xn, Xn, LS, GW)
            K[np.diag_indices_from(K)] += NOISE + JIT
            self.cf = cho_factor(K)
            self.alpha = cho_solve(self.cf, (t - self.loc) / self.scale)
            self.Xtr = Xn
            return self

        def predict(self, X):
            Xq = norm(X)
            Ks = block_ard_cov(Xq, self.Xtr, LS, GW)
            mz = Ks @ self.alpha
            v = cho_solve(self.cf, Ks.T)
            var = np.maximum(1.0 - np.einsum("ij,ji->i", Ks, v), 0.0) + NOISE
            mean = self.loc + self.scale * mz
            if self.trend_on:
                mean = mean + self.trend.predict(self.sc.transform(features(X)))
            return mean, self.scale * np.sqrt(var)

    return T37, T37Fixed, BlockGP, RBF, ConstantKernel, WhiteKernel


# ---------------------------------------------------------------------------------------- data
def load_train():
    import hashlib
    import pandas as pd
    data = REPO / "dataset_same8"
    man = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    assert hashlib.sha256((data / "train.csv").read_bytes()).hexdigest() == man["train_sha256"]
    t = pd.read_csv(data / "train.csv", float_precision="round_trip", dtype={"candidate_id": str})
    t = t.sort_values("candidate_id", kind="stable").reset_index(drop=True)
    return t.iloc[:, 1:22].to_numpy(float), t["ramping"].to_numpy(float)


def load_run(run):
    import numpy as np
    pts = []
    for line in open(REH / "rehearsals" / run / "online_log.jsonl", encoding="utf-8"):
        d = json.loads(line)
        if d["type"] == "point" and d.get("R") is not None and np.isfinite(d["R"]):
            m = d.get("meta", {})
            pts.append(dict(call=d["call"], W=np.asarray(d["W"], float), R=float(d["R"]),
                            kind=m.get("kind"), move=m.get("move") or m.get("kind")))
    return pts


def frozen_kernel():
    """T37 kernel hyperparameters fitted once on the training rows only (centre = best row), then frozen."""
    import numpy as np
    T37, *_ = build_models()
    X, y = load_train()
    m = T37().fit(X, y, X[int(np.argmin(y))])
    k = m.m.gp.kernel_
    return k, repr(k)


def run_one(args):
    run, frozen_k, calls = args
    _setup_threads(2)
    import numpy as np
    T37, T37Fixed, BlockGP, RBF, ConstantKernel, WhiteKernel = build_models()
    from sklearn.base import clone
    azq_k = (ConstantKernel(1.0, constant_value_bounds="fixed")
             * RBF(np.full(21, 1.0), length_scale_bounds="fixed")
             + WhiteKernel(0.1, noise_level_bounds="fixed"))
    models = [T37(), T37Fixed("azq_fixed", azq_k, gpr_alpha=1.0),
              T37Fixed("t37_frozen", clone(frozen_k)),
              BlockGP("hp_global", trend=False), BlockGP("trend_hp", trend=True)]
    Xt, yt = load_train()
    pts = load_run(run)
    last = max(p["call"] for p in pts)
    rows, secs = [], {m.name: 0.0 for m in models}
    for k in range(2, last + 1):
        if calls and k not in calls:
            continue
        prev = [p for p in pts if p["call"] < k]
        cur = [p for p in pts if p["call"] == k]
        if not cur:
            continue
        X = np.vstack([Xt] + [p["W"][None] for p in prev])
        y = np.concatenate([yt, [p["R"] for p in prev]])
        i = int(np.argmin(y))
        c, R_inc = X[i], float(y[i])
        Xq = np.vstack([p["W"] for p in cur])
        for m in models:
            t0 = time.time()
            mu, sd = m.fit(X, y, c).predict(Xq)
            secs[m.name] += time.time() - t0
            for p, a, b in zip(cur, mu, sd):
                rows.append(dict(run=run, call=k, model=m.name, kind=p["kind"], move=p["move"],
                                 R=p["R"], R_inc=R_inc, pred=float(a), std=float(b)))
    return run, rows, secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--runs", nargs="*")
    ap.add_argument("--calls", nargs="*", type=int)
    ap.add_argument("--out", default=str(HERE / "preds.csv"))
    a = ap.parse_args()
    _setup_threads(2)
    import pandas as pd
    runs = a.runs or [f"{v}_s{s}" for v in ("v1", "v2", "v4", "v5", "v6") for s in (11, 2026, 7)] + ["v3_s2026"]
    runs = sorted(runs)
    fk, fk_repr = frozen_kernel()
    print("t37_frozen kernel:", fk_repr, flush=True)
    all_rows, t0 = [], time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for run, rows, secs in ex.map(run_one, [(r, fk, a.calls) for r in runs]):
            all_rows += rows
            print(f"{run}: {len(rows)} rows, fit+predict secs {({k: round(v) for k, v in secs.items()})}, "
                  f"elapsed {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(all_rows).to_csv(a.out, index=False)
    (HERE / "frozen_kernel.txt").write_text(fk_repr + "\n", encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
