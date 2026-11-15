"""Local rehearsal of the online phase: 20 sequential calls of <= 50 weight sets each.

    python rehearse.py --evaluator solver --variant v2 --seed 2026 --run v2_s2026
    python rehearse.py --evaluator synthetic --variant v2 --seed 1 --run dry --calls 2

The policy only ever receives the training rows and this run's own results (State). The real evaluator is
the local solver.exe run through analysis/lowr_search/pool.py with this folder's own worker copies and a per-run
log, so no earlier measurement is visible to the run (not even as a cache). Resumable: rerunning the same
command continues after the last finished call.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / "dataset_same8"
sys.path.insert(0, str(HERE))

from policy import BlockDescentPolicy, Params, State, norm_blocks  # noqa: E402


def load_training():
    csv = DATA / "train.csv"
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(csv.read_bytes()).hexdigest()
    assert digest == manifest["train_sha256"], "training table does not match its manifest"
    t = pd.read_csv(csv, float_precision="round_trip", dtype={"candidate_id": str})
    t = t.sort_values("candidate_id", kind="stable").reset_index(drop=True)
    X = t.iloc[:, 1:22].to_numpy(float)
    return X, t["ramping"].to_numpy(float)


def variant_params(variant: str) -> Params:
    """v1: base policy; v2: + training-derived shift bias; v3: v2 + fewer surrogate picks after call 6;
    v4: v2 + null-step filter, late coordinate scans, adaptive shares, no repeats (policy_v4.py)."""
    if variant == "v4":
        from policy_v4 import ParamsV4
        return ParamsV4(shift_bias=True, n_nav=3)
    return {"v1": Params(), "v2": Params(shift_bias=True, n_nav=3),
            "v3": Params(shift_bias=True, n_nav=3, share_model_late=0.2, share_random_late=0.3)}[variant]


def _run_config(evaluator: str, seed: int, variant: str, params: Params) -> dict:
    """Return the immutable configuration that identifies a rehearsal trajectory."""
    return {"evaluator": evaluator, "seed": seed, "variant": variant, "params": params.__dict__}


def _load_or_write_config(path: Path, expected: dict, log_exists: bool) -> None:
    """Reject accidental resume with a different policy instead of overwriting provenance."""
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        # v1_s2026 predates the explicit variant field; its Params identify it as v1.
        existing.setdefault("variant", "v1")
        old_params = {**Params().__dict__, **existing.get("params", {})}
        same = all(existing.get(key) == expected[key] for key in ("evaluator", "seed", "variant"))
        same = same and old_params == expected["params"]
        if not same:
            raise SystemExit(
                "run configuration differs from the existing log; choose a new --run name"
            )
        return
    if log_exists:
        raise SystemExit("cannot resume a run without its config.json")
    path.write_text(json.dumps(expected, indent=1), encoding="utf-8")


def make_policy(variant: str, seed: int):
    if variant == "v4":
        from policy_v4 import BlockDescentPolicyV4
        return BlockDescentPolicyV4(variant_params(variant), seed=seed)
    return BlockDescentPolicy(variant_params(variant), seed=seed)


def synthetic(W):
    """Cheap additive stand-in with plateaus, only to exercise the code path."""
    W = norm_blocks(W)
    out = []
    for w in W:
        tot = 120.0
        for b, (rs, c) in enumerate([(0.27, 1.0), (0.24, 0.4), (0.25, 0.7)]):
            s = w[b, 4] + w[b, 5]
            r = w[b, 4] / s if s > 0 else 0.5
            minors = w[b, [0, 1, 2, 3, 6]] / max(s, 1e-9)
            tot += c * (30 * (r - rs) ** 2 + 4 * np.log10(1 + 20 * minors.sum()) - 3 * s)
        h = int(hashlib.md5(np.round(w, 4).tobytes()).hexdigest()[:8], 16) / 2**32
        out.append(tot + 0.5 * h)
    return np.array(out)


class Screen:
    def __init__(self, X_train):
        self.known = norm_blocks(X_train).reshape(-1, 21)

    def check(self, W, sent):
        W = norm_blocks(W)[0].reshape(21)
        if not np.all(np.isfinite(W)) or (W < 0).any():
            return "invalid"
        for rows in (self.known, sent):
            if len(rows) and np.min(np.abs(rows - W).max(axis=1)) <= 1e-9:
                return "duplicate"
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluator", choices=["solver", "synthetic"], default="solver")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--run", required=True)
    ap.add_argument("--calls", type=int, default=20)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--variant", choices=["v1", "v2", "v3", "v4"], default="v1")
    a = ap.parse_args()
    if not 1 <= a.calls <= 20:
        ap.error("--calls must be between 1 and 20")
    if a.workers < 1:
        ap.error("--workers must be at least 1")

    run = HERE / "rehearsals" / a.run
    run.mkdir(parents=True, exist_ok=True)
    log = run / "online_log.jsonl"
    X_train, y_train = load_training()
    st = State(X_train, y_train)
    params = variant_params(a.variant)
    _load_or_write_config(run / "config.json", _run_config(a.evaluator, a.seed, a.variant, params), log.exists())
    if log.exists():                                      # resume: only fully finished calls count
        recs = [json.loads(l) for l in log.open(encoding="utf-8")]
        types = {r.get("type") for r in recs}
        if not types <= {"point", "call_done"}:
            raise SystemExit("log contains an unknown record type; inspect it before resuming")
        done = {r["call"] for r in recs if r.get("type") == "call_done"}
        done_records = [r for r in recs if r.get("type") == "call_done"]
        if len(done_records) != len(done):
            raise SystemExit("log contains duplicate call_done records; inspect it before resuming")
        all_calls = {r["call"] for r in recs}
        if done and done != set(range(1, max(done) + 1)):
            raise SystemExit("log calls are not contiguous; inspect it before resuming")
        if done and max(done) > 20:
            raise SystemExit("log exceeds the 20-call rehearsal budget; inspect it before resuming")
        if all_calls != done:
            raise SystemExit("log has an unfinished call; inspect it before resuming (decision 28)")
        for c in sorted(done):
            points = [r for r in recs if r.get("type") == "point" and r["call"] == c]
            summary = next(r for r in recs if r.get("type") == "call_done" and r["call"] == c)
            if summary.get("n") != len(points) or not 1 <= len(points) <= 50:
                raise SystemExit(f"call {c} has an invalid point count; inspect it before resuming")
            st.calls.append([{"W": np.array(r["W"]).reshape(3, 7),
                              "R": float("nan") if r["R"] is None else r["R"],
                              "meta": r["meta"]} for r in points])

    if a.evaluator == "solver":
        sys.path.insert(0, str(HERE.parent / "lowr_search"))
        from pool import SolverPool
        pool = SolverPool(n_workers=a.workers, root=HERE / "workers" / a.run, log=run / "pool_log.jsonl")
        evaluate = lambda W: pool.evaluate(W.reshape(-1, 21), tag=f"rehearsal:{a.run}")   # noqa: E731
    else:
        evaluate = synthetic

    policy = make_policy(a.variant, a.seed)
    screen = Screen(X_train)
    best = min([float(y_train.min())] + [e["R"] for c in st.calls for e in c if np.isfinite(e["R"])])
    while len(st.calls) < a.calls:
        t0 = time.time()
        props, info = policy.propose(st, 50)
        sent = np.array([e["W"].reshape(21) for c in st.calls for e in c]).reshape(-1, 21)
        batch, metas, dropped = [], [], 0
        for i, (W, meta) in enumerate(props):
            why = screen.check(W, np.vstack([sent] + [b.reshape(1, 21) for b in batch]) if batch else sent)
            if why:
                dropped += 1
                continue
            meta = dict(meta, id=i)
            batch.append(norm_blocks(W)[0])
            metas.append(meta)
        batch = np.array(batch)[:50]
        metas = metas[:len(batch)]
        if not 1 <= len(batch) <= 50:
            raise SystemExit("pre-submit screen left no valid points for this call")
        t1 = time.time()
        solver_invocations_before = getattr(pool, "n_solver_invocations", 0) if a.evaluator == "solver" else 0
        R = np.asarray(evaluate(batch), float)
        solver_invocations = (getattr(pool, "n_solver_invocations", 0) - solver_invocations_before
                              if a.evaluator == "solver" else 0)
        t2 = time.time()
        call = len(st.calls) + 1
        with log.open("a", encoding="utf-8") as fh:
            for W, r, m in zip(batch, R, metas):
                fh.write(json.dumps({"type": "point", "call": call, "W": W.reshape(21).tolist(),
                                     "R": None if not np.isfinite(r) else float(r), "meta": m}) + "\n")
            fh.write(json.dumps({"type": "call_done", "call": call, "n": len(batch), "dropped": dropped,
                                 "batch_min": float(np.nanmin(R)), "propose_secs": round(t1 - t0, 1),
                                 "eval_secs": round(t2 - t1, 1), "solver_invocations": solver_invocations,
                                 **info}) + "\n")
        st.calls.append([{"W": W, "R": float(r), "meta": m} for W, r, m in zip(batch, R, metas)])
        best = min(best, float(np.nanmin(R)))
        kinds = {}
        for m, r in zip(metas, R):
            k = m["kind"] if m["kind"] != "single" else f"b{m['building'] + 1}:{m['move']}"
            kinds.setdefault(k, []).append(r)
        brief = " ".join(f"{k}={min(v):.2f}" for k, v in sorted(kinds.items()))
        print(f"call {call:2d}: min {np.nanmin(R):.4f} best {best:.4f} | inc {info['R_inc']:.3f} "
              f"scale {info['scale']} | propose {t1 - t0:.0f}s eval {t2 - t1:.0f}s | {brief}", flush=True)
    print(f"final best {best:.5f}")


if __name__ == "__main__":
    main()
