"""SDM374 online phase: policy v4, seed 7 (verification package).

    python online_run.py check                       # environment, data, solver checksums
    python online_run.py verify [--calls 1 10 20]    # no solver: recompute the reference batches, compare
    python online_run.py run --solver PATH\\solver.exe [--out runs\\NAME]   # 20 real calls, compared as they go

Data boundary: before call 1 the policy sees only data/train.csv (2048 rows); afterwards also the results of
its own calls in this run. reference/ is read only by the comparison code, never passed to the policy.
`run` never deletes or edits usage.count; it consumes up to 20 calls of the given solver copy's quota.
"""
from __future__ import annotations

import os

# The BLAS thread count changes the summation order of linear algebra and so, eventually, the batches.
# The reference run used OpenBLAS's default (one thread per logical CPU, 20 on the machine that made it).
# Set OPENBLAS_NUM_THREADS before starting Python to try another value; `check` reports what is in effect.
# OpenBLAS picks its kernels by CPU model; on the reference machine it fell back to the generic "Prescott"
# kernels, which run on every x86-64 CPU, so they are forced here to remove one machine-dependent difference.
# The thread count cannot be forced above the machine's logical CPU count.
os.environ.setdefault("OPENBLAS_CORETYPE", "Prescott")

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "policy"))

from policy import State, norm_blocks  # noqa: E402
from policy_v4 import BlockDescentPolicyV4, ParamsV4  # noqa: E402

SEED = 7
CALLS = 20
BATCH = 50
SOLVER_SHA256 = "293867b81dbf0c6194f97a57115487b9a39239d0c7dadbd51d749a203a077247"
TRAIN_SHA256 = json.loads((HERE / "data" / "manifest.json").read_text(encoding="utf-8"))["train_sha256"]
PINNED = {"python": "3.11", "numpy": "1.24.4", "scipy": "1.10.1", "scikit-learn": "1.2.2"}
REFERENCE = HERE / "reference" / "v4_s7_online_log.jsonl"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_training():
    """The 2048 training rows, checked against the manifest, sorted by candidate_id (as in the rehearsal)."""
    path = HERE / "data" / "train.csv"
    if sha256(path) != TRAIN_SHA256:
        raise SystemExit(f"{path} does not match data/manifest.json")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: r["candidate_id"])
    cols = [c for c in rows[0] if c not in ("candidate_id", "ramping")]
    assert len(cols) == 21
    X = np.array([[float(r[c]) for c in cols] for r in rows])
    y = np.array([float(r["ramping"]) for r in rows])
    return X, y


def make_policy():
    return BlockDescentPolicyV4(ParamsV4(shift_bias=True, n_nav=3), seed=SEED)


def screen(props, X_train, sent):
    """Pre-submit check: finite, non-negative, not a training row, not a point sent before (max abs 1e-9).
    Returns the first 50 passing proposals in the policy's order and their metadata."""
    known = norm_blocks(X_train).reshape(-1, 21)
    batch, metas = [], []
    for W, meta in props:
        w = norm_blocks(W)[0].reshape(21)
        if not np.all(np.isfinite(w)) or (w < 0).any():
            continue
        rows = [known] + ([sent] if len(sent) else []) + ([np.array(batch)] if batch else [])
        if any(np.min(np.abs(r - w).max(axis=1)) <= 1e-9 for r in rows):
            continue
        batch.append(w)
        metas.append(meta)
    return np.array(batch)[:BATCH], metas[:BATCH]


def propose(policy, st, X_train):
    props, info = policy.propose(st, BATCH)
    sent = np.array([e["W"].reshape(21) for c in st.calls for e in c]).reshape(-1, 21)
    batch, metas = screen(props, X_train, sent)
    return batch, metas, info


def load_reference():
    recs = [json.loads(l) for l in REFERENCE.open(encoding="utf-8")]
    pts = [r for r in recs if r["type"] == "point"]
    by_call = {}
    for r in pts:
        by_call.setdefault(r["call"], []).append(r)
    return by_call


def compare(batch, ref_points, R=None):
    ref_W = np.array([r["W"] for r in ref_points])
    if len(batch) != len(ref_W):
        return False, f"{len(batch)} points vs {len(ref_W)} in the reference"
    dw = float(np.max(np.abs(batch - ref_W)))
    msg = f"max |dW| {dw:.1e}"
    ok = dw <= 1e-12
    if R is not None:
        ref_R = np.array([r["R"] for r in ref_points], float)
        dR = float(np.max(np.abs(np.asarray(R) - ref_R)))
        msg += f", max |dR| {dR:.1e}"
        ok = ok and dR <= 1e-9
    return ok, msg


# ---------------------------------------------------------------------------------------------- commands
def cmd_check(args):
    import scipy
    import sklearn
    got = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
           "scikit-learn": sklearn.__version__}
    print(f"machine: {platform.platform()} | {platform.processor()} | {os.cpu_count()} logical CPUs "
          f"(reference: 20)")
    ok = True
    for k, v in PINNED.items():
        match = got[k].startswith(v) if k == "python" else got[k] == v
        ok &= match
        print(f"  {k:13s} {got[k]:10s} pinned {v:8s} {'ok' if match else 'DIFFERENT (results may differ)'}")
    from threadpoolctl import threadpool_info
    blas = [f"{i.get('internal_api')} {i.get('version')} threads={i.get('num_threads')} arch={i.get('architecture')}"
            for i in threadpool_info() if i.get("user_api") == "blas"]
    print(f"  BLAS          {'; '.join(blas)} (reference: openblas 0.3.21 in numpy and 0.3.18 in scipy, 20 threads, Prescott)"
          f"  OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')}")
    try:
        load_training()
        print("  data/train.csv sha256 ok (2048 rows)")
    except SystemExit as e:
        ok = False
        print(f"  {e}")
    bad = []
    for line in (HERE / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines():
        h, name = line.split(maxsplit=1)
        if sha256(HERE / name.strip().lstrip("*")) != h:
            bad.append(name)
    print("  package files:", "ok" if not bad else f"CHANGED: {bad}")
    ok &= not bad
    if args.solver:
        s = Path(args.solver)
        if not s.is_file():
            print(f"  solver: {s} not found"); ok = False
        else:
            good = sha256(s) == SOLVER_SHA256
            ok &= good
            print(f"  solver sha256 {'ok' if good else 'DIFFERENT from the pinned course build'}; "
                  f"usage.count {'present' if (s.parent / 'usage.count').exists() else 'absent'}")
    print("check:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_verify(args):
    X, y = load_training()
    ref = load_reference()
    calls = args.calls or list(range(1, CALLS + 1))
    policy = make_policy()
    ok_all = True
    t0 = time.time()
    for c in calls:
        st = State(X, y)
        for k in range(1, c):                       # the results of earlier calls, exactly as measured
            st.calls.append([{"W": np.array(r["W"]).reshape(3, 7), "R": r["R"], "meta": r["meta"]}
                             for r in ref[k]])
        batch, _, _ = propose(policy, st, X)
        ok, msg = compare(batch, ref[c])
        ok_all &= ok
        print(f"call {c:2d}: {'identical' if ok else 'DIFFERENT'} ({msg})", flush=True)
    print(f"verify: {'PASS' if ok_all else 'FAIL'} ({len(calls)} calls, {time.time() - t0:.0f} s)")
    return 0 if ok_all else 1


def run_solver(exe: Path, batch: np.ndarray, timeout: float):
    payload = json.dumps({"weights": batch.tolist()})
    proc = subprocess.run([str(exe)], input=payload, capture_output=True, text=True, cwd=str(exe.parent),
                          timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if proc.returncode != 0:
        raise RuntimeError(f"solver exit {proc.returncode}: {proc.stderr.strip()[:500]}")
    R = [float(v) for v in json.loads(proc.stdout)["R"]]
    if len(R) != len(batch) or not all(np.isfinite(R)):
        raise RuntimeError(f"solver returned {len(R)} values for {len(batch)} sets")
    return R


def cmd_run(args):
    exe = Path(args.solver).resolve()
    if sha256(exe) != SOLVER_SHA256:
        raise SystemExit(f"{exe} is not the pinned course solver.exe ({SOLVER_SHA256[:12]}...)")
    out = Path(args.out or HERE / "runs" / time.strftime("%Y%m%d-%H%M%S")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = out / "online_log.jsonl"
    X, y = load_training()
    st = State(X, y)
    if log.exists():                                 # resume after the last finished call
        recs = [json.loads(l) for l in log.open(encoding="utf-8")]
        started = {r["call"] for r in recs if r["type"] == "call_start"}
        done = {r["call"] for r in recs if r["type"] == "call_done"}
        if started - done:
            raise SystemExit(f"call {sorted(started - done)} started but did not finish: the solver may have "
                             f"charged it; inspect {log} before deciding how to continue")
        for c in sorted(done):
            st.calls.append([{"W": np.array(r["W"]).reshape(3, 7), "R": r["R"], "meta": r["meta"]}
                             for r in recs if r["type"] == "point" and r["call"] == c])
        print(f"resuming after call {len(st.calls)}")
    policy = make_policy()
    ref = load_reference()
    best = float(y.min())
    for c in st.calls:
        best = min(best, min(e["R"] for e in c))
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"type": "run_info", "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                             "machine": platform.platform(), "python": platform.python_version(),
                             "numpy": np.__version__, "solver": str(exe)}) + "\n")
    last = min(CALLS, args.max_calls or CALLS)
    while len(st.calls) < last:
        c = len(st.calls) + 1
        t0 = time.time()
        batch, metas, info = propose(policy, st, X)
        t1 = time.time()
        with log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"type": "call_start", "call": c, "n": len(batch)}) + "\n")
        R = run_solver(exe, batch, args.timeout)
        t2 = time.time()
        ok, msg = compare(batch, ref[c], R)
        with log.open("a", encoding="utf-8") as fh:
            for w, r, m in zip(batch, R, metas):
                fh.write(json.dumps({"type": "point", "call": c, "W": w.tolist(), "R": r, "meta": m}) + "\n")
            fh.write(json.dumps({"type": "call_done", "call": c, "n": len(batch), "batch_min": min(R),
                                 "propose_secs": round(t1 - t0, 1), "eval_secs": round(t2 - t1, 1),
                                 "matches_reference": ok, "compare": msg, **info}) + "\n")
        st.calls.append([{"W": w.reshape(3, 7), "R": r, "meta": m} for w, r, m in zip(batch, R, metas)])
        best = min(best, min(R))
        print(f"call {c:2d}: batch min {min(R):.4f}  best {best:.4f}  | propose {t1 - t0:.0f}s solver {t2 - t1:.0f}s"
              f" | {'matches reference' if ok else 'DIFFERS from reference: ' + msg}", flush=True)
    print(f"run {'finished' if len(st.calls) == CALLS else f'stopped after call {len(st.calls)} (--max-calls)'}: "
          f"best R {best:.10f}; log {log}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check"); p.add_argument("--solver")
    p = sub.add_parser("verify"); p.add_argument("--calls", type=int, nargs="*")
    p = sub.add_parser("run"); p.add_argument("--solver", required=True); p.add_argument("--out")
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--max-calls", type=int, help="stop after this call (testing only; rerun to continue)")
    args = ap.parse_args()
    sys.exit({"check": cmd_check, "verify": cmd_verify, "run": cmd_run}[args.cmd](args))


if __name__ == "__main__":
    main()
