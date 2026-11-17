"""Standalone SDM374 online runner.

    python run.py check [--solver PATH\\solver.exe]
    python run.py run --evaluator synthetic --max-calls 2
    python run.py run --evaluator solver --solver PATH\\solver.exe

Before call 1 the policy sees only data/train.csv; afterwards it sees only results from this run. The
solver evaluator never edits usage.count. Every started call is checkpointed before evaluation and is
never silently resent after an interruption.
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

CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
SEED = int(CONFIG["seed"])
CALLS = int(CONFIG["calls"])
BATCH = int(CONFIG["batch"])
SOLVER_SHA256 = "293867b81dbf0c6194f97a57115487b9a39239d0c7dadbd51d749a203a077247"
TRAIN_SHA256 = json.loads((HERE / "data" / "manifest.json").read_text(encoding="utf-8"))["train_sha256"]
PINNED = {"python": "3.11", "numpy": "1.24.4", "scipy": "1.10.1", "scikit-learn": "1.2.2"}


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
    return BlockDescentPolicyV4(ParamsV4(**CONFIG["params"]), seed=SEED)


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


def _json_default(value):
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot encode {type(value).__name__}")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes((json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n")
                          .encode("utf-8"))
    os.replace(temporary, path)


def package_hashes() -> dict[str, str]:
    names = ["run.py", "config.json", "data/train.csv", "data/manifest.json",
             "policy/policy.py", "policy/policy_v4.py", "policy/surrogate.py"]
    return {name: sha256(HERE / name) for name in names}


def ensure_run_manifest(out: Path, solver: Path | None) -> None:
    path = out / "checkpoints" / "run.json"
    current = package_hashes()
    if solver is not None and solver.is_file():
        current["solver.exe"] = sha256(solver)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("files") != current:
            raise SystemExit("checkpoint run.json does not match the current package; refuse to resume")
        return
    write_json(path, {"version": 1, "seed": SEED, "calls": CALLS, "batch": BATCH, "files": current})


def checkpoint(out: Path, call: int, payload: dict, previous_manifest: str | None) -> str:
    """Write one complete checkpoint atomically and return its manifest hash."""
    root = out / "checkpoints"
    temporary = root / f"call_{call:02d}.tmp"
    final = root / f"call_{call:02d}"
    temporary.mkdir(parents=True, exist_ok=True)
    write_json(temporary / "result.json", payload)
    files = {p.relative_to(temporary).as_posix(): sha256(p)
             for p in temporary.rglob("*") if p.is_file()}
    write_json(temporary / "manifest.json", {"version": 1, "call": call,
                                               "previous_manifest_sha256": previous_manifest,
                                               "files": files})
    if final.exists():
        raise SystemExit(f"checkpoint {final} already exists")
    os.replace(temporary, final)
    return sha256(final / "manifest.json")


def verify_checkpoints(out: Path) -> str | None:
    """Verify the hash chain and return the latest manifest hash."""
    previous = None
    folders = sorted((out / "checkpoints").glob("call_[0-9][0-9]"))
    for folder in folders:
        manifest_path = folder / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("previous_manifest_sha256") != previous:
            raise SystemExit(f"checkpoint chain broken at {folder}")
        for name, expected in manifest["files"].items():
            if sha256(folder / name) != expected:
                raise SystemExit(f"checkpoint file changed: {folder / name}")
        previous = sha256(manifest_path)
    return previous


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
        s = Path(args.solver).resolve()
        if not s.is_file():
            print(f"  solver: {s} not found"); ok = False
        else:
            good = sha256(s) == SOLVER_SHA256
            ok &= good
            print(f"  solver sha256 {'ok' if good else 'DIFFERENT from the pinned course build'}; "
                  f"usage.count {'present' if (s.parent / 'usage.count').exists() else 'absent'}")
    print("check:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


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


def run_synthetic(batch: np.ndarray, X_train: np.ndarray, y_train: np.ndarray) -> list[float]:
    """Deterministic development truth derived from the checked training table only."""
    center = X_train[int(np.argmin(y_train))]
    distance = np.sum((batch - center[None]) ** 2, axis=1)
    return (float(y_train.min()) + 100.0 * distance).tolist()


def check_solver_stdin(exe: Path) -> None:
    """Check the course executable's stdin channel with an invalid, non-billable request."""
    bad = np.zeros((1, 3, 7), dtype=float)
    bad[0, 0, 0] = -1.0
    proc = subprocess.run([str(exe)], input=json.dumps({"weights": bad.tolist()}), capture_output=True,
                          text=True, cwd=str(exe.parent), timeout=60,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    stderr = proc.stderr.strip()
    if proc.returncode != 2 or "Weights must be finite and nonnegative" not in stderr or "Expecting value" in stderr:
        raise SystemExit("solver stdin preflight failed; refusing to spend a call: " + stderr[:500])


def cmd_run(args):
    if args.evaluator == "solver":
        exe = Path(args.solver or HERE / "solver.exe").resolve()
        if not exe.is_file() or sha256(exe) != SOLVER_SHA256:
            raise SystemExit(f"{exe} is not the pinned course solver.exe ({SOLVER_SHA256[:12]}...)")
    else:
        exe = None
    out = Path(args.out or HERE / "runs" / time.strftime("%Y%m%d-%H%M%S")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = out / "online_log.jsonl"
    X, y = load_training()
    st = State(X, y)
    ensure_run_manifest(out, exe)
    previous_manifest = verify_checkpoints(out)
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
    if len(st.calls) == 0 and args.evaluator == "solver":
        check_solver_stdin(exe)
    policy = make_policy()
    best = float(y.min())
    for c in st.calls:
        best = min(best, min(e["R"] for e in c))
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"type": "run_info", "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                             "machine": platform.platform(), "python": platform.python_version(),
                             "numpy": np.__version__, "evaluator": args.evaluator,
                             "solver": str(exe) if exe else None}) + "\n")
    last = min(CALLS, args.max_calls or CALLS)
    while len(st.calls) < last:
        c = len(st.calls) + 1
        t0 = time.time()
        batch, metas, info = propose(policy, st, X)
        t1 = time.time()
        with log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"type": "call_start", "call": c, "n": len(batch),
                                 "evaluator": args.evaluator}) + "\n")
            fh.flush(); os.fsync(fh.fileno())
        write_json(out / "checkpoints" / f"call_{c:02d}.sent.json",
                   {"call": c, "evaluator": args.evaluator, "weights": batch.tolist(),
                    "batch_sha256": hashlib.sha256(json.dumps(batch.tolist()).encode()).hexdigest()})
        try:
            R = run_solver(exe, batch, args.timeout) if args.evaluator == "solver" else run_synthetic(batch, X, y)
        except Exception as error:
            with log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"type": "call_failed", "call": c, "error": repr(error)}) + "\n")
                fh.flush(); os.fsync(fh.fileno())
            checkpoint(out, c, {"status": "failed", "call": c, "error": repr(error),
                                "weights": batch.tolist()}, previous_manifest)
            raise
        t2 = time.time()
        with log.open("a", encoding="utf-8") as fh:
            for w, r, m in zip(batch, R, metas):
                fh.write(json.dumps({"type": "point", "call": c, "W": w.tolist(), "R": r, "meta": m}) + "\n")
            fh.write(json.dumps({"type": "call_done", "call": c, "n": len(batch), "batch_min": min(R),
                                 "propose_secs": round(t1 - t0, 1), "eval_secs": round(t2 - t1, 1),
                                 **info}) + "\n")
            fh.flush(); os.fsync(fh.fileno())
        previous_manifest = checkpoint(out, c, {"status": "complete", "call": c,
                                                "evaluator": args.evaluator, "weights": batch.tolist(),
                                                "R": R, "meta": metas, "diagnostics": info}, previous_manifest)
        st.calls.append([{"W": w.reshape(3, 7), "R": r, "meta": m} for w, r, m in zip(batch, R, metas)])
        best = min(best, min(R))
        print(f"call {c:2d}: batch min {min(R):.4f}  best {best:.4f}  | propose {t1 - t0:.0f}s solver {t2 - t1:.0f}s"
              f" | {args.evaluator}", flush=True)
    print(f"run {'finished' if len(st.calls) == CALLS else f'stopped after call {len(st.calls)} (--max-calls)'}: "
          f"best R {best:.10f}; log {log}")
    return 0


def cmd_verify(args):
    """Compatibility alias: run a short synthetic end-to-end check without a solver."""
    calls = max(args.calls or [2])
    return cmd_run(argparse.Namespace(evaluator="synthetic", solver=None, out=args.out,
                                      timeout=60.0, max_calls=calls))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check"); p.add_argument("--solver")
    p = sub.add_parser("verify"); p.add_argument("--calls", type=int, nargs="*"); p.add_argument("--out")
    p = sub.add_parser("run"); p.add_argument("--evaluator", choices=("solver", "synthetic"), default="solver")
    p.add_argument("--solver"); p.add_argument("--out")
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--max-calls", type=int, help="stop after this call (testing only; rerun to continue)")
    args = ap.parse_args()
    sys.exit({"check": cmd_check, "verify": cmd_verify, "run": cmd_run}[args.cmd](args))


if __name__ == "__main__":
    main()
