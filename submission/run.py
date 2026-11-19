"""SDM374 Project 1 online run: the course solver/run.py with its two todos filled in (decision 31).

The course template calls solver.exe 20 times, piping weight.json through PowerShell, and leaves "read R,
refit the surrogate, write the next weight.json" as todo. This file keeps that loop and fills it in: before
each call the policy (policy.py, surrogate.py, constants in config.json) refits on train.csv plus this run's
results and writes the next batch of at most 50 weight sets; the batch goes to solver.exe through the same
PowerShell pipe; the R values come back into the next refit.

    uv run run.py                          graded run (solver.exe next to this file; resumes if interrupted)
    uv run run.py --solver D:\\dev\\solver.exe --out runs\\rehearsal-1     development rehearsal on another copy
    uv run run.py check [--solver PATH]    environment, pinned files and solver, no call
    uv run run.py verify --reference LOG   recompute every batch of a recorded run, no call
    uv run run.py compare RUN_A RUN_B      first call and field where two runs differ

Data boundary (decisions 19, 27, 29, 32): before call 1 the policy reads only train.csv (the 2048 course
training rows); afterwards only the results of this run. No locally measured point, R value or weight is
in this folder; config.json holds constants only. This file never touches usage.count.
"""
from __future__ import annotations

import os

# OpenBLAS picks its kernels by CPU model, and the kernel changes the last bits of linear algebra, which can
# change a policy choice. The v7 reference log was recorded with the rehearsal machine's native kernels
# (Intel i5-10600, which OpenBLAS maps to "Cooperlake"; "Haswell" gives the same batches and runs on any x86-64
# CPU with AVX2), so they are forced here. The generic "Prescott" kernels used by the v4 package do not
# reproduce the v7 log at call 3. This only affects an OpenBLAS-linked numpy (the uv-locked pip wheels).
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
# The thread count is part of the arithmetic too: the reference log was recorded on 12 logical CPUs
# (environment.logical_cpus = 12), and the same Haswell kernels with another thread count already differ at
# call 3. It is pinned for the same reason as the kernels; an externally set value still wins.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "12")

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from policy import (BlockDescentPolicyV4, BlockDescentPolicyV7, ParamsV4, ParamsV7,  # noqa: E402
                    State, norm_blocks)

CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
SEED = int(CONFIG["seed"])
CALLS = int(CONFIG["calls"])
BATCH = int(CONFIG["batch"])
TIMEOUT = float(CONFIG["solver_timeout_seconds"])
# Pinned values (README): the released course solver.exe (solver/CHECKSUMS.sha256, decision 30) and train.csv.
SOLVER_SHA256 = "293867b81dbf0c6194f97a57115487b9a39239d0c7dadbd51d749a203a077247"
TRAIN_SHA256 = "877b5f8ee265245dd693148a5d7b11f28f48b426e43ac3ac1be8b7a934249023"
TRAIN_ROWS = 2048
# A resume refuses to continue when any of these differs from checkpoints/run.json (decision 31).
FROZEN = ("config.json", "train.csv", "run.py", "policy.py", "surrogate.py")
PINNED = {"python": "3.11", "numpy": "1.24.4", "scipy": "1.10.1", "scikit-learn": "1.2.2"}
# The solver's own messages (analysis/solver_smoke/reports/, issue #28).
STDIN_OK = "Weights must be finite and nonnegative"
STDIN_EMPTY = "Expecting value"
QUOTA_USED = "Usage limit reached"


# ---------------------------------------------------------------------------------------------- files
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _plain(value):
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def write_json(path: Path, value) -> None:
    """Write, flush and fsync; never a partial file under the final name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with open(temporary, "wb") as fh:
        fh.write((json.dumps(_plain(value), indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(temporary, path)


def write_text(path: Path, text: str) -> None:
    with open(path, "wb") as fh:
        fh.write(text.encode("utf-8"))
        fh.flush()
        os.fsync(fh.fileno())


def canonical(value):
    """Floats as float.hex, so two machines compare bit for bit (decision 31, Comparison)."""
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value.hex() if math.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return [canonical(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    return value


# ---------------------------------------------------------------------------------------------- data, policy
def load_training():
    """The 2048 training rows, checked against the pinned sha256, sorted by candidate_id (as in the rehearsal)."""
    path = HERE / "train.csv"
    if sha256(path) != TRAIN_SHA256:
        raise SystemExit(f"{path} does not match the pinned sha256 {TRAIN_SHA256[:12]}…; stop")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: r["candidate_id"])
    cols = [c for c in rows[0] if c not in ("candidate_id", "ramping")]
    if len(rows) != TRAIN_ROWS or len(cols) != 21:
        raise SystemExit(f"train.csv has {len(rows)} rows and {len(cols)} weight columns; expected 2048 and 21")
    X = np.array([[float(r[c]) for c in cols] for r in rows])
    y = np.array([float(r["ramping"]) for r in rows])
    return X, y


def make_policy():
    if CONFIG["variant"] == "v7":
        return BlockDescentPolicyV7(ParamsV7(**CONFIG["params"]), seed=SEED)
    return BlockDescentPolicyV4(ParamsV4(**CONFIG["params"]), seed=SEED)


def rng_streams(call: int) -> list:
    """Random streams a call draws from: v4 uses [seed, call, 4]; v7 keeps that stream for the v4 batch
    and adds [seed, call, 7] for its multiscale replacements (calls 1-freeze_calls use only [seed, call, 4])."""
    if CONFIG["variant"] == "v7":
        streams = [[SEED, call, 4]]
        if call > int(CONFIG["params"].get("freeze_calls", 11)):
            streams.append([SEED, call, 7])
        return streams
    return [[SEED, call, 4]]


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


def blas_report() -> str:
    try:
        from threadpoolctl import threadpool_info
        blas = [f"{i.get('internal_api')} {i.get('version')} threads={i.get('num_threads')} "
                f"arch={i.get('architecture')}" for i in threadpool_info() if i.get("user_api") == "blas"]
    except Exception as error:  # report, never fail on this
        blas = [f"unknown ({error!r})"]
    return (f"BLAS: {'; '.join(blas)} | OPENBLAS_CORETYPE={os.environ.get('OPENBLAS_CORETYPE')} "
            f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')} "
            f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')}")


# ---------------------------------------------------------------------------------------------- evaluators
class Outcome:
    """`ok`: R for every set. `rejected`: the solver refused before evaluating anything (exit 2; costs no
    call). `failed`: anything else after the solver started; the batch counts as sent without R."""

    def __init__(self, kind, R=None, returncode=None, stdout="", stderr=""):
        self.kind, self.R, self.returncode, self.stdout, self.stderr = kind, R, returncode, stdout, stderr


def _ps_quote(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def powershell_pipe(exe: Path, weight_json: Path, timeout: float):
    """`Get-Content weight.json -Raw | .\\solver.exe`, as in the course run.py, run in the exe's folder.
    Returns (returncode or None on timeout, stdout, stderr). `powershell -Command` exits 1 whenever a native
    command fails, so the exe's own exit code (2 = rejected before evaluation) is passed on explicitly."""
    command = f"Get-Content -LiteralPath {_ps_quote(weight_json)} -Raw | & {_ps_quote(exe)}; exit $LASTEXITCODE"
    process = subprocess.Popen(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                               cwd=str(exe.parent), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, encoding="utf-8", errors="replace")
    try:
        out, err = process.communicate(timeout=timeout)
        return process.returncode, out, err
    except subprocess.TimeoutExpired:
        # solver.exe starts a second process and a worker pool; kill the whole tree.
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True, check=False)
        out, err = process.communicate()
        return None, out, err + f"\n[run.py] timeout after {timeout:.0f} s; process tree killed"


class SolverEvaluator:
    name = "solver"

    def __init__(self, exe: Path, timeout: float):
        self.exe, self.timeout = exe, timeout

    def stdin_check(self) -> None:
        """Before call 1: send one invalid set (a negative weight). The solver must reject it with its
        validation message, which proves stdin reaches the exe; a rejection costs no call (decision 28)."""
        bad = np.full((1, 3, 7), 1 / 7)
        bad[0, 0, 0] = -1.0
        with tempfile.TemporaryDirectory(prefix="sdm374-stdin-check-") as folder:
            path = Path(folder) / "weight.json"
            path.write_text(json.dumps({"weights": bad.tolist()}), encoding="utf-8")
            code, _, err = powershell_pipe(self.exe, path, timeout=120)
        if QUOTA_USED in err:
            raise SystemExit(f"stdin check: this solver copy's quota is used up ({err.strip()[:300]}); stop")
        if code != 2 or STDIN_OK not in err or STDIN_EMPTY in err:
            raise SystemExit("stdin check failed (exit "
                             f"{code}): the weights did not reach solver.exe; no call was spent. "
                             f"stderr: {err.strip()[:500]}")
        print("stdin check: solver.exe received the test input and rejected it as expected (no call spent)")

    def evaluate(self, batch: np.ndarray, folder: Path) -> Outcome:
        weight_json = folder / "weight.json"
        write_text(weight_json, json.dumps({"weights": batch.reshape(-1, 3, 7).tolist()}))
        try:
            code, out, err = powershell_pipe(self.exe, weight_json, self.timeout)
        except Exception as error:  # could not even start PowerShell: still treat as sent (decision 28)
            return Outcome("failed", stderr=f"[run.py] {error!r}")
        if code == 2:
            return Outcome("rejected", returncode=code, stdout=out, stderr=err)
        if code == 0:
            try:
                R = [float(v) for v in json.loads(out)["R"]]
                if len(R) == len(batch) and all(math.isfinite(r) for r in R):
                    return Outcome("ok", R, code, out, err)
            except (ValueError, KeyError, TypeError):
                pass
        return Outcome("failed", returncode=code, stdout=out, stderr=err)


class SyntheticEvaluator:
    """Closed-form stand-in for tests and machine comparisons; never used by the graded run."""
    name = "synthetic"

    def __init__(self, X_train, y_train):
        self.center = X_train[int(np.argmin(y_train))]
        self.floor = float(y_train.min())

    def stdin_check(self) -> None:
        pass

    def evaluate(self, batch: np.ndarray, folder: Path) -> Outcome:
        distance = np.sum((batch - self.center[None]) ** 2, axis=1)
        return Outcome("ok", (self.floor + 100.0 * distance).tolist(), 0)


class ExternalEvaluator:
    """Development only: `--evaluator module:Class` (module importable from the working directory), whose
    instance returns one R per row of an (n, 21) array. Used for the T17 / T18 replay truths."""

    def __init__(self, spec: str):
        module, _, cls = spec.partition(":")
        sys.path.insert(0, os.getcwd())
        self.impl = getattr(importlib.import_module(module), cls)()
        self.name = spec

    def stdin_check(self) -> None:
        pass

    def evaluate(self, batch: np.ndarray, folder: Path) -> Outcome:
        R = [float(r) for r in self.impl(batch)]
        return Outcome("ok", R, 0)


# ---------------------------------------------------------------------------------------------- checkpoints
def frozen_hashes() -> dict:
    return {name: sha256(HERE / name) for name in FROZEN}


def call_dir(root: Path, call: int) -> Path:
    return root / f"call_{call:02d}"


def check_chain(root: Path) -> list:
    """Verify every complete checkpoint and its link; return the calls as the policy sees them.
    Stops (naming the folder) on any damage instead of guessing."""
    previous = sha256(root / "run.json")
    calls = []
    call = 1
    while call_dir(root, call).is_dir():
        folder = call_dir(root, call)
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("previous_manifest_sha256") != previous:
            raise SystemExit(f"{folder}: its link to the previous checkpoint does not hold; stop")
        for name, expected in manifest["files"].items():
            if not (folder / name).is_file() or sha256(folder / name) != expected:
                raise SystemExit(f"{folder / name} does not match its manifest; stop")
        result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        calls.append([{"W": np.array(w, float).reshape(3, 7), "R": float("nan") if r is None else float(r),
                       "meta": m} for w, r, m in zip(result["weights"], result["R"], result["meta"])])
        previous = sha256(folder / "manifest.json")
        call += 1
    for later in sorted(root.glob("call_[0-9][0-9]")):
        if later.is_dir() and int(later.name[5:]) > call:
            raise SystemExit(f"{later} exists but call_{call:02d}/ is missing; stop")
    return calls


def write_checkpoint(root: Path, call: int, tmp: Path, *, status: str, batch, R, metas, info, outcome: Outcome,
                     evaluator: str, solver_sha: str | None) -> None:
    """Fill call_<kk>.tmp/, fsync every file, then rename it to call_<kk>/ (decision 31, Checkpoints)."""
    tmp.mkdir(parents=True, exist_ok=True)
    rng = rng_streams(call)
    record = {"call": call, "weights": batch, "R": R, "meta": metas,
              "policy": {"seed": SEED, "rng": rng, "params": CONFIG["params"]}}
    write_json(tmp / "result.json", {"status": status, "call": call, "evaluator": evaluator,
                                     "weights": batch, "R": R, "meta": metas,
                                     "solver_returncode": outcome.returncode})
    write_json(tmp / "policy_state.json", {"seed": SEED, "rng": rng, "params": CONFIG["params"],
                                           "variant": CONFIG["variant"], "diagnostics": info})
    write_json(tmp / "inputs.json", {"train_sha256": sha256(HERE / "train.csv"), "train_rows": TRAIN_ROWS,
                                     "solver_sha256": solver_sha, "evaluator": evaluator})
    write_text(tmp / "canonical.json", json.dumps(canonical(record), sort_keys=True, separators=(",", ":")) + "\n")
    write_text(tmp / "stdout.txt", outcome.stdout or "")
    write_text(tmp / "stderr.txt", outcome.stderr or "")
    if not (tmp / "weight.json").exists():
        write_text(tmp / "weight.json", json.dumps({"weights": np.asarray(batch).reshape(-1, 3, 7).tolist()}))
    previous = sha256(root / "run.json") if call == 1 else sha256(call_dir(root, call - 1) / "manifest.json")
    files = {p.name: sha256(p) for p in sorted(tmp.iterdir()) if p.is_file() and p.name != "manifest.json"}
    write_json(tmp / "manifest.json", {"call": call, "previous_manifest_sha256": previous, "files": files})
    os.replace(tmp, call_dir(root, call))


# ---------------------------------------------------------------------------------------------- commands
def resolve_evaluator(args, X, y):
    if args.evaluator == "solver":
        exe = Path(args.solver).resolve() if args.solver else HERE / "solver.exe"
        if not exe.is_file():
            raise SystemExit(f"{exe} not found: put the course solver.exe next to run.py or pass --solver PATH")
        digest = sha256(exe)
        if digest != SOLVER_SHA256:
            # The course computer's exe is the one that is graded (decision 31): warn, record, continue.
            print(f"WARNING: {exe} sha256 {digest[:12]}… differs from the pinned {SOLVER_SHA256[:12]}…; "
                  "continuing; the hash is recorded in every checkpoint")
        return SolverEvaluator(exe, args.timeout), exe, digest
    if args.evaluator == "synthetic":
        return SyntheticEvaluator(X, y), None, None
    return ExternalEvaluator(args.evaluator), None, None


def cmd_run(args) -> int:
    print(f"{platform.platform()} | Python {platform.python_version()} | numpy {np.__version__} | "
          f"{os.cpu_count()} logical CPUs")
    print(blas_report())
    X, y = load_training()
    evaluator, exe, solver_sha = resolve_evaluator(args, X, y)
    out = Path(args.out).resolve() if args.out else HERE
    root = out / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    run_json = root / "run.json"
    frozen = frozen_hashes()
    if run_json.exists():
        saved = json.loads(run_json.read_text(encoding="utf-8"))
        changed = [n for n in FROZEN if saved["files"].get(n) != frozen[n]]
        if changed:
            raise SystemExit(f"{changed} differ from {run_json}; refusing to resume with other code or data")
        if saved.get("evaluator") != evaluator.name:
            raise SystemExit(f"{run_json} was written for evaluator {saved.get('evaluator')!r}; stop")
    else:
        write_json(run_json, {"seed": SEED, "calls": CALLS, "batch": BATCH, "evaluator": evaluator.name,
                              "files": {**frozen, "solver.exe": solver_sha},
                              "started": time.strftime("%Y-%m-%dT%H:%M:%S")})

    st = State(X, y)
    st.calls = check_chain(root)
    k = len(st.calls) + 1
    rejected = root / f"call_{k:02d}.rejected.json"
    sent = root / f"call_{k:02d}.sent.json"
    if rejected.exists() and not sent.exists():
        if not args.after_rejection:
            raise SystemExit(f"{rejected} records a batch the solver refused (no call was spent). Fix the cause, "
                             f"then rerun with --after-rejection to send call {k} again; nothing is resent on its own")
        rejected.rename(root / f"call_{k:02d}.rejected.{time.strftime('%Y%m%d-%H%M%S')}.json")
    if sent.exists():
        # Started, never finished: a batch sent without R. Its points count as sent, the call as used, and it
        # is never resent (decision 28). Only the last call is handled automatically.
        sent_rec = json.loads(sent.read_text(encoding="utf-8"))
        tmp = root / f"call_{k:02d}.tmp"
        print(f"call {k} was started but has no result: recording it as sent without R (never resent)")
        nan = [None] * len(sent_rec["weights"])
        write_checkpoint(root, k, tmp, status="sent_without_R", batch=sent_rec["weights"], R=nan,
                         metas=sent_rec["meta"], info={"recovered": True},
                         outcome=Outcome("failed", stderr="[run.py] recovered after an interruption"),
                         evaluator=evaluator.name, solver_sha=solver_sha)
        st.calls = check_chain(root)
    if st.calls:
        print(f"resuming after call {len(st.calls)}")

    log = out / "online_log.jsonl"
    finite = [e["R"] for c in st.calls for e in c if math.isfinite(e["R"])]
    best = min([float(y.min())] + finite)
    last = min(CALLS, args.max_calls or CALLS)
    if not st.calls and last > 0:
        evaluator.stdin_check()
    policy = make_policy()

    for call in range(len(st.calls) + 1, last + 1):  # the course run.py loop: for i in range(1, 21)
        t0 = time.time()
        batch, metas, info = propose(policy, st, X)           # todo 1 of run.py: refit, next weight.json
        t1 = time.time()
        sent = root / f"call_{call:02d}.sent.json"
        write_json(sent, {"call": call, "weights": batch, "meta": metas})   # from here on the batch counts as sent
        tmp = root / f"call_{call:02d}.tmp"
        tmp.mkdir(exist_ok=True)
        outcome = evaluator.evaluate(batch, tmp)               # Get-Content weight.json -Raw | .\solver.exe
        t2 = time.time()
        if outcome.kind == "rejected":
            write_json(root / f"call_{call:02d}.rejected.json",
                       {"call": call, "weights": batch, "meta": metas, "returncode": outcome.returncode,
                        "stderr": outcome.stderr})
            sent.unlink()
            for p in tmp.iterdir():
                p.unlink()
            tmp.rmdir()
            print(f"call {call}: solver.exe rejected the batch before evaluating it (exit 2, no call spent): "
                  f"{outcome.stderr.strip()[:300]}\nStopped. A person decides whether to send again "
                  "(rerun with --after-rejection).")
            return 2
        R = outcome.R if outcome.kind == "ok" else [None] * len(batch)   # todo 2 of run.py: read R
        status = "complete" if outcome.kind == "ok" else "sent_without_R"
        write_checkpoint(root, call, tmp, status=status, batch=batch, R=R, metas=metas, info=info,
                         outcome=outcome, evaluator=evaluator.name, solver_sha=solver_sha)
        st.calls.append([{"W": w.reshape(3, 7), "R": float("nan") if r is None else float(r), "meta": m}
                         for w, r, m in zip(batch, R, metas)])
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(_plain({"call": call, "status": status, "n": len(batch),
                                        "batch_min": min([r for r in R if r is not None], default=None),
                                        "propose_secs": round(t1 - t0, 1), "eval_secs": round(t2 - t1, 1),
                                        "time": time.strftime("%Y-%m-%dT%H:%M:%S")})) + "\n")
        if outcome.kind == "ok":
            best = min(best, min(R))
            print(f"[{call}/{CALLS}] batch min {min(R):.6f}  best {best:.6f}  | refit+propose {t1 - t0:.0f} s, "
                  f"solver {t2 - t1:.0f} s", flush=True)
        else:
            print(f"[{call}/{CALLS}] FAILED after solver.exe started (exit {outcome.returncode}): counted as sent "
                  f"without R, never resent. {outcome.stderr.strip()[:300]}", flush=True)
    done = len(st.calls)
    print(f"{'finished' if done == CALLS else f'stopped after call {done} (--max-calls)'}: best R {best!r}; "
          f"checkpoints in {root}")
    return 0


def cmd_check(args) -> int:
    import scipy
    import sklearn
    got = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
           "scikit-learn": sklearn.__version__}
    print(f"{platform.platform()} | {platform.processor()} | {os.cpu_count()} logical CPUs (reference: 20)")
    ok = True
    for k, v in PINNED.items():
        match = got[k].startswith(v) if k == "python" else got[k] == v
        ok &= match
        print(f"  {k:13s} {got[k]:10s} pinned {v:8s} {'ok' if match else 'DIFFERENT'}")
    print("  " + blas_report())
    try:
        load_training()
        print(f"  train.csv sha256 {TRAIN_SHA256[:12]}… ok ({TRAIN_ROWS} rows)")
    except SystemExit as error:
        ok = False
        print(f"  {error}")
    for name, digest in frozen_hashes().items():
        print(f"  {name:13s} {digest}")
    exe = Path(args.solver).resolve() if args.solver else HERE / "solver.exe"
    if exe.is_file():
        digest = sha256(exe)
        print(f"  solver.exe    {exe}\n                sha256 {'= pinned' if digest == SOLVER_SHA256 else 'DIFFERS from the pinned ' + SOLVER_SHA256[:12] + '… (the run warns and continues)'}; "
              f"usage.count {'present' if (exe.parent / 'usage.count').exists() else 'absent'}")
    else:
        print(f"  solver.exe    {exe} not found")
    print("check:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_verify(args) -> int:
    """Recompute each call's batch from a recorded run's earlier results and compare point by point."""
    X, y = load_training()
    by_call: dict[int, list] = {}
    for line in Path(args.reference).open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("type") == "point":
            by_call.setdefault(r["call"], []).append(r)
    calls = args.calls or sorted(by_call)
    policy = make_policy()
    ok_all = True
    t0 = time.time()
    for c in calls:
        st = State(X, y)
        for k in range(1, c):
            st.calls.append([{"W": np.array(r["W"], float).reshape(3, 7),
                              "R": float("nan") if r["R"] is None else r["R"], "meta": r["meta"]}
                             for r in by_call[k]])
        batch, _, _ = propose(policy, st, X)
        ref = np.array([r["W"] for r in by_call[c]], float)
        same = batch.shape == ref.shape and np.array_equal(batch, ref)
        diff = float(np.max(np.abs(batch - ref))) if batch.shape == ref.shape else float("inf")
        ok_all &= same
        print(f"call {c:2d}: {'identical' if same else 'DIFFERENT'} (max |dW| {diff:.1e})", flush=True)
    print(f"verify: {'PASS' if ok_all else 'FAIL'} ({len(calls)} calls, {time.time() - t0:.0f} s)")
    return 0 if ok_all else 1


def cmd_compare(args) -> int:
    """Compare two runs' canonical records call by call; name the first call and field that differ."""
    def load(path):
        root = Path(path) / "checkpoints" if (Path(path) / "checkpoints").is_dir() else Path(path)
        out = {}
        for folder in sorted(root.glob("call_[0-9][0-9]")):
            if folder.is_dir():
                out[int(folder.name[5:])] = json.loads((folder / "canonical.json").read_text(encoding="utf-8"))
        return out
    a, b = load(args.run_a), load(args.run_b)
    for call in sorted(set(a) | set(b)):
        if call not in a or call not in b:
            print(f"call {call}: present in only one run")
            return 1
        for field in sorted(set(a[call]) | set(b[call])):
            if a[call].get(field) != b[call].get(field):
                print(f"first difference: call {call}, field {field!r}")
                return 1
    print(f"identical: {len(a)} calls")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-"):
        argv.insert(0, "run")            # `uv run run.py` alone is the graded run
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="the online run (default)")
    p.add_argument("--solver", help="solver.exe to call (default: next to run.py, the graded copy)")
    p.add_argument("--evaluator", default="solver",
                   help="solver (default, the only one for the graded run), synthetic, or module:Class (development)")
    p.add_argument("--out", help="folder for checkpoints/ and online_log.jsonl (default: this folder)")
    p.add_argument("--timeout", type=float, default=TIMEOUT, help="seconds per call before the process tree is killed")
    p.add_argument("--max-calls", type=int, help="stop after this call (testing; rerun to continue)")
    p.add_argument("--after-rejection", action="store_true",
                   help="send the next call again after the solver rejected it (a person's decision)")
    p = sub.add_parser("check", help="environment and pinned files; no call")
    p.add_argument("--solver")
    p = sub.add_parser("verify", help="recompute the batches of a recorded run; no call")
    p.add_argument("--reference", required=True, help="online_log.jsonl with one 'point' record per sent set")
    p.add_argument("--calls", type=int, nargs="*")
    p = sub.add_parser("compare", help="compare two runs' canonical records")
    p.add_argument("run_a")
    p.add_argument("run_b")
    args = ap.parse_args(argv)
    return {"run": cmd_run, "check": cmd_check, "verify": cmd_verify, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
