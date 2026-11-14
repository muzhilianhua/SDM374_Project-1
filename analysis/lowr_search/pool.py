"""Parallel, cached, append-only access to the course evaluator (research use only).

Exploration harness: it resets ``usage.count`` inside its *own* worker folders
(hard links of ``solver.exe``), never in ``solver/`` of the repository, and it
is not part of the graded online policy (decisions 19, 26).

Every evaluation is appended to ``eval_log.jsonl``; a point that was already
evaluated is answered from the log (the evaluator is deterministic, decision 20).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXE_SRC = HERE.parents[1] / "solver" / "solver.exe"
EXE_SHA256 = "293867b81dbf0c6194f97a57115487b9a39239d0c7dadbd51d749a203a077247"
LOG = HERE / "eval_log.jsonl"
MAX_SETS = 50
CHUNK = 12          # sets per solver call when a batch is split: a slow or failed call then loses little work


def normalize(W) -> np.ndarray:
    """Return weights as (n, 21), each building's 7 weights summing to 1."""
    W = np.asarray(W, dtype=float).reshape(-1, 3, 7)
    if not np.all(np.isfinite(W)) or (W < 0).any() or (W.sum(axis=2) <= 0).any():
        raise ValueError("weights must be finite, non-negative, one positive per building")
    return (W / W.sum(axis=2, keepdims=True)).reshape(-1, 21)


def key_of(w: np.ndarray) -> str:
    return ",".join(repr(float(v)) for v in w)


def check_exe(path: Path) -> None:
    """Refuse to run an evaluator other than the pinned course release (solver/CHECKSUMS.sha256)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    if h.hexdigest() != EXE_SHA256:
        raise RuntimeError(f"{path}: sha256 {h.hexdigest()} is not the pinned solver.exe {EXE_SHA256}")


class SolverPool:
    def __init__(self, n_workers: int = 8, root: Path | None = None, log: Path = LOG):
        self.root = Path(root or HERE / "workers")
        self.log = Path(log)
        self.n_workers = n_workers
        self._lock = threading.Lock()
        self.cache: dict[str, float] = {}
        self.n_logged = 0
        self.n_solver_evals = 0
        if self.log.exists():
            with self.log.open(encoding="utf-8") as fh:
                for line in fh:
                    rec = json.loads(line)
                    self.cache[key_of(np.asarray(rec["weights"], dtype=float))] = rec["R"]
                    self.n_logged += 1
        if os.name == "nt":      # keep our solver processes off E-core-only background scheduling, see qos.py
            import qos
            seen: set = set()
            qos.unthrottle(os.getpid())

            def watch():
                while True:
                    try:
                        qos.sweep(os.getpid(), seen)
                    except Exception:
                        pass
                    time.sleep(5)

            threading.Thread(target=watch, daemon=True).start()
        check_exe(EXE_SRC)
        for i in range(n_workers):
            d = self.root / f"w{i:02d}"
            (d / "tmp").mkdir(parents=True, exist_ok=True)
            exe = d / "solver.exe"
            if exe.exists() and not os.path.samefile(exe, EXE_SRC):
                check_exe(exe)        # a copy (no hard link possible) or a link to an older exe
            if not exe.exists():
                try:
                    os.link(EXE_SRC, exe)
                except OSError:
                    shutil.copy2(EXE_SRC, exe)

    # ------------------------------------------------------------------ solver
    def _call(self, worker: int, W: np.ndarray) -> list[float]:
        """One solver.exe invocation (<= 50 sets) in the worker's own folder."""
        d = self.root / f"w{worker:02d}"
        tmp = d / "tmp"
        usage = d / "usage.count"
        if usage.exists():          # research harness: quota reset is local to this folder
            usage.unlink()
        env = dict(os.environ, TEMP=str(tmp), TMP=str(tmp))
        payload = json.dumps({"weights": [w.reshape(3, 7).tolist() for w in W]})
        try:
            proc = subprocess.run(
                [str(d / "solver.exe")], input=payload, capture_output=True, text=True,
                cwd=str(d), env=env, timeout=300 + 120 * len(W),   # normal speed is about 15 s per set
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        finally:
            for p in tmp.iterdir():   # leftovers of a killed run
                try:
                    shutil.rmtree(p) if p.is_dir() else p.unlink()
                except OSError:
                    pass
        if proc.returncode != 0:
            raise RuntimeError(f"solver exit {proc.returncode}: {proc.stderr.strip()[:500]}")
        R = json.loads(proc.stdout)["R"]
        if len(R) != len(W):
            raise RuntimeError(f"solver returned {len(R)} values for {len(W)} sets")
        return [float(r) for r in R]

    def call_uncached(self, W, worker: int = 0) -> list[float]:
        """One fresh solver call on normalised W (<= 50 sets): no cache, no retry, nothing logged.
        For re-measuring points that are already in the log (verify.py)."""
        return self._call(worker, normalize(W))

    def _call_retry(self, worker: int, W: np.ndarray) -> list[float]:
        try:
            return self._call(worker, W)
        except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError, KeyError) as exc:
            if len(W) == 1:
                print(f"  [pool] worker {worker}: point failed ({exc!r})", flush=True)
                return [float("nan")]
            mid = len(W) // 2          # isolate a failing point by bisection
            return self._call_retry(worker, W[:mid]) + self._call_retry(worker, W[mid:])

    # ---------------------------------------------------------------- public
    def evaluate(self, W, tag: str = "") -> np.ndarray:
        """R for every row of W (n, 21) or (n, 3, 7); cached points cost nothing."""
        W = normalize(W)
        keys = [key_of(w) for w in W]
        todo: dict[str, np.ndarray] = {}
        for k, w in zip(keys, W):
            if k not in self.cache and k not in todo:
                todo[k] = w
        if todo and os.environ.get("LOWR_CACHE_ONLY"):
            raise RuntimeError(f"LOWR_CACHE_ONLY is set but {len(todo)} point(s) of '{tag}' are not in {self.log.name}")
        if todo:
            new_keys = list(todo)
            new_W = np.array([todo[k] for k in new_keys])
            n = len(new_W)
            size = min(MAX_SETS, CHUNK, max(1, -(-n // self.n_workers)))
            chunks = [(s, min(n, s + size)) for s in range(0, n, size)]
            free = list(range(self.n_workers))
            free_lock = threading.Lock()

            def run(span):
                with free_lock:
                    worker = free.pop()
                t0 = time.time()
                try:
                    R = self._call_retry(worker, new_W[span[0]:span[1]])
                finally:
                    with free_lock:
                        free.append(worker)
                secs = time.time() - t0
                with self._lock, self.log.open("a", encoding="utf-8") as fh:
                    for j, r in zip(range(span[0], span[1]), R):
                        if r != r:      # failed point: not cached, not logged
                            continue
                        self.cache[new_keys[j]] = r
                        fh.write(json.dumps({
                            "eval_id": self.n_logged, "tag": tag, "weights": new_W[j].tolist(),
                            "R": r, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "worker": worker, "call_secs": round(secs, 1), "call_sets": span[1] - span[0],
                        }) + "\n")
                        self.n_logged += 1
                        self.n_solver_evals += 1

            with ThreadPoolExecutor(self.n_workers) as ex:
                list(ex.map(run, chunks))
        return np.array([self.cache.get(k, float("nan")) for k in keys])


def load_log(path: Path = LOG):
    """(W (n, 21), R (n,), tags) of everything evaluated so far."""
    W, R, tags = [], [], []
    if Path(path).exists():
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                W.append(rec["weights"]); R.append(rec["R"]); tags.append(rec.get("tag", ""))
    return np.array(W, dtype=float).reshape(-1, 21), np.array(R, dtype=float), tags
