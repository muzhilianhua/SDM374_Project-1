"""One interface for every source of real R: the course solver, a table of measured rows, a synthetic truth.

The online loop (T08), the replay (T17) and the tests call `Evaluator.evaluate(W)` and never know which
one they talk to, so a policy that works in replay runs the same code path against `solver.exe`.

Shared behaviour, matching the solver as measured by the Windows smoke test (decisions.md, Resolved):
- a call takes 1–50 weight sets and returns one R per set, in input order;
- each building's 7 weights must be finite, non-negative and have one positive entry; the evaluator
  rescales each building to sum 1, so the stand-ins renormalise before they look up or predict;
- invalid batches are rejected before anything runs and cost no quota (`InvalidBatch`);
- after `quota` calls the next call raises `QuotaExhausted`. The solver counts every call that reaches it,
  so `SolverEvaluator` charges failed and timed-out attempts against its quota too.

Callers that deduplicate points (T08) must compare renormalised weights: W and 2·W are the same point.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np

from ..paths import REPO_ROOT
from ..simplex import DIM, N_BUILDINGS, N_METRICS, as_batch

BATCH_LIMIT = 50
QUOTA = 20
SOLVER_DIR = REPO_ROOT / "solver"
QUOTA_MESSAGE = "Usage limit reached"
# 50 sets took 289 s on A's PC (smoke test); the course computer may be slower. Killing a running call may
# still cost quota, so the limit is generous: it only guards against a hung process.
DEFAULT_TIMEOUT = 3600.0
# Deleting usage.count resets the solver's quota. Only a development machine may do that, so the switch
# needs this environment variable as well as the argument; configs can never turn it on.
RESET_USAGE_ENV = "SDM_DEV_RESET_USAGE"


class EvaluatorError(RuntimeError):
    """The evaluator ran but failed; `stderr` and `returncode` hold what the solver reported."""

    def __init__(self, message: str, *, stderr: str = "", returncode: int | None = None):
        super().__init__(message)
        self.stderr = stderr
        self.returncode = returncode


class QuotaExhausted(EvaluatorError):
    """No calls left: the solver printed its usage-limit message, or a stand-in reached its quota."""


class EvaluatorTimeout(EvaluatorError):
    """The solver did not finish within the timeout and was killed; the call may still have used quota."""


class InvalidBatch(ValueError):
    """The batch breaks the evaluator contract; it was rejected locally and used no quota."""


class PointNotInTable(KeyError):
    """A `TableEvaluator` was asked for a point that is not one of its rows."""


@dataclass(frozen=True)
class CallRecord:
    """What one successful call sent and got back, for logs and provenance."""

    call: int                      # 1-based count of successful calls made through this evaluator object
    n: int
    R: tuple[float, ...]
    seconds: float
    stderr: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def validate_batch(W, *, limit: int = BATCH_LIMIT) -> np.ndarray:
    """Return W as an (n, 21) float array or raise InvalidBatch, using the solver's own input rules."""
    try:
        W = as_batch(W)
    except ValueError as error:
        raise InvalidBatch(str(error)) from None
    if not 1 <= len(W) <= limit:
        raise InvalidBatch(f"a call takes 1 to {limit} weight sets; got {len(W)}")
    if not np.isfinite(W).all() or (W < 0).any():
        raise InvalidBatch("weights must be finite and non-negative")
    if (W.reshape(-1, N_BUILDINGS, N_METRICS).sum(axis=2) <= 0).any():
        raise InvalidBatch("each building needs at least one positive weight")
    return W


def renormalise(W: np.ndarray) -> np.ndarray:
    """Rescale each building's 7 weights to sum 1, as the solver does internally."""
    blocks = W.reshape(-1, N_BUILDINGS, N_METRICS)
    return (blocks / blocks.sum(axis=2, keepdims=True)).reshape(-1, DIM)


class Evaluator(ABC):
    """Measures R for batches of weights. Subclasses implement `_run`; `evaluate` adds the shared rules."""

    kind: str = "abstract"

    def __init__(self, *, quota: int | None = QUOTA, batch_limit: int = BATCH_LIMIT):
        self.quota = quota
        self.batch_limit = batch_limit
        self.calls = 0
        self.history: list[CallRecord] = []

    @property
    def spent(self) -> int:
        """Calls charged against the quota: successful calls here; the solver also charges failed ones."""
        return self.calls

    @property
    def remaining(self) -> int | None:
        """Calls left according to this object's own count (None when unlimited)."""
        return None if self.quota is None else max(self.quota - self.spent, 0)

    def evaluate(self, W) -> np.ndarray:
        """Return R, shape (n,), for 1 to `batch_limit` weight sets, in input order."""
        W = validate_batch(W, limit=self.batch_limit)
        self._before_call()
        start = time.perf_counter()
        R, stderr, extra = self._run(W)
        seconds = time.perf_counter() - start
        R = np.asarray(R, dtype=float)
        if R.shape != (len(W),) or not np.isfinite(R).all():
            raise EvaluatorError(f"expected {len(W)} finite R values; got {R.tolist()!r}", stderr=stderr)
        self.calls += 1
        self.history.append(CallRecord(self.calls, len(W), tuple(R.tolist()), seconds, stderr, extra))
        return R

    def _before_call(self) -> None:
        if self.remaining == 0:
            raise QuotaExhausted(f"quota of {self.quota} calls used")

    @abstractmethod
    def _run(self, W: np.ndarray) -> tuple[np.ndarray, str, dict[str, Any]]:
        """Evaluate a validated (n, 21) batch; return (R, stderr text, extra provenance)."""

    def describe(self) -> dict[str, Any]:
        """JSON-serialisable provenance for a run's meta.json."""
        return {"kind": self.kind, "quota": self.quota, "batch_limit": self.batch_limit}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pinned_sha256(checksums: Path, filename: str = "solver.exe") -> str:
    for line in checksums.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("*") == filename:
            return parts[0].lower()
    raise ValueError(f"{checksums} has no checksum for {filename}")


class SolverEvaluator(Evaluator):
    """The course `solver.exe`: JSON on stdin, `{"R": [...]}` on stdout, exit code 2 with a reason on error.

    `usage.count` (an opaque signed token next to the exe) holds the solver's own 20-call quota. Only a
    development run may set `reset_usage=True`, which deletes it before every call; the default never
    touches it, which is what the graded run must use.
    """

    kind = "solver"

    def __init__(self, exe: Path | str = SOLVER_DIR / "solver.exe", *, command: list[str] | None = None,
                 checksums: Path | str | None = SOLVER_DIR / "CHECKSUMS.sha256", reset_usage: bool = False,
                 timeout: float = DEFAULT_TIMEOUT, quota: int | None = QUOTA, batch_limit: int = BATCH_LIMIT):
        """`command` replaces `[exe]` as the process to start (tests use a Python stand-in); the working
        directory is always the exe's folder, where the solver keeps `usage.count`. `checksums=None`
        skips the sha256 check (stand-ins only). `reset_usage=True` also needs `SDM_DEV_RESET_USAGE=1`."""
        if reset_usage and os.environ.get(RESET_USAGE_ENV) != "1":
            raise EvaluatorError(f"reset_usage deletes usage.count and is for development only; "
                                 f"set {RESET_USAGE_ENV}=1 on a development machine to allow it")
        super().__init__(quota=quota, batch_limit=batch_limit)
        # Every process start, whether it succeeds or not; failures keep their reason for the run record.
        self.attempts = 0
        self.failures: list[dict[str, Any]] = []
        self.exe = Path(exe).resolve()
        self.command = list(command) if command else [str(self.exe)]
        self.reset_usage = reset_usage
        self.timeout = timeout
        self.sha256: str | None = None
        if checksums is not None:
            if not self.exe.is_file():
                raise FileNotFoundError(f"{self.exe} not found; copy solver.exe from the course solver.zip")
            expected = pinned_sha256(Path(checksums), self.exe.name)
            self.sha256 = file_sha256(self.exe)
            if self.sha256 != expected:
                raise EvaluatorError(f"{self.exe} sha256 {self.sha256} does not match the pinned {expected}")

    @property
    def usage_file(self) -> Path:
        return self.exe.parent / "usage.count"

    @property
    def spent(self) -> int:
        # The solver may count a call before it fails or is killed, so every start counts.
        return self.attempts

    def _before_call(self) -> None:
        super()._before_call()
        if self.reset_usage:
            self.usage_file.unlink(missing_ok=True)

    def _run(self, W):
        # json.dumps writes floats with repr, which round-trips exactly, so the solver sees our numbers.
        payload = json.dumps({"weights": W.tolist()})
        usage_before = self.usage_file.exists()
        self.attempts += 1
        start = time.perf_counter()
        try:
            try:
                result = subprocess.run(self.command, input=payload, capture_output=True, text=True,
                                        encoding="utf-8", errors="replace", cwd=self.exe.parent,
                                        timeout=self.timeout, check=False)
            except subprocess.TimeoutExpired as error:
                stderr = error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else error.stderr
                raise EvaluatorTimeout(f"solver did not finish within {self.timeout:g} s and was killed",
                                       stderr=stderr or "") from None
            stderr = result.stderr.strip()
            if result.returncode != 0:
                error_type = QuotaExhausted if QUOTA_MESSAGE in stderr else EvaluatorError
                raise error_type(f"solver exited with code {result.returncode}: {stderr or '(no stderr)'}",
                                 stderr=stderr, returncode=result.returncode)
            R = parse_solver_output(result.stdout, stderr)
            # Checked here as well as in evaluate(), so this charged attempt is recorded as a failure.
            if len(R) != len(W):
                raise EvaluatorError(f"expected {len(W)} finite R values; got {R!r}", stderr=stderr)
        except EvaluatorError as error:
            self.failures.append({"attempt": self.attempts, "n": len(W), "error": type(error).__name__,
                                  "message": str(error), "returncode": error.returncode,
                                  "seconds": time.perf_counter() - start})
            raise
        return R, stderr, {"attempt": self.attempts, "usage_file_before": usage_before,
                           "usage_file_after": self.usage_file.exists()}

    def describe(self):
        return {**super().describe(), "exe": str(self.exe), "sha256": self.sha256,
                "reset_usage": self.reset_usage, "timeout": self.timeout, "attempts": self.attempts,
                "failures": self.failures}


def parse_solver_output(stdout: str, stderr: str = "") -> list[float]:
    """R values from the solver's stdout; tolerates log lines before the final JSON line."""
    lines = [line for line in stdout.strip().splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            data = json.loads(line)
        except ValueError:
            continue
        if isinstance(data, dict) and isinstance(data.get("R"), list):
            values = data["R"]
            if all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                   for value in values):
                return [float(value) for value in values]
    raise EvaluatorError(f"no {{\"R\": [...]}} line in solver output: {stdout[-500:]!r}", stderr=stderr)


class TableEvaluator(Evaluator):
    """Looks up measured R for points that are rows of a table (replay over a held-out pool, tests).

    A point matches a row when their renormalised weights differ by at most `atol` in every coordinate;
    any other point raises `PointNotInTable`, so a replay policy must choose from the pool.
    """

    kind = "table"

    def __init__(self, X, y, *, ids=None, atol: float = 1e-9, quota: int | None = QUOTA,
                 batch_limit: int = BATCH_LIMIT):
        super().__init__(quota=quota, batch_limit=batch_limit)
        if len(as_batch(X)) == 0:
            raise ValueError("a table needs at least one row")
        self.X = renormalise(validate_batch(X, limit=len(as_batch(X))))
        self.y = np.asarray(y, dtype=float)
        if self.y.shape != (len(self.X),):
            raise ValueError(f"y has shape {self.y.shape}; expected ({len(self.X)},)")
        self.ids = tuple(ids) if ids is not None else tuple(str(index) for index in range(len(self.X)))
        if len(self.ids) != len(self.X):
            raise ValueError(f"got {len(self.ids)} ids for {len(self.X)} rows")
        self.atol = atol

    @classmethod
    def from_dataset(cls, dataset, rows=None, **kwargs) -> TableEvaluator:
        """A table over all rows of a `sdm.data.Dataset`, or the row positions in `rows`."""
        rows = np.arange(len(dataset.y)) if rows is None else np.asarray(rows)
        return cls(dataset.X[rows], dataset.y[rows], ids=[dataset.ids[row] for row in rows], **kwargs)

    def locate(self, W) -> np.ndarray:
        """Row position of each point; raises PointNotInTable for the first point not in the table."""
        points = renormalise(validate_batch(W, limit=len(as_batch(W))))
        positions = []
        for index, point in enumerate(points):
            gaps = np.abs(self.X - point).max(axis=1)
            nearest = int(np.argmin(gaps))
            if gaps[nearest] > self.atol:
                raise PointNotInTable(f"point {index} is not a table row (nearest gap {gaps[nearest]:.3g})")
            positions.append(nearest)
        return np.asarray(positions)

    def _run(self, W):
        positions = self.locate(W)
        return self.y[positions], "", {"ids": [self.ids[position] for position in positions]}

    def describe(self):
        return {**super().describe(), "rows": len(self.y), "atol": self.atol}


class SyntheticEvaluator(Evaluator):
    """A fitted surrogate's mean plays the true function (semi-synthetic replay, as in analysis/coupling)."""

    kind = "synthetic"

    def __init__(self, truth, *, name: str = "", quota: int | None = QUOTA, batch_limit: int = BATCH_LIMIT):
        super().__init__(quota=quota, batch_limit=batch_limit)
        self.truth = truth
        self.name = name or getattr(truth, "name", type(truth).__name__)

    @classmethod
    def fit(cls, model_name: str, params: dict | None, X, y, **kwargs) -> SyntheticEvaluator:
        """Build the registered model `model_name` with `params`, fit it on (X, y) and use it as the truth."""
        from ..registry import build_model
        return cls(build_model(model_name, params).fit(as_batch(X), np.asarray(y, dtype=float)),
                   name=model_name, **kwargs)

    def _run(self, W):
        mean, _ = self.truth.predict(renormalise(W))
        return np.asarray(mean, dtype=float), "", {}

    def describe(self):
        return {**super().describe(), "truth": self.name,
                "truth_params": repr(getattr(self.truth, "params", None))}


def build_evaluator(kind: str, params: dict | None = None, *, dataset=None) -> Evaluator:
    """Factory for configs: kind "solver" (params go to SolverEvaluator), "table" (needs `dataset`; optional
    `rows`) or "synthetic" (needs `dataset`; params `model` and optional `model_params`)."""
    params = dict(params or {})
    if kind == "solver":
        if "reset_usage" in params:
            raise ValueError("reset_usage cannot come from a config: a frozen config runs on the course computer")
        return SolverEvaluator(**params)
    if kind in ("table", "synthetic") and dataset is None:
        raise ValueError(f"evaluator kind {kind!r} needs the dataset")
    if kind == "table":
        return TableEvaluator.from_dataset(dataset, **params)
    if kind == "synthetic":
        if "model" not in params:
            raise ValueError("evaluator kind 'synthetic' needs params.model (a registered model name)")
        model = params.pop("model")
        return SyntheticEvaluator.fit(model, params.pop("model_params", None), dataset.X, dataset.y, **params)
    raise ValueError(f"unknown evaluator kind {kind!r}; expected 'solver', 'table' or 'synthetic'")
