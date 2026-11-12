"""The online main loop (T08): propose -> screen -> call the evaluator -> log -> refit, up to the call budget.

One run folder (`runs/<id>-online-<config>/`) holds the whole session and is the only state:

| File | Content |
|---|---|
| `config.toml`, `meta.json` | the config used and its provenance (`sdm.runs.create_run`) |
| `online_log.csv` | every successful call (`sdm.online.OnlineLog`, T07): the measured R values |
| `attempts.jsonl` | journal of evaluator attempts: a `started` line before each launch, then `ok`, `failed` or `lost` |
| `calls/call_NN.json` | per call: candidates, screening, the batch sent, predictions, attempts, R |
| `predictions.csv` | `call,point,predicted_R` for logged points (the T26 sidecar) |
| `sessions.jsonl` | one line per invocation: git state, evaluator provenance, options |
| `metrics.json` | progress summary, rewritten after every call |

The call budget counts attempts, not successful calls: the solver may charge a call that fails or is killed,
so every `started` line uses one of the calls. A failed batch is never resent (decision 28): the solver had
started, so its points count as sent (decision 20) even though no R came back. They go to
`OnlineState.X_unmeasured`, the pre-submit check excludes them, and the next attempt sends a new batch.
A `started` line without an outcome means the process died or was interrupted (Ctrl-C) during a call. If the
call record already holds R, resume logs it; otherwise resuming needs an explicit `accept_lost_attempt`,
which records it as `lost` (the call may or may not have been charged) and treats its points the same way.
An attempt the evaluator rejected before running (`InvalidBatch`, `PointNotInTable`) is closed as
`rejected` and not charged. An unfinished last journal line is cut and kept in `attempts.jsonl.torn`. Each `started` line carries its batch, so the journal alone says what was sent.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from ..config import Config, load_config
from ..data import Dataset, load_dataset
from ..interfaces import BatchPolicy, OnlineState
from ..registry import build_policy
from ..runs import Run, create_run, git_state, resolve_run
from ..simplex import DIM, N_BUILDINGS, N_METRICS
from .evaluator import (
    Evaluator,
    EvaluatorError,
    EvaluatorTimeout,
    InvalidBatch,
    PointNotInTable,
    QuotaExhausted,
    SolverEvaluator,
    build_evaluator,
)
from .log import OnlineLog
from .screen import screen

LOG_NAME = "online_log.csv"
JOURNAL_NAME = "attempts.jsonl"
DEFAULT_POLICY = "surrogate"


class OnlineRunError(RuntimeError):
    """The run cannot continue safely without a decision from the operator."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_default(value):
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot write {type(value).__name__} to JSON")


def _write_json_atomic(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_json_default) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def repair_torn_tail(path: Path) -> str | None:
    """Cut an unfinished last line (the process died while writing it) and keep it in `<name>.torn`.

    Every journal line is written and fsynced before the step it announces: a torn `started` line means the
    solver was never launched, and a torn outcome line leaves its attempt pending, which the recovery and
    `--accept-lost-attempt` paths handle. Only the last line may be torn; any other bad line still raises.
    """
    if not path.exists():
        return None
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return None
    cut = data.rfind(b"\n") + 1
    fragment = data[cut:]
    with open(path.with_name(path.name + ".torn"), "ab") as stream:
        stream.write(fragment + b"\n")
    with open(path, "r+b") as stream:
        stream.truncate(cut)
        stream.flush()
        os.fsync(stream.fileno())
    return fragment.decode("utf-8", errors="replace")


def batch_digest(W: np.ndarray) -> str:
    """sha256 of the batch as sent (repr floats), to tie journal lines to calls/call_NN.json."""
    return hashlib.sha256(json.dumps(np.asarray(W).tolist()).encode()).hexdigest()


@dataclass(frozen=True)
class Journal:
    """The attempts journal as read from disk."""

    records: tuple[dict, ...]

    @property
    def started(self) -> list[dict]:
        return [record for record in self.records if record["event"] == "started"]

    @property
    def pending(self) -> list[dict]:
        closed = {record["attempt"] for record in self.records if record["event"] != "started"}
        return [record for record in self.started if record["attempt"] not in closed]

    @property
    def used(self) -> int:
        """Attempts charged against the budget: every start except those rejected before the solver ran."""
        rejected = {record["attempt"] for record in self.records if record["event"] == "rejected"}
        return sum(record["attempt"] not in rejected for record in self.started)

    @property
    def last_outcome(self) -> str | None:
        """Event of the latest closed attempt (`ok`, `failed`, `lost`, `rejected`), or None."""
        outcomes = [record["event"] for record in self.records if record["event"] != "started"]
        return outcomes[-1] if outcomes else None

    def unmeasured(self) -> np.ndarray:
        """Points of attempts that ended `failed` or `lost`: sent, never measured, never to be sent again."""
        ended = {record["attempt"] for record in self.records if record["event"] in ("failed", "lost")}
        batches = []
        for record in self.started:
            if record["attempt"] not in ended:
                continue
            if "weights" not in record:
                raise OnlineRunError(f"attempt {record['attempt']} has no batch in the journal, so its points "
                                     "cannot be excluded from later calls")
            batches.append(np.asarray(record["weights"], dtype=float).reshape(-1, DIM))
        return np.vstack(batches) if batches else np.empty((0, DIM))


class OnlineSession:
    """One run folder and the objects that act on it; `run()` advances it call by call."""

    def __init__(self, run: Run, config: Config, dataset: Dataset, *, policy: BatchPolicy,
                 evaluator_factory: Callable[[int], Evaluator], echo: Callable[[str], None] = print):
        self.run_folder = run
        self.config = config
        self.settings = config.online
        self.dataset = dataset
        self.policy = policy
        self.evaluator_factory = evaluator_factory
        self.echo = echo
        self.log = OnlineLog(run.path / LOG_NAME)
        self.journal_path = run.path / JOURNAL_NAME
        self.evaluator: Evaluator | None = None

    # ----- state on disk -------------------------------------------------------------------------------
    def journal(self) -> Journal:
        return Journal(tuple(_read_jsonl(self.journal_path)))

    def state(self) -> OnlineState:
        rows = self.log.rows()
        X_online = np.array([row.weights for row in rows], dtype=float).reshape(-1, DIM)
        y_online = np.array([row.R for row in rows], dtype=float)
        calls = np.array([row.call for row in rows], dtype=int)
        journal = self.journal()
        return OnlineState(call=(int(calls.max()) + 1 if len(calls) else 1), calls_total=self.settings.calls,
                           calls_left=self.settings.calls - journal.used, batch_size=self.settings.batch_size,
                           seed=self.settings.seed, X_train=self.dataset.X, y_train=self.dataset.y,
                           X_online=X_online, y_online=y_online, online_call=calls,
                           X_unmeasured=journal.unmeasured())

    def call_path(self, call: int) -> Path:
        return self.run_folder.path / "calls" / f"call_{call:02d}.json"

    def recover_measured(self) -> int:
        """Finish calls whose R values were saved in calls/call_NN.json but not yet logged (crash in between)."""
        recovered = 0
        for record in self.journal().pending:
            path = self.call_path(record["call"])
            if not path.exists():
                continue
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("status") != "measured" or saved.get("batch_sha256") != record["batch_sha256"]:
                continue
            if self.log.next_call == record["call"]:
                weights = np.asarray(saved["weights"], dtype=float).reshape(-1, DIM)
                self.log.append(weights, saved["R"], source=saved["policy"], run_id=self.run_folder.id,
                                call=record["call"])
            _append_jsonl(self.journal_path, {"attempt": record["attempt"], "call": record["call"], "event": "ok",
                                              "time": _now(), "recovered": True})
            recovered += 1
        return recovered

    def accept_lost_attempts(self) -> int:
        """Close every pending attempt as `lost`: it stays charged and its points count as sent (decision 28)."""
        pending = self.journal().pending
        for record in pending:
            if "weights" not in record:
                raise OnlineRunError(f"attempt {record['attempt']} has no batch in the journal, so its points "
                                     "cannot be excluded from later calls")
            _append_jsonl(self.journal_path, {"attempt": record["attempt"], "call": record["call"], "event": "lost",
                                              "time": _now(), "message": "process ended during the call; "
                                              "outcome unknown, accepted by the operator"})
            path = self.call_path(record["call"])
            if path.exists():
                saved = json.loads(path.read_text(encoding="utf-8"))
                if saved.get("batch_sha256") == record["batch_sha256"]:
                    saved.update(status="lost", attempt=record["attempt"])
                    _write_json_atomic(self.failed_attempt_path(record["call"], record["attempt"]), saved)
                    path.unlink()
        return len(pending)

    # ----- one call ------------------------------------------------------------------------------------
    def propose(self, state: OnlineState) -> tuple[np.ndarray, dict]:
        """Ask the policy for candidates, screen them, and return the batch to send with its record."""
        settings = self.settings
        n = settings.batch_size
        requested = n + settings.candidate_margin
        sent = state.X_sent
        rounds, rejections, diagnostics = [], [], {}
        # Points accepted so far, each with the prediction and tag from the round that proposed it. Rounds
        # accumulate: a policy's answer for a larger n need not contain its answer for a smaller one.
        rows: list[np.ndarray] = []
        means: list[float | None] = []
        stds: list[float | None] = []
        tags: list[str | None] = []
        for round_number in range(1, settings.max_rounds + 1):
            start = time.perf_counter()
            proposal = self.policy.propose(state, requested)
            seconds = time.perf_counter() - start
            weights = np.asarray(proposal.weights, dtype=float).reshape(-1, DIM)
            result = screen(weights, train=state.X_train, sent=sent, n=n - len(rows), atol=settings.dedupe_atol,
                            accepted=np.asarray(rows).reshape(-1, DIM))
            for index in result.accepted:
                rows.append(weights[index])
                means.append(_pick(proposal.predicted_mean, index))
                stds.append(_pick(proposal.predicted_std, index))
                tags.append(proposal.tags[index] if proposal.tags is not None else None)
            rejections += [{**item, "round": round_number} for item in result.rejected]
            if round_number == 1:
                diagnostics = _jsonable(proposal.diagnostics)
            rounds.append({"round": round_number, "requested": requested, "returned": len(weights),
                           "accepted": len(result.accepted), "rejected": result.counts(), "seconds": seconds})
            if len(rows) >= n:
                break
            requested *= 2
        batch = np.asarray(rows, dtype=float).reshape(-1, DIM)
        record = {
            "call": state.call, "status": "proposed", "policy": self.policy.name, "time": _now(),
            "calls_left_before": state.calls_left, "observations": len(state.y),
            "rounds": rounds, "rejections": rejections,
            "weights": batch.reshape(-1, N_BUILDINGS, N_METRICS).tolist(),
            # A prediction column is kept only when every point in the batch has one.
            "predicted_mean": means if rows and None not in means else None,
            "predicted_std": stds if rows and None not in stds else None,
            "tags": tags if rows and None not in tags else None,
            "diagnostics": diagnostics, "batch_sha256": batch_digest(batch) if len(batch) else None,
            "attempts": [],
        }
        return batch, record

    def send(self, call: int, batch: np.ndarray, record: dict) -> np.ndarray | None:
        """Make one evaluator attempt with this batch; returns R, or None when it failed.

        A failed batch is never resent (decision 28): its points stay in the journal and count as sent.
        """
        journal = self.journal()
        if journal.used >= self.settings.calls:
            record["stop_reason"] = "budget_used"
            return None
        attempt = journal.used + 1
        record["attempt"] = attempt
        _append_jsonl(self.journal_path, {"attempt": attempt, "call": call, "event": "started", "time": _now(),
                                          "n": len(batch), "batch_sha256": record["batch_sha256"],
                                          "weights": batch.tolist()})
        start = time.perf_counter()
        try:
            R = self.evaluator.evaluate(batch)
        except (EvaluatorError, EvaluatorTimeout) as error:
            outcome = {"attempt": attempt, "call": call, "event": "failed", "time": _now(),
                       "seconds": time.perf_counter() - start, "error": type(error).__name__,
                       "message": str(error), "stderr": error.stderr[-2000:], "returncode": error.returncode}
            _append_jsonl(self.journal_path, outcome)
            record["attempts"].append(outcome)
            self.echo(f"call {call}: attempt {attempt} failed: {error}; its {len(batch)} points count as sent")
            record["stop_reason"] = "quota_exhausted" if isinstance(error, QuotaExhausted) else None
            return None
        except (InvalidBatch, PointNotInTable) as error:
            # Raised before anything was evaluated: not charged, and the points were not sent.
            _append_jsonl(self.journal_path, {"attempt": attempt, "call": call, "event": "rejected",
                                              "time": _now(), "error": type(error).__name__, "message": str(error)})
            raise
        # Anything else (Ctrl-C, SystemExit, a bug) may arrive after the solver used the call and computed R.
        # The attempt stays `started` with no outcome, so resume must recover it or have the operator accept
        # it as lost; the evaluator kills the solver's process tree on the way out.
        outcome = {"attempt": attempt, "call": call, "event": "ok", "time": _now(),
                   "seconds": time.perf_counter() - start}
        # The R values reach disk in the call record first, so a crash before the log append loses
        # nothing: recover_measured() finishes the bookkeeping on resume.
        record["attempts"].append(outcome)
        record.update(status="measured", R=R.tolist(), min_R=float(R.min()))
        _write_json_atomic(self.call_path(call), record)
        self.log.append(batch, R, source=self.policy.name, run_id=self.run_folder.id, call=call)
        _append_jsonl(self.journal_path, outcome)
        return R

    def failed_attempt_path(self, call: int, attempt: int) -> Path:
        return self.run_folder.path / "calls" / f"call_{call:02d}_attempt_{attempt:02d}_failed.json"

    # ----- the loop ------------------------------------------------------------------------------------
    def run(self, *, max_calls: int | None = None, dry_run: bool = False) -> dict:
        """Make calls until the budget is used, `max_calls` calls were made, or a call cannot be made."""
        torn = repair_torn_tail(self.journal_path)
        if torn is not None:
            self.echo(f"cut an unfinished last journal line (kept in {JOURNAL_NAME}.torn): {torn[:120]!r}")
        if self.recover_measured():
            self.echo("recovered measured R values that were saved but not yet logged")
            self.write_outputs(stop_reason=None)
        if self.journal().pending:
            raise OnlineRunError(
                f"{self.run_folder.id}: attempt {self.journal().pending[0]['attempt']} started but has no outcome "
                "(the process ended during a call). Its R values are not in the log and the solver may have "
                "charged it. Resume with --accept-lost-attempt to count it as used and propose again.")
        stop_reason = None
        made = 0
        failures_in_a_row = 0
        while stop_reason is None:
            state = self.state()
            if state.call > self.settings.calls or state.calls_left <= 0:
                # Running out right after a failed or lost attempt is a failure, not a normal finish.
                last = self.journal().last_outcome
                stop_reason = "evaluator_failed" if last in ("failed", "lost") else "budget_used"
                break
            if max_calls is not None and made >= max_calls:
                stop_reason = "max_calls"
                break
            batch, record = self.propose(state)
            path = self.call_path(state.call)
            if len(batch) == 0:
                record.update(status="not_sent", stop_reason="no_valid_candidates")
                _write_json_atomic(path, record)
                stop_reason = "no_valid_candidates"
                break
            if dry_run:
                record["status"] = "dry_run"
                _write_json_atomic(path.with_name(path.stem + "_dry_run.json"), record)
                self.echo(f"call {state.call} (dry run): would send {len(batch)} points, "
                          f"predicted min {_fmt(record['predicted_mean'])}, batch sha256 {record['batch_sha256'][:12]}")
                stop_reason = "dry_run"
                break
            _write_json_atomic(path, record)
            if self.evaluator is None:
                self.evaluator = self.evaluator_factory(state.calls_left)
            R = self.send(state.call, batch, record)
            if R is None:
                if record.get("stop_reason") == "budget_used":
                    last = self.journal().last_outcome
                    stop_reason = "evaluator_failed" if last in ("failed", "lost") else "budget_used"
                    break
                # Keep the failed batch under its attempt number; the next proposal for this call number
                # writes call_NN.json afresh, with the failed points screened out as already sent.
                record["status"] = "failed"
                _write_json_atomic(self.failed_attempt_path(state.call, record["attempt"]), record)
                path.unlink(missing_ok=True)
                failures_in_a_row += 1
                if record["stop_reason"] is not None:
                    stop_reason = record["stop_reason"]
                elif failures_in_a_row >= self.settings.max_failures:
                    stop_reason = "evaluator_failed"
                continue
            failures_in_a_row = 0
            made += 1
            self.write_outputs(stop_reason=None)
            summary = self.summary()
            self.echo(f"call {state.call}/{self.settings.calls}: sent {len(batch)} "
                      f"(screened out {len(record['rejections'])}), min R {R.min():.6f}, "
                      f"best {summary['best_R']:.6f} (call {summary['best_call']}), "
                      f"{record['attempts'][-1]['seconds']:.1f} s")
        self.write_outputs(stop_reason=stop_reason)
        return self.summary()

    # ----- outputs -------------------------------------------------------------------------------------
    def summary(self) -> dict:
        rows = self.log.rows()
        journal = self.journal()
        metrics: dict[str, Any] = {"calls_completed": len({row.call for row in rows}),
                                   "attempts": journal.used, "calls_left": self.settings.calls - journal.used,
                                   "points_evaluated": len(rows), "training_best_R": float(self.dataset.y.min())}
        if rows:
            best = min(rows, key=lambda row: row.R)
            metrics.update(best_R=best.R, best_call=best.call, best_point=best.point,
                           best_weights=np.reshape(best.weights, (N_BUILDINGS, N_METRICS)).tolist(),
                           call_min_R=[min(row.R for row in rows if row.call == call)
                                       for call in sorted({row.call for row in rows})])
        else:
            metrics.update(best_R=None, best_call=None)
        return metrics

    def write_outputs(self, *, stop_reason: str | None) -> None:
        summary = self.summary()
        summary["stop_reason"] = stop_reason
        summary["finished"] = summary["calls_left"] <= 0 or summary["calls_completed"] >= self.settings.calls
        summary["predictions_complete"] = self._write_predictions()
        _write_json_atomic(self.run_folder.path / "metrics.json", summary)

    def _write_predictions(self) -> bool:
        """Write `call,point,predicted_R` (T26 format) when every logged point has a prediction.

        T26 needs exact key coverage of the log, so a partial file is never written: if any logged call
        lacks predictions, an existing `predictions.csv` is removed and this returns False.
        """
        path = self.run_folder.path / "predictions.csv"
        rows = self.log.rows()
        lines = []
        for call in sorted({row.call for row in rows}):
            record = json.loads(self.call_path(call).read_text(encoding="utf-8"))
            predicted = record.get("predicted_mean")
            logged = sum(row.call == call for row in rows)
            if predicted is None or len(predicted) != logged:
                path.unlink(missing_ok=True)
                return False
            lines += [f"{call},{point},{value:.17g}" for point, value in enumerate(predicted, 1)]
        if not lines:
            path.unlink(missing_ok=True)
            return False
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text("call,point,predicted_R\n" + "".join(line + "\n" for line in lines), encoding="utf-8")
        os.replace(temporary, path)
        return True


def _pick(values, index: int) -> float | None:
    return None if values is None else float(np.asarray(values, dtype=float)[index])


def _jsonable(data: dict) -> dict:
    out = {}
    for key, value in (data or {}).items():
        if isinstance(value, (np.floating, np.integer, np.bool_)):
            value = value.item()
        out[str(key)] = value if isinstance(value, (int, float, str, bool, type(None), list)) else repr(value)
    return out


def _fmt(values) -> str:
    return f"{min(values):.4f}" if values else "n/a"


# ----- entry points used by the CLI ----------------------------------------------------------------------
def _policy_from(config: Config) -> BatchPolicy:
    spec = config.policy
    name, params = (spec.name, spec.params) if spec is not None else (DEFAULT_POLICY, {})
    return build_policy(name, params, model=config.model, optimizer=config.optimizer)


def _evaluator_factory(config: Config, dataset: Dataset, *, dev_reset_usage: bool) -> Callable[[int], Evaluator]:
    if config.evaluator is None:
        raise ValueError("an online config needs an [evaluator] section (kind = \"solver\", \"table\" or \"synthetic\")")
    spec = config.evaluator

    def factory(calls_left: int) -> Evaluator:
        if spec.kind == "solver" and dev_reset_usage:
            return SolverEvaluator(**spec.params, reset_usage=True, quota=calls_left)
        return build_evaluator(spec.kind, {**spec.params, "quota": calls_left}, dataset=dataset)
    return factory


def _record_session(run: Run, *, resumed: bool, options: dict, evaluator_kind: str) -> None:
    _append_jsonl(run.path / "sessions.jsonl", {"time": _now(), "resumed": resumed, "git": git_state(),
                                                "evaluator": evaluator_kind, "options": options})


def start(config_path, *, runs_root=None, name: str | None = None, max_calls: int | None = None,
          dry_run: bool = False, dev_reset_usage: bool = False, echo=print) -> tuple[Run, dict]:
    """Create a run folder from an online config and run it."""
    config_path = Path(config_path)
    config = load_config(config_path)
    dataset = load_dataset(config.dataset)
    policy = _policy_from(config)                       # build before the folder, so typos leave nothing
    factory = _evaluator_factory(config, dataset, dev_reset_usage=dev_reset_usage)
    options = {"max_calls": max_calls, "dry_run": dry_run, "dev_reset_usage": dev_reset_usage}
    run = create_run("online", config_path, name=name or config_path.stem, options=options,
                     dataset=dataset, root=runs_root)
    _record_session(run, resumed=False, options=options, evaluator_kind=config.evaluator.kind)
    session = OnlineSession(run, config, dataset, policy=policy, evaluator_factory=factory, echo=echo)
    return run, _finish(session, max_calls=max_calls, dry_run=dry_run)


def resume(reference: str, *, runs_root=None, max_calls: int | None = None, dry_run: bool = False,
           dev_reset_usage: bool = False, accept_lost_attempt: bool = False, allow_code_change: bool = False,
           echo=print) -> tuple[Run, dict]:
    """Continue an online run from its folder, with the config saved there."""
    run = resolve_run(reference, runs_root)
    meta = run.read_json("meta.json")
    if meta["kind"] != "online":
        raise ValueError(f"{run.id} is a {meta['kind']} run, not an online run")
    config = load_config(run.config_path)
    dataset = load_dataset(config.dataset)
    if dataset.sha256 != meta["dataset"]["sha256"]:
        raise OnlineRunError(f"{run.id} was started on dataset {meta['dataset']['sha256'][:12]}, "
                             f"not {dataset.sha256[:12]}")
    now, then = git_state(), meta["git"]
    if not allow_code_change and (now["commit"] != then["commit"] or now["dirty"] or then["dirty"]):
        raise OnlineRunError(
            f"{run.id} was started at commit {then['commit']} (dirty={then['dirty']}); the code is now at "
            f"{now['commit']} (dirty={now['dirty']}). Resuming with other code may change the remaining batches "
            "(decision 22). Check out the original commit, or pass --allow-code-change.")
    policy = _policy_from(config)
    factory = _evaluator_factory(config, dataset, dev_reset_usage=dev_reset_usage)
    session = OnlineSession(run, config, dataset, policy=policy, evaluator_factory=factory, echo=echo)
    if accept_lost_attempt:
        # Recover first: an attempt whose R values were saved is measured, not lost.
        torn = repair_torn_tail(session.journal_path)
        if torn is not None:
            echo(f"cut an unfinished last journal line (kept in {JOURNAL_NAME}.torn): {torn[:120]!r}")
        if session.recover_measured():
            echo("recovered measured R values that were saved but not yet logged")
        lost = session.accept_lost_attempts()
        if lost:
            echo(f"recorded {lost} lost attempt(s); they stay counted against the budget")
    options = {"max_calls": max_calls, "dry_run": dry_run, "dev_reset_usage": dev_reset_usage,
               "accept_lost_attempt": accept_lost_attempt, "allow_code_change": allow_code_change}
    _record_session(run, resumed=True, options=options, evaluator_kind=config.evaluator.kind)
    return run, _finish(session, max_calls=max_calls, dry_run=dry_run)


def _finish(session: OnlineSession, *, max_calls, dry_run) -> dict:
    try:
        return session.run(max_calls=max_calls, dry_run=dry_run)
    finally:
        if session.evaluator is not None:
            _append_jsonl(session.run_folder.path / "sessions.jsonl",
                          {"time": _now(), "event": "evaluator", **_jsonable(session.evaluator.describe())})


__all__ = ["OnlineRunError", "OnlineSession", "Journal", "batch_digest", "resume", "start"]
