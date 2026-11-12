"""sdm.online.loop (T08): the online main loop, its pre-submit check, budget accounting and resume."""
import csv
import json
import sys

import numpy as np
import pytest

from sdm.cli import main
from sdm.config import load_config
from sdm.interfaces import BatchPolicy, Proposal
from sdm.online import Evaluator, EvaluatorError, QuotaExhausted, SolverEvaluator
from sdm.online import loop
from sdm.online.evaluator import renormalise
from sdm.online.screen import screen
from sdm.runs import create_run, list_runs
from sdm.simplex import sample
from tests.test_online_evaluator import FAKE

CONFIG = """
[data]
dataset = "dataset_same8"

[model]
name = "ridge2"

[optimizer]
name = "trust-dirichlet"

[optimizer.params]
centers = 4
iterations = 5
samples_per_center = 10
seed = 7

[evaluator]
kind = "synthetic"

[evaluator.params]
model = "ridge2"

[online]
calls = {calls}
batch_size = {batch}
candidate_margin = 5
"""


def truth(W):
    return 110 + 10 * (renormalise(np.asarray(W)) ** 2).sum(axis=1)


def write_config(tmp_path, calls=3, batch=5, name="online-test"):
    path = tmp_path / f"{name}.toml"
    path.write_text(CONFIG.format(calls=calls, batch=batch), encoding="utf-8")
    return path


def log_rows(run):
    with open(run.path / "online_log.csv", encoding="utf-8") as stream:
        return [{key: value for key, value in row.items() if key not in ("run_id", "timestamp")}
                for row in csv.DictReader(stream)]


def journal(run):
    return [json.loads(line) for line in (run.path / "attempts.jsonl").read_text().splitlines()]


class Truth(Evaluator):
    """Stand-in whose first `fail` calls raise `error`."""

    kind = "truth"

    def __init__(self, fail=0, error=EvaluatorError, **kwargs):
        super().__init__(**kwargs)
        self.fail, self.error = fail, error

    def _run(self, W):
        if self.fail:
            self.fail -= 1
            raise self.error("boom", stderr="worker died", returncode=2)
        return truth(W), "", {}


class Sampler(BatchPolicy):
    """Random points, deterministic per call; `lead` puts copies of training rows first."""

    name = "sampler"

    def __init__(self, lead=0, only_training=False):
        super().__init__()
        self.lead, self.only_training, self.states = lead, only_training, []

    def propose(self, state, n):
        self.states.append(state)
        if self.only_training:
            return Proposal(state.X_train[:n])
        points = sample(n, np.random.default_rng(state.seed + state.call))
        weights = np.vstack([state.X_train[:self.lead], points])[:max(n, self.lead + 1)]
        return Proposal(weights, predicted_mean=truth(weights) - 1, predicted_std=np.ones(len(weights)))


@pytest.fixture
def session_factory(tmp_path, dataset):
    def make(policy=None, evaluator=None, calls=3, batch=5, **online):
        path = write_config(tmp_path, calls=calls, batch=batch)
        if online:
            path.write_text(path.read_text() + "".join(f"{k} = {v}\n" for k, v in online.items()))
        config = load_config(path)
        run = create_run("online", path, name=path.stem, options={}, dataset=dataset, root=tmp_path / "runs")
        made = evaluator if evaluator is not None else Truth()
        session = loop.OnlineSession(run, config, dataset, policy=policy or Sampler(),
                                     evaluator_factory=lambda calls_left: made, echo=lambda text: None)
        return session
    return make


def test_cli_run_and_resume_send_the_same_batches(tmp_path, capsys):
    config = write_config(tmp_path)
    runs = tmp_path / "runs"
    assert main(["--runs-dir", str(runs), "online", "-c", str(config), "--max-calls", "1"]) == 0
    (first,) = list_runs(runs)
    assert first.read_json("metrics.json")["stop_reason"] == "max_calls"
    assert main(["--runs-dir", str(runs), "online", "--resume", first.id, "--allow-code-change"]) == 0
    assert main(["--runs-dir", str(runs), "online", "-c", str(config), "--name", "straight"]) == 0
    straight = next(run for run in list_runs(runs) if run.id.endswith("straight"))

    assert log_rows(first) == log_rows(straight) and len(log_rows(first)) == 15
    metrics = first.read_json("metrics.json")
    assert metrics["finished"] and metrics["stop_reason"] == "budget_used"
    assert metrics["attempts"] == metrics["calls_completed"] == 3 and metrics["calls_left"] == 0
    assert metrics["best_R"] == min(float(row["R"]) for row in log_rows(first))
    with open(first.path / "predictions.csv", encoding="utf-8") as stream:
        keys = [(row["call"], row["point"]) for row in csv.DictReader(stream)]
    assert keys == [(row["call"], row["point"]) for row in log_rows(first)]
    sessions = [json.loads(line) for line in (first.path / "sessions.jsonl").read_text().splitlines()]
    assert [item.get("resumed") for item in sessions if "resumed" in item] == [False, True]
    assert "online-test" in capsys.readouterr().out


def test_first_call_sees_only_training_rows(session_factory, dataset):
    policy = Sampler()
    session = session_factory(policy=policy, calls=2)
    session.run()
    first, second = policy.states
    assert first.call == 1 and len(first.X_online) == 0 and np.array_equal(first.X, dataset.X)
    assert second.call == 2 and len(second.X_online) == 5 and set(second.online_call) == {1}
    assert second.calls_left == 1 and second.best_online == min(row.R for row in session.log.rows())


def test_screen_drops_invalid_training_sent_scaled_and_repeated_points(dataset):
    new = sample(3, np.random.default_rng(0))
    broken = new[0].copy()
    broken[0] += 0.5
    candidates = np.vstack([broken, dataset.X[4], new[1], new[1] + 0, new[2], new[0]])
    result = screen(candidates, train=dataset.X, sent=(new[2] * 3).reshape(1, -1), n=5)
    assert result.accepted == (2, 5)
    assert [item["reason"] for item in result.rejected] == ["constraint", "training_row", "batch_duplicate",
                                                           "sent_before"]
    assert screen(candidates, train=dataset.X, sent=np.empty((0, 21)), n=1).examined == 3


def test_policy_is_asked_for_more_when_too_many_points_are_dropped(session_factory):
    session = session_factory(policy=Sampler(lead=12), calls=1)
    session.run()
    record = json.loads(session.call_path(1).read_text())
    assert [item["requested"] for item in record["rounds"]] == [10, 20]
    assert record["rounds"][0]["rejected"]["training_row"] == 12 and len(record["weights"]) == 5
    assert len(record["predicted_mean"]) == 5 and record["status"] == "measured"


def test_no_valid_candidates_stops_without_calling(session_factory):
    session = session_factory(policy=Sampler(only_training=True))
    summary = session.run()
    assert not session.journal_path.exists() and summary["attempts"] == 0
    assert session.run_folder.read_json("metrics.json")["stop_reason"] == "no_valid_candidates"
    assert json.loads(session.call_path(1).read_text())["status"] == "not_sent"


def _same_points(a, b):
    a, b = renormalise(np.asarray(a).reshape(-1, 21)), renormalise(np.asarray(b).reshape(-1, 21))
    return sum(bool((np.abs(b - row).max(axis=1) <= 1e-12).any()) for row in a)


def test_failed_batch_is_charged_and_never_resent(session_factory):
    # Decision 28: once the solver started, the batch counts as sent even without R.
    policy = Sampler()
    session = session_factory(policy=policy, evaluator=Truth(fail=1), calls=3)
    summary = session.run()
    events = [(item["attempt"], item["call"], item["event"]) for item in journal(session.run_folder)]
    assert events == [(1, 1, "started"), (1, 1, "failed"), (2, 1, "started"), (2, 1, "ok"),
                      (3, 2, "started"), (3, 2, "ok")]
    assert summary["calls_completed"] == 2 and summary["attempts"] == 3 and summary["calls_left"] == 0

    failed = json.loads(session.failed_attempt_path(1, 1).read_text())
    assert failed["status"] == "failed" and failed["attempts"][0]["stderr"] == "worker died"
    failed_points = np.asarray(failed["weights"]).reshape(-1, 21)
    sent_later = np.array([row.weights for row in session.log.rows()])
    assert _same_points(failed_points, sent_later) == 0
    # The policy proposed the same points again (it is deterministic per call); the screen dropped them.
    retry = json.loads(session.call_path(1).read_text())
    assert retry["status"] == "measured" and retry["rounds"][0]["rejected"]["sent_before"] == 5
    # Later calls see the failed points as sent but not as observations.
    later = policy.states[-1]
    assert len(later.X_unmeasured) == 5 and len(later.X_sent) == len(later.X_online) + 5
    assert len(later.y) == len(later.y_train) + len(later.y_online)


def test_repeated_failure_stops_the_run(session_factory):
    session = session_factory(evaluator=Truth(fail=5), calls=5, max_failures=2)
    summary = session.run()
    assert summary["attempts"] == 2 and summary["calls_completed"] == 0
    assert session.run_folder.read_json("metrics.json")["stop_reason"] == "evaluator_failed"
    first, second = (json.loads(session.failed_attempt_path(1, k).read_text())["weights"] for k in (1, 2))
    assert _same_points(first, second) == 0


def test_failures_stop_at_the_budget(session_factory):
    # Running out of calls right after a failure is a failure, not a normal finish (PR #21 review).
    session = session_factory(evaluator=Truth(fail=5), calls=2, max_failures=5)
    summary = session.run()
    assert summary["attempts"] == 2 and summary["calls_left"] == 0
    assert session.run_folder.read_json("metrics.json")["stop_reason"] == "evaluator_failed"


def test_last_attempt_failing_exits_nonzero(tmp_path, monkeypatch):
    config = write_config(tmp_path, calls=1)
    monkeypatch.setattr(loop, "build_evaluator", lambda kind, params, dataset: Truth(fail=1))
    assert main(["--runs-dir", str(tmp_path / "runs"), "online", "-c", str(config)]) == 1
    (run,) = list_runs(tmp_path / "runs")
    metrics = run.read_json("metrics.json")
    assert metrics["stop_reason"] == "evaluator_failed" and metrics["calls_completed"] == 0


class Dwindling(BatchPolicy):
    """Round 1 offers one new point after training rows; later rounds offer training rows only."""

    name = "dwindling"

    def propose(self, state, n):
        if n <= state.batch_size + 5:
            fresh = sample(1, np.random.default_rng(state.call))
            return Proposal(np.vstack([state.X_train[:4], fresh]), predicted_mean=np.arange(5.0))
        return Proposal(state.X_train[:n], predicted_mean=np.zeros(n))


def test_refill_keeps_points_accepted_in_earlier_rounds(session_factory):
    session = session_factory(policy=Dwindling(), calls=1)
    summary = session.run()
    record = json.loads(session.call_path(1).read_text())
    assert [item["accepted"] for item in record["rounds"]] == [1, 0, 0]
    assert len(record["weights"]) == 1 and record["predicted_mean"] == [4.0]
    assert summary["calls_completed"] == 1 and summary["points_evaluated"] == 1
    assert {item["round"] for item in record["rejections"]} == {1, 2, 3}


def test_refill_rounds_screen_against_points_already_taken(dataset):
    taken = sample(2, np.random.default_rng(3))
    result = screen(np.vstack([taken[1], taken[0]]), train=dataset.X, sent=np.empty((0, 21)), n=5,
                    accepted=taken)
    assert result.accepted == () and result.counts()["batch_duplicate"] == 2


class Patchy(Sampler):
    """Predicts on every call except `skip`."""

    def __init__(self, skip):
        super().__init__()
        self.skip = skip

    def propose(self, state, n):
        proposal = super().propose(state, n)
        if state.call == self.skip:
            return Proposal(proposal.weights)
        return proposal


def test_predictions_sidecar_needs_every_logged_point(session_factory):
    session = session_factory(policy=Patchy(skip=2), calls=3)
    session.run(max_calls=1)
    path = session.run_folder.path / "predictions.csv"
    assert path.exists() and session.run_folder.read_json("metrics.json")["predictions_complete"]
    session.run()
    assert session.run_folder.read_json("metrics.json")["calls_completed"] == 3
    # Call 2 has no predictions, so no file is better than a stale or partial one (T26 needs exact keys).
    assert not path.exists() and session.run_folder.read_json("metrics.json")["predictions_complete"] is False


class Interrupted(Truth):
    """The solver used the call, then Ctrl-C arrived before R came back."""

    def _run(self, W):
        raise KeyboardInterrupt


def test_ctrl_c_during_a_call_leaves_the_attempt_open(session_factory):
    session = session_factory(evaluator=Interrupted(), calls=3)
    with pytest.raises(KeyboardInterrupt):
        session.run()
    assert [item["event"] for item in journal(session.run_folder)] == ["started"]
    session.evaluator = Truth()
    with pytest.raises(loop.OnlineRunError, match="accept-lost-attempt"):
        session.run()
    assert session.accept_lost_attempts() == 1
    summary = session.run()
    assert summary["attempts"] == 3 and summary["calls_completed"] == 2


class OutsidePool(Truth):
    def _run(self, W):
        from sdm.online import PointNotInTable
        raise PointNotInTable("not a table row")


def test_rejected_attempt_is_not_charged(session_factory):
    from sdm.online import PointNotInTable
    session = session_factory(evaluator=OutsidePool(), calls=3)
    with pytest.raises(PointNotInTable):
        session.run()
    assert [item["event"] for item in journal(session.run_folder)] == ["started", "rejected"]
    assert session.journal().used == 0 and session.journal().pending == []
    assert len(session.state().X_sent) == 0


def test_solver_process_tree_is_killed_on_ctrl_c(tmp_path, monkeypatch):
    import subprocess
    from sdm.online import evaluator as evaluator_module
    script = tmp_path / "fake_solver.py"
    script.write_text(FAKE)
    solver = SolverEvaluator(tmp_path / "solver.exe", command=[sys.executable, str(script)], checksums=None)
    killed = []
    monkeypatch.setattr(evaluator_module, "kill_process_tree", lambda process: (killed.append(process.pid),
                                                                               process.kill()))

    def interrupt(self, *args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(subprocess.Popen, "communicate", interrupt)
    with pytest.raises(KeyboardInterrupt):
        solver.evaluate(np.full(21, 1 / 7))
    assert len(killed) == 1


def test_accept_lost_attempt_recovers_measured_values_first(tmp_path, monkeypatch):
    config = write_config(tmp_path, calls=2)
    runs = tmp_path / "runs"
    original = loop.OnlineLog.append

    def crash(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(loop.OnlineLog, "append", crash)
    with pytest.raises(KeyboardInterrupt):
        loop.start(config, runs_root=runs, echo=lambda text: None)
    monkeypatch.setattr(loop.OnlineLog, "append", original)
    (run,) = list_runs(runs)
    loop.resume(run.id, runs_root=runs, accept_lost_attempt=True, allow_code_change=True, echo=lambda text: None)
    events = [item["event"] for item in journal(run)]
    assert "lost" not in events and any(item.get("recovered") for item in journal(run))
    assert run.read_json("metrics.json")["calls_completed"] == 2


def test_torn_last_journal_line_is_cut_and_kept(session_factory):
    session = session_factory(calls=3)
    session.run(max_calls=1)
    with open(session.journal_path, "a", encoding="utf-8") as stream:
        stream.write('{"attempt": 2, "call": 2, "event": "sta')
    summary = session.run()
    assert summary["calls_completed"] == 3 and summary["attempts"] == 3
    assert (session.run_folder.path / "attempts.jsonl.torn").read_text() == '{"attempt": 2, "call": 2, "event": "sta\n'


def test_quota_message_stops_at_once(session_factory):
    session = session_factory(evaluator=Truth(fail=1, error=QuotaExhausted), max_failures=3)
    session.run()
    assert session.run_folder.read_json("metrics.json")["stop_reason"] == "quota_exhausted"
    assert session.journal().used == 1


def test_attempt_without_outcome_needs_the_operator(session_factory):
    session = session_factory(calls=3)
    session.run(max_calls=1)
    # The process died while the solver ran call 2: the journal has the batch, the log has no R for it.
    batch, record = session.propose(session.state())
    loop._write_json_atomic(session.call_path(2), record)
    with open(session.journal_path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps({"attempt": 2, "call": 2, "event": "started", "n": len(batch),
                                 "batch_sha256": record["batch_sha256"], "weights": batch.tolist()}) + "\n")
    with pytest.raises(loop.OnlineRunError, match="accept-lost-attempt"):
        session.run()
    assert session.accept_lost_attempts() == 1
    assert json.loads(session.failed_attempt_path(2, 2).read_text())["status"] == "lost"
    summary = session.run()
    assert summary["attempts"] == 3 and summary["calls_completed"] == 2 and summary["calls_left"] == 0
    # The lost batch is not resent after the restart.
    assert _same_points(batch, [row.weights for row in session.log.rows()]) == 0


def test_lost_attempt_without_its_batch_is_refused(session_factory):
    session = session_factory(calls=3)
    session.run(max_calls=1)
    with open(session.journal_path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps({"attempt": 2, "call": 2, "event": "started", "n": 5, "batch_sha256": "x"}) + "\n")
    with pytest.raises(loop.OnlineRunError, match="no batch in the journal"):
        session.accept_lost_attempts()


def test_measured_values_survive_a_crash_before_logging(session_factory, monkeypatch):
    session = session_factory(calls=2)
    original = type(session.log).append

    def crash(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(type(session.log), "append", crash)
    with pytest.raises(KeyboardInterrupt):
        session.run()
    monkeypatch.setattr(type(session.log), "append", original)
    # The solver answered, but the process died before the log append: the attempt is still open.
    assert journal(session.run_folder)[-1]["event"] == "started" and session.log.rows() == []
    summary = session.run()
    assert summary["calls_completed"] == 2 and summary["attempts"] == 2
    assert any(item.get("recovered") for item in journal(session.run_folder))


def test_resume_refuses_other_code(tmp_path, monkeypatch):
    config = write_config(tmp_path, calls=2)
    runs = tmp_path / "runs"
    assert main(["--runs-dir", str(runs), "online", "-c", str(config), "--max-calls", "1"]) == 0
    (run,) = list_runs(runs)
    monkeypatch.setattr(loop, "git_state", lambda: {"commit": "other", "dirty": False})
    assert main(["--runs-dir", str(runs), "online", "--resume", run.id]) == 2
    assert run.read_json("metrics.json")["calls_completed"] == 1
    assert main(["--runs-dir", str(runs), "online", "--resume", run.id, "--allow-code-change"]) == 0


def test_dry_run_calls_nothing(tmp_path):
    config = write_config(tmp_path)
    runs = tmp_path / "runs"
    assert main(["--runs-dir", str(runs), "online", "-c", str(config), "--dry-run"]) == 0
    (run,) = list_runs(runs)
    assert not (run.path / "attempts.jsonl").exists() and (run.path / "calls" / "call_01_dry_run.json").exists()
    assert run.read_json("metrics.json")["stop_reason"] == "dry_run"


def test_solver_stand_in_end_to_end_keeps_usage_count(session_factory, tmp_path):
    script = tmp_path / "fake_solver.py"
    script.write_text(FAKE)
    solver = SolverEvaluator(tmp_path / "solver.exe", command=[sys.executable, str(script)], checksums=None)
    session = session_factory(evaluator=solver, calls=2, batch=3)
    summary = session.run()
    assert summary["calls_completed"] == 2 and (tmp_path / "usage.count").read_text() == "2"
    np.testing.assert_allclose([row.R for row in session.log.rows()],
                               truth(np.array([row.weights for row in session.log.rows()])), atol=1e-12)


@pytest.mark.parametrize("extra, message", [
    ("[online]\nbatch_size = 51\n", "batch_size"),
    ("[online]\nquota = 3\n", r"\[online\] accepts"),
    ("[evaluator]\nname = \"solver\"\n", r"\[evaluator\] needs `kind`"),
])
def test_bad_online_sections_are_rejected(tmp_path, extra, message):
    path = tmp_path / "bad.toml"
    path.write_text("[model]\nname = \"ridge2\"\n" + extra, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_config(path)


def test_online_config_needs_an_evaluator(tmp_path, capsys):
    path = tmp_path / "no-evaluator.toml"
    path.write_text("[model]\nname = \"ridge2\"\n[optimizer]\nname = \"trust-dirichlet\"\n", encoding="utf-8")
    assert main(["--runs-dir", str(tmp_path / "runs"), "online", "-c", str(path)]) == 2
    assert "[evaluator]" in capsys.readouterr().err and not (tmp_path / "runs").exists()


def test_shipped_online_configs_load():
    from sdm.paths import REPO_ROOT
    for name in ("online-baseline", "online-baseline-synthetic"):
        config = load_config(REPO_ROOT / "experiments" / f"{name}.toml")
        assert config.online.calls == 20 and config.online.batch_size == 50 and config.policy.name == "surrogate"
    assert load_config(REPO_ROOT / "experiments" / "online-baseline.toml").evaluator.kind == "solver"
