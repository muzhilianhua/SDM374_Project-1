"""sdm.online.evaluator: the solver wrapper against a Python stand-in, and the table / synthetic stand-ins."""
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pytest

from sdm.data import load_dataset
from sdm.online import (
    EvaluatorError,
    EvaluatorTimeout,
    InvalidBatch,
    PointNotInTable,
    QuotaExhausted,
    SolverEvaluator,
    SyntheticEvaluator,
    TableEvaluator,
    build_evaluator,
)
from sdm.online.evaluator import RESET_USAGE_ENV, SOLVER_DIR, parse_solver_output, renormalise
from sdm.paths import REPO_ROOT
from sdm.simplex import sample

# Follows solver/README.md and the smoke-test findings: renormalises per building, exit 2 with a reason,
# a per-call quota in usage.count next to it. FAKE_SLEEP / FAKE_NOISE switch on misbehaviour for tests.
FAKE = '''import json, math, os, pathlib, sys, time
def fail(msg):
    print(msg, file=sys.stderr); sys.exit(2)
if os.environ.get("FAKE_SLEEP"):
    time.sleep(float(os.environ["FAKE_SLEEP"]))
data = json.loads(sys.stdin.read())
sets = data["weights"]
if not 1 <= len(sets) <= 50:
    fail("Batch must contain 1 to 50 weight sets.")
out = []
for v in sets:
    if len(v) != 21 or any(not math.isfinite(x) or x < 0 for x in v):
        fail("Weights must be finite and nonnegative.")
    b = [v[i * 7:(i + 1) * 7] for i in range(3)]
    n = [x / sum(bb) for bb in b for x in bb]
    out.append(110 + 10 * sum(x * x for x in n))
c = pathlib.Path("usage.count")
k = int(c.read_text()) if c.exists() else 0
if k >= 20:
    fail("Usage limit reached (20 batch invocations).")
c.write_text(str(k + 1))
if os.environ.get("FAKE_NOISE"):
    print("loading model...")
print(json.dumps({"R": out}))
'''


def truth(W):
    return 110 + 10 * (renormalise(np.asarray(W)) ** 2).sum(axis=1)


@pytest.fixture
def fake_solver(tmp_path):
    script = tmp_path / "fake_solver.py"
    script.write_text(FAKE)

    def make(**kwargs):
        return SolverEvaluator(tmp_path / "solver.exe", command=[sys.executable, str(script)],
                               checksums=None, **kwargs)
    return make


def test_solver_returns_r_in_input_order(fake_solver):
    W = sample(5, np.random.default_rng(0))
    evaluator = fake_solver()
    R = evaluator.evaluate(W)
    np.testing.assert_allclose(R, truth(W), rtol=0, atol=1e-12)
    assert evaluator.calls == 1 and evaluator.remaining == 19
    record = evaluator.history[0]
    assert record.n == 5 and record.R == tuple(R) and record.seconds > 0
    assert record.extra == {"attempt": 1, "usage_file_before": False, "usage_file_after": True}


def test_solver_accepts_one_set_in_any_shape_and_unnormalised_weights(fake_solver):
    W = sample(1, np.random.default_rng(1))[0]
    evaluator = fake_solver()
    scaled = W.reshape(3, 7) * np.array([[3.0], [0.5], [10.0]])
    assert evaluator.evaluate(W)[0] == pytest.approx(evaluator.evaluate(scaled)[0], abs=1e-12)


@pytest.mark.parametrize("W, message", [
    (np.full((51, 21), 1 / 7), "1 to 50"),
    (np.full((1, 20), 0.05), "expected weights of shape"),
    (np.r_[-0.1, np.full(20, 0.1)], "non-negative"),
    (np.r_[np.zeros(7), np.full(14, 1 / 7)], "positive"),
    (np.r_[np.nan, np.full(20, 0.1)], "finite"),
])
def test_invalid_batches_are_rejected_before_the_solver_runs(fake_solver, tmp_path, W, message):
    evaluator = fake_solver()
    with pytest.raises(InvalidBatch, match=message):
        evaluator.evaluate(W)
    assert evaluator.calls == 0 and not (tmp_path / "usage.count").exists()


def test_formal_mode_keeps_usage_count_and_hits_the_solver_quota(fake_solver, tmp_path):
    (tmp_path / "usage.count").write_text("19")      # e.g. a resumed run: the solver's count persists
    evaluator = fake_solver()
    evaluator.evaluate(np.full(21, 1 / 7))
    with pytest.raises(QuotaExhausted, match="Usage limit reached") as caught:
        evaluator.evaluate(np.full(21, 1 / 7))
    assert caught.value.returncode == 2 and "20 batch invocations" in caught.value.stderr
    assert evaluator.calls == 1 and (tmp_path / "usage.count").read_text() == "20"


def test_local_quota_stops_before_the_solver_is_started(fake_solver, tmp_path):
    evaluator = fake_solver(quota=2)
    for _ in range(2):
        evaluator.evaluate(np.full(21, 1 / 7))
    with pytest.raises(QuotaExhausted, match="quota of 2"):
        evaluator.evaluate(np.full(21, 1 / 7))
    assert (tmp_path / "usage.count").read_text() == "2"


def test_development_mode_resets_usage_before_every_call(fake_solver, tmp_path, monkeypatch):
    monkeypatch.setenv(RESET_USAGE_ENV, "1")
    (tmp_path / "usage.count").write_text("20")
    evaluator = fake_solver(reset_usage=True, quota=None)
    for _ in range(3):
        evaluator.evaluate(np.full(21, 1 / 7))
    assert (tmp_path / "usage.count").read_text() == "1" and evaluator.remaining is None


def test_reset_usage_needs_the_development_switch(fake_solver, monkeypatch):
    monkeypatch.delenv(RESET_USAGE_ENV, raising=False)
    with pytest.raises(EvaluatorError, match=RESET_USAGE_ENV):
        fake_solver(reset_usage=True)


def test_configs_cannot_reset_usage(monkeypatch):
    # A frozen config runs on the course computer, where deleting usage.count would bypass the quota.
    monkeypatch.setenv(RESET_USAGE_ENV, "1")
    with pytest.raises(ValueError, match="cannot come from a config"):
        build_evaluator("solver", {"reset_usage": True, "checksums": None})


def test_failed_and_timed_out_attempts_count_against_the_quota(fake_solver, tmp_path, monkeypatch):
    script = tmp_path / "broken.py"
    script.write_text("import sys; print('boom', file=sys.stderr); sys.exit(3)")
    broken = SolverEvaluator(tmp_path / "solver.exe", command=[sys.executable, str(script)], checksums=None)
    with pytest.raises(EvaluatorError):
        broken.evaluate(np.full(21, 1 / 7))
    assert broken.calls == 0 and broken.attempts == 1 and broken.remaining == 19
    assert broken.failures[0]["error"] == "EvaluatorError" and broken.failures[0]["returncode"] == 3

    monkeypatch.setenv("FAKE_SLEEP", "5")
    slow = fake_solver(timeout=0.5, quota=1)
    with pytest.raises(EvaluatorTimeout):
        slow.evaluate(np.full(21, 1 / 7))
    with pytest.raises(QuotaExhausted, match="quota of 1"):
        slow.evaluate(np.full(21, 1 / 7))
    assert slow.attempts == 1 and slow.describe()["failures"][0]["error"] == "EvaluatorTimeout"


def test_solver_error_keeps_stderr(tmp_path):
    script = tmp_path / "broken.py"
    script.write_text("import sys; print('boom', file=sys.stderr); sys.exit(3)")
    evaluator = SolverEvaluator(tmp_path / "solver.exe", command=[sys.executable, str(script)], checksums=None)
    with pytest.raises(EvaluatorError, match="code 3: boom") as caught:
        evaluator.evaluate(np.full(21, 1 / 7))
    assert not isinstance(caught.value, QuotaExhausted) and caught.value.stderr == "boom"
    assert evaluator.calls == 0


def test_timeout_kills_the_solver(fake_solver, monkeypatch):
    monkeypatch.setenv("FAKE_SLEEP", "5")
    with pytest.raises(EvaluatorTimeout, match="within 0.5 s"):
        fake_solver(timeout=0.5).evaluate(np.full(21, 1 / 7))


def _alive(pid: int) -> bool:
    if sys.platform == "win32":
        # Bytes: tasklist prints in the console code page (cp936 on the course computer), not UTF-8.
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True).stdout
        return str(pid).encode() in out
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_timeout_kills_the_whole_process_tree(tmp_path):
    # Like solver.exe, which starts Lab6Solver-<hash>.exe and a process pool under it.
    pid_file = tmp_path / "child.pid"
    script = tmp_path / "spawner.py"
    script.write_text(
        "import pathlib, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid))\n"
        "time.sleep(60)\n")
    evaluator = SolverEvaluator(tmp_path / "solver.exe", command=[sys.executable, str(script)],
                                checksums=None, timeout=3)
    with pytest.raises(EvaluatorTimeout) as caught:
        evaluator.evaluate(np.full(21, 1 / 7))
    assert caught.value.returncode is not None
    assert evaluator.failures[0]["returncode"] is not None
    child = int(pid_file.read_text())
    deadline = time.time() + 10
    while _alive(child) and time.time() < deadline:
        time.sleep(0.2)
    assert not _alive(child)


def test_error_line_on_stdout_is_reported(tmp_path):
    # A dying pool worker makes the solver print {"error": ...} on stdout, possibly with exit code 0.
    script = tmp_path / "pool_error.py"
    script.write_text('import json, sys; sys.stdin.read(); '
                      'print(json.dumps({"error": "A process in the process pool was terminated abruptly"}))')
    evaluator = SolverEvaluator(tmp_path / "solver.exe", command=[sys.executable, str(script)], checksums=None)
    with pytest.raises(EvaluatorError, match="terminated abruptly") as caught:
        evaluator.evaluate(np.full(21, 1 / 7))
    assert not isinstance(caught.value, QuotaExhausted)
    assert evaluator.attempts == 1 and evaluator.remaining == 19
    assert evaluator.failures[0]["message"].startswith("solver reported an error")


def test_usage_limit_is_recognised_on_stdout(tmp_path):
    with pytest.raises(QuotaExhausted):
        parse_solver_output('{"error": "Usage limit reached (20 batch invocations)."}')
    script = tmp_path / "limit_stdout.py"
    script.write_text("import sys; sys.stdin.read(); print('Usage limit reached (20 batch invocations).'); sys.exit(2)")
    evaluator = SolverEvaluator(tmp_path / "solver.exe", command=[sys.executable, str(script)], checksums=None)
    with pytest.raises(QuotaExhausted, match="code 2: Usage limit"):
        evaluator.evaluate(np.full(21, 1 / 7))


def test_log_lines_before_the_json_are_ignored(fake_solver, monkeypatch):
    monkeypatch.setenv("FAKE_NOISE", "1")
    assert fake_solver().evaluate(np.full(21, 1 / 7))[0] == pytest.approx(110 + 30 / 7)


@pytest.mark.parametrize("stdout", ["", "not json", '{"R": [1.0, NaN]}', '{"r": [1.0]}', '{"R": [true]}'])
def test_unusable_output_raises(stdout):
    with pytest.raises(EvaluatorError, match="no"):
        parse_solver_output(stdout)


def test_wrong_number_of_r_values_raises(tmp_path):
    script = tmp_path / "short.py"
    script.write_text('import json, sys; sys.stdin.read(); print(json.dumps({"R": [1.0]}))')
    evaluator = SolverEvaluator(tmp_path / "solver.exe", command=[sys.executable, str(script)], checksums=None)
    with pytest.raises(EvaluatorError, match="expected 2 finite R values"):
        evaluator.evaluate(np.full((2, 21), 1 / 7))
    # The solver ran, so the attempt is charged and recorded for a resumed run.
    assert evaluator.calls == 0 and evaluator.attempts == 1 and evaluator.remaining == 19
    assert [failure["error"] for failure in evaluator.describe()["failures"]] == ["EvaluatorError"]


def test_checksum_is_verified(tmp_path):
    exe = tmp_path / "solver.exe"
    exe.write_bytes(b"not the course solver")
    sums = tmp_path / "CHECKSUMS.sha256"
    sums.write_text("0" * 64 + "  solver.exe\n")
    with pytest.raises(EvaluatorError, match="does not match the pinned"):
        SolverEvaluator(exe, checksums=sums)
    with pytest.raises(FileNotFoundError, match="solver.zip"):
        SolverEvaluator(tmp_path / "missing.exe", checksums=sums)


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(REPO_ROOT / "dataset_same8")


def test_table_returns_measured_rows_and_rejects_other_points(dataset):
    rows = [0, 5, 7]
    table = TableEvaluator.from_dataset(dataset, rows=rows)
    np.testing.assert_array_equal(table.evaluate(dataset.X[[7, 0]]), dataset.y[[7, 0]])
    assert table.history[0].extra["ids"] == [dataset.ids[7], dataset.ids[0]]
    np.testing.assert_array_equal(table.evaluate(dataset.X[5] * 4), dataset.y[[5]])
    with pytest.raises(PointNotInTable):
        table.evaluate(dataset.X[[0, 1]])
    assert table.calls == 2


def test_empty_table_is_rejected():
    with pytest.raises(ValueError, match="at least one row"):
        TableEvaluator(np.empty((0, 21)), np.empty(0))


def test_table_ids_must_match_the_rows(dataset):
    with pytest.raises(ValueError, match="1 ids for 3 rows"):
        TableEvaluator(dataset.X[:3], dataset.y[:3], ids=[dataset.ids[0]])


def test_best_training_row_is_in_the_full_table(dataset):
    table = build_evaluator("table", dataset=dataset)
    best = dataset.ids.index("p01969")
    assert table.evaluate(dataset.X[best])[0] == pytest.approx(115.76481925917324, abs=1e-9)


def test_synthetic_truth_is_the_fitted_mean_and_is_deterministic(dataset):
    evaluator = build_evaluator("synthetic", {"model": "ridge2"}, dataset=dataset)
    W = sample(4, np.random.default_rng(2))
    first = evaluator.evaluate(W)
    np.testing.assert_array_equal(first, evaluator.truth.predict(W)[0])
    np.testing.assert_array_equal(first, evaluator.evaluate(W[::-1])[::-1])
    assert evaluator.describe()["truth"] == "ridge2"
    assert isinstance(evaluator, SyntheticEvaluator)


def test_build_evaluator_rejects_bad_specs(dataset):
    with pytest.raises(ValueError, match="unknown evaluator kind"):
        build_evaluator("oracle", dataset=dataset)
    with pytest.raises(ValueError, match="needs the dataset"):
        build_evaluator("table")
    with pytest.raises(ValueError, match="params.model"):
        build_evaluator("synthetic", {}, dataset=dataset)


REAL_EXE = SOLVER_DIR / "solver.exe"


@pytest.mark.skipif(sys.platform != "win32" or not REAL_EXE.exists(), reason="needs the course solver.exe on Windows")
def test_real_solver_reproduces_the_best_training_row(dataset, tmp_path):
    # Runs in a copy of solver/ so the repository's usage.count is never touched.
    for name in ("solver.exe", "CHECKSUMS.sha256"):
        shutil.copy2(SOLVER_DIR / name, tmp_path / name)
    evaluator = SolverEvaluator(tmp_path / "solver.exe", checksums=tmp_path / "CHECKSUMS.sha256")
    best = dataset.ids.index("p01969")
    assert evaluator.evaluate(dataset.X[best])[0] == pytest.approx(115.76481925917324, abs=1e-6)
    assert json.dumps(evaluator.describe())
