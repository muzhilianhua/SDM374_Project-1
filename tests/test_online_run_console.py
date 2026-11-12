"""T11 readers for strict T07 batches and the T08 attempts journal."""
import json

import numpy as np

from online_log import read_online_log
from online_run import discover_online_runs, read_attempts, read_run
from sdm.online.log import OnlineLog


def _weights(n):
    values = np.full((n, 21), 1 / 7)
    for row in range(n):
        delta = (row + 1) * 1e-3
        values[row, 0] += delta
        values[row, 1] -= delta
    return values


def test_official_log_requires_complete_point_n_batch(tmp_path):
    path = tmp_path / "online_log.csv"
    OnlineLog(path).append(_weights(2), [120, 119], source="online-batch", run_id="r", call=1)
    frame, error = read_online_log(path)
    assert error is None and frame["point"].tolist() == [1, 2]
    text = path.read_text(encoding="utf-8")
    path.write_text(text.rsplit("\n", 2)[0] + "\n", encoding="utf-8")
    frame, error = read_online_log(path)
    assert frame is None and "不完整" in error


def test_attempts_journal_counts_pending_and_rejected(tmp_path):
    path = tmp_path / "attempts.jsonl"
    records = [
        {"attempt": 1, "call": 1, "event": "started"},
        {"attempt": 1, "call": 1, "event": "rejected"},
        {"attempt": 2, "call": 1, "event": "started"},
        {"attempt": 2, "call": 1, "event": "ok"},
        {"attempt": 3, "call": 2, "event": "started"},
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    summary = read_attempts(path)
    assert summary["used"] == 2 and summary["pending"] == [3]


def test_online_run_does_not_understate_missing_journal(tmp_path):
    run = tmp_path / "2026-online-demo"
    run.mkdir()
    (run / "meta.json").write_text(json.dumps({"kind": "online"}), encoding="utf-8")
    (run / "online_log.csv").write_text("call,R\n1,120\n", encoding="utf-8")
    snapshot = read_run(run)
    assert snapshot.used is None and "缺少 attempts" in snapshot.error
    assert discover_online_runs(tmp_path) == [run]
