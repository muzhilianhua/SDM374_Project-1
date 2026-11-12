"""Validation tests for the read-only online dashboard log reader."""
import sys

import pandas as pd
import pytest

sys.path.insert(0, "app")
from online_log import read_online_log, summarize_checks  # noqa: E402


def test_online_log_reader_normalises_call_and_score(tmp_path):
    path = tmp_path / "online_log.csv"
    pd.DataFrame({"call_no": [1, 1, 2], "R": [120.0, 119.0, 118.0], "simplex_ok": [True] * 3}).to_csv(path, index=False)

    frame, error = read_online_log(path)

    assert error is None
    assert frame["_call"].tolist() == [1, 1, 2]
    assert frame["_r"].tolist() == [120.0, 119.0, 118.0]


def test_online_log_reader_rejects_gapped_calls_and_partial_snapshot(tmp_path):
    path = tmp_path / "online_log.csv"
    pd.DataFrame({"call": [1, 3], "R": [120.0, 118.0]}).to_csv(path, index=False)
    frame, error = read_online_log(path)
    assert frame is None
    assert "连续" in error

    path.write_text("call,R\n1,120\n2,", encoding="utf-8")
    frame, error = read_online_log(path)
    assert frame is None
    assert "换行" in error


def test_summarize_checks_distinguishes_unknown_and_failure(tmp_path):
    path = tmp_path / "online_log.csv"
    pd.DataFrame({
        "call": [1, 1, 2], "R": [120.0, 119.0, 118.0],
        "simplex_ok": [True, True, True], "unique_ok": [True, "maybe", False],
    }).to_csv(path, index=False)
    frame, error = read_online_log(path)
    assert error is None
    summary = summarize_checks(frame)
    assert summary["全部通过"].tolist() == ["未知", "未通过"]


@pytest.mark.parametrize("columns", [
    {"call": [1], "R": ["bad"]},
    {"call": [1], "R": [float("inf")]},
    {"call": [1], "R": [float("nan")]},
    {"call": [1], "R": [True]},
    {"call": [True], "R": [120.0]},
    {"call": [0], "R": [120.0]},
    {"call": [1.5], "R": [120.0]},
    {"call": [1e30], "R": [120.0]},
    {"call": [2], "R": [120.0]},
    {"call": [1, 2, 1], "R": [120.0] * 3},
    {"call": [1], "prediction": [120.0]},
    {"call": [1], " call ": [1], "R": [120.0]},
    {"call": [1] * 51, "R": [120.0] * 51},
])
def test_online_log_reader_rejects_invalid_rows(tmp_path, columns):
    path = tmp_path / "online_log.csv"
    pd.DataFrame(columns).to_csv(path, index=False)

    frame, error = read_online_log(path)

    assert frame is None
    assert error


@pytest.mark.parametrize("contents", [None, "", "call,R\n"])
def test_missing_and_empty_logs_wait_for_the_writer(tmp_path, contents):
    path = tmp_path / "online_log.csv"
    if contents is not None:
        path.write_text(contents, encoding="utf-8")
    assert read_online_log(path) == (None, None)


def test_partial_numeric_score_is_not_shown_as_a_finished_observation(tmp_path):
    path = tmp_path / "online_log.csv"
    path.write_text("call,R\n1,120\n2,11", encoding="utf-8")
    frame, error = read_online_log(path)
    assert frame is None
    assert error
    with path.open("a", encoding="utf-8") as stream:
        stream.write("8\n")
    frame, error = read_online_log(path)
    assert error is None
    assert frame["_r"].tolist() == [120.0, 118.0]


@pytest.mark.parametrize("contents", [
    'call,R\n1,"unfinished\n', "call,R\n1,120,extra\n", "call,R\n1,1,120\n",
    "call,R,R\n1,120,110\n", "call,R\n1\n",
])
def test_malformed_csv_does_not_escape_as_a_page_exception(tmp_path, contents):
    path = tmp_path / "online_log.csv"
    path.write_text(contents, encoding="utf-8")
    frame, error = read_online_log(path)
    assert frame is None
    assert error


def test_check_values_accept_numeric_booleans_and_do_not_hide_unknowns():
    frame = pd.DataFrame({
        "_call": [1, 1, 2, 2, 3, 3],
        "check_simplex": [True, " YES ", True, "pending", None, False],
        "unique_ok": [1.0, "ok", "1.0", 1, True, "0.0"],
    })
    checks = summarize_checks(frame)
    assert checks["全部通过"].tolist() == ["通过", "未知", "未通过"]
    assert checks["check_simplex"].tolist() == ["通过", "未知", "未通过"]
    assert checks["unique_ok"].tolist() == ["通过", "通过", "未通过"]
    assert summarize_checks(frame[["_call"]]).empty
