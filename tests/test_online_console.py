"""Validation tests for the read-only online dashboard log reader."""
import sys

import pandas as pd
import pytest

sys.path.insert(0, "app")
from online_log import read_online_log  # noqa: E402


def test_online_log_reader_normalises_call_and_score(tmp_path):
    path = tmp_path / "online_log.csv"
    pd.DataFrame({"call_no": [1, 1, 2], "R": [120.0, 119.0, 118.0], "simplex_ok": [True] * 3}).to_csv(path, index=False)

    frame, error = read_online_log(path)

    assert error is None
    assert frame["_call"].tolist() == [1, 1, 2]
    assert frame["_r"].tolist() == [120.0, 119.0, 118.0]


@pytest.mark.parametrize("columns", [{"call": [1], "R": ["bad"]}, {"call": [0], "R": [120.0]}])
def test_online_log_reader_rejects_invalid_rows(tmp_path, columns):
    path = tmp_path / "online_log.csv"
    pd.DataFrame(columns).to_csv(path, index=False)

    frame, error = read_online_log(path)

    assert frame is None
    assert error
