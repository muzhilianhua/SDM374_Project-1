import csv

import numpy as np
import pytest

from sdm.online.log import FIELDS, OnlineLog, OnlineLogError


def weights(n):
    result = np.full((n, 21), 1 / 7)
    for index in range(n):
        result[index, 0] += index / 100
        result[index, 1] -= index / 100
    return result


def shifted_weights(shift):
    result = weights(1)
    result[0, 0] += shift
    result[0, 1] -= shift
    return result


def test_append_writes_header_batch_and_resume(tmp_path):
    log = OnlineLog(tmp_path / "online_log.csv")
    first = log.append(weights(2), [120.0, 119.0], source="explore", run_id="run-a")
    second = log.append(shifted_weights(0.02), [118.0], source="exploit", run_id="run-b")
    assert [row.call for row in first + second] == [1, 1, 2]
    assert log.next_call == 3
    assert log.best().R == 118.0
    with (tmp_path / "online_log.csv").open(newline="") as stream:
        assert next(csv.reader(stream)) == list(FIELDS)


def test_rejects_non_contiguous_call_and_duplicate_points(tmp_path):
    log = OnlineLog(tmp_path / "online_log.csv")
    log.append(weights(1), [120], source="a", run_id="r")
    with pytest.raises(OnlineLogError, match="expected call 2"):
        log.append(shifted_weights(0.02), [119], source="b", run_id="r", call=3)
    with pytest.raises(OnlineLogError, match="duplicate weights"):
        log.append(weights(1), [119], source="b", run_id="r")


def test_rejects_malformed_sequence_and_header(tmp_path):
    path = tmp_path / "online_log.csv"
    path.write_text("call,point\n1,1\n", encoding="utf-8")
    with pytest.raises(OnlineLogError, match="unexpected header"):
        OnlineLog(path).rows()


def test_rejects_log_that_does_not_start_at_first_call(tmp_path):
    path = tmp_path / "online_log.csv"
    log = OnlineLog(path)
    log.append(shifted_weights(0.02), [120], source="a", run_id="r", call=1)
    text = path.read_text(encoding="utf-8").replace("1,1,", "2,1,", 1)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(OnlineLogError, match="start at call 1"):
        log.rows()


def test_rejects_more_than_50_points(tmp_path):
    log = OnlineLog(tmp_path / "online_log.csv")
    batch = np.full((51, 21), 1 / 7)
    for index in range(51):
        batch[index, 0] += index / 1000
        batch[index, 1] -= index / 1000
    with pytest.raises(OnlineLogError, match="1 to 50"):
        log.append(batch, np.full(51, 120), source="a", run_id="r")


def test_rejects_invalid_weights_and_scores(tmp_path):
    log = OnlineLog(tmp_path / "online_log.csv")
    with pytest.raises(ValueError):
        log.append(np.zeros((1, 21)), [120], source="a", run_id="r")
    with pytest.raises(OnlineLogError, match="same length"):
        log.append(weights(1), [120, 121], source="a", run_id="r")


def test_log_preserves_exact_weight_round_trip(tmp_path):
    log = OnlineLog(tmp_path / "online_log.csv")
    point = weights(1) + np.arange(21).reshape(1, 21) * 1e-12
    log.append(point, [120.123456789], source="model", run_id="r")
    np.testing.assert_array_equal(log.rows()[0].weights, point[0])
