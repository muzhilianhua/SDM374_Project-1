import csv
import hashlib
import json

import numpy as np
import pytest

from sdm.online.log import OnlineLog
from analysis.online_results.plot import plot_online_log


def weights(n):
    result = np.full((n, 21), 1 / 7)
    for index in range(n):
        result[index, 0] += (index + 1) / 100
        result[index, 1] -= (index + 1) / 100
    return result


def test_plot_online_log_writes_both_figures(tmp_path):
    log_path = tmp_path / "online_log.csv"
    log = OnlineLog(log_path)
    log.append(weights(2), [120, 119], source="explore", run_id="r1")
    log.append(weights(1) + np.array([[.04, -.04] + [0] * 19]), [118], source="exploit", run_id="r2")
    prediction_path = tmp_path / "predictions.csv"
    with prediction_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["call", "point", "predicted_R"])
        writer.writerows([[1, 1, 121], [1, 2, 118], [2, 1, 117]])
    summary = plot_online_log(log_path, tmp_path / "figures", prediction_path)
    assert summary["best_R"] == 118 and summary["prediction_rows"] == 3
    assert (tmp_path / "figures/online_best_curve.png").is_file()
    assert (tmp_path / "figures/prediction_vs_measurement.png").is_file()
    saved = json.loads((tmp_path / "figures/summary.json").read_text(encoding="utf-8"))
    assert saved == summary
    assert saved["run_ids"] == ["r1", "r2"]
    assert saved["log_sha256"] == hashlib.sha256(log_path.read_bytes()).hexdigest()
    assert saved["prediction_sha256"] == hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    assert saved["per_call"][0]["mae"] == 1
    assert saved["per_call"][0]["bias"] == 0
    assert not saved["beat_offline_baseline"]


@pytest.mark.parametrize("prediction", ["nan", "inf", "-inf"])
def test_prediction_reader_rejects_nonfinite_values(tmp_path, prediction):
    path = tmp_path / "predictions.csv"
    path.write_text("call,point,predicted_R\n1,1," + prediction + "\n", encoding="utf-8")
    from analysis.online_results.plot import read_predictions
    with pytest.raises(ValueError, match="invalid prediction row"):
        read_predictions(path)


def test_plot_rejects_prediction_keys_not_in_log(tmp_path):
    log_path = tmp_path / "online_log.csv"
    OnlineLog(log_path).append(weights(1), [120], source="explore", run_id="r1")
    prediction_path = tmp_path / "predictions.csv"
    prediction_path.write_text(
        "call,point,predicted_R\n1,1,120\n1,2,121\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        plot_online_log(log_path, tmp_path / "figures", prediction_path)
    assert not (tmp_path / "figures").exists()


def test_reference_parameters_and_log_only_summary(tmp_path):
    log_path = tmp_path / "online_log.csv"
    OnlineLog(log_path).append(weights(1), [118], source="test", run_id="synthetic")
    summary = plot_online_log(log_path, tmp_path / "figures",
                              offline_baseline=119, training_best=120)
    assert summary["beat_offline_baseline"]
    assert summary["offline_baseline"] == 119 and summary["training_best"] == 120
    assert summary["prediction_sha256"] is None
