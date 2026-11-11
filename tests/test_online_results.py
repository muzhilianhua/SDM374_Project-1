import csv

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
