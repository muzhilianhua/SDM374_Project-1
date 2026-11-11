"""Generate report figures from the append-only online log (T26).

The optional prediction file must contain ``call,point,predicted_R``. Actual
scores always come from ``online_log.csv``; predictions are never treated as
measurements.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sdm.online.log import OnlineLog, OnlineLogError


def read_predictions(path: Path) -> dict[tuple[int, int], float]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        expected = {"call", "point", "predicted_R"}
        if set(reader.fieldnames or ()) != expected:
            raise ValueError(f"prediction CSV must have columns {sorted(expected)}")
        result = {}
        for raw in reader:
            key = (int(raw["call"]), int(raw["point"]))
            if key in result:
                raise ValueError(f"duplicate prediction key {key}")
            result[key] = float(raw["predicted_R"])
        return result


def plot_online_log(log_path: Path, output: Path, prediction_path: Path | None = None) -> dict:
    rows = OnlineLog(log_path).rows()
    if not rows:
        raise OnlineLogError("online log is empty")
    output.mkdir(parents=True, exist_ok=True)
    by_call = {}
    for row in rows:
        by_call.setdefault(row.call, []).append(row.R)
    calls = np.array(sorted(by_call))
    batch_min = np.array([min(by_call[call]) for call in calls])
    running_best = np.minimum.accumulate(batch_min)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(calls, batch_min, "o-", label="batch minimum")
    ax.plot(calls, running_best, "s--", label="running best")
    ax.set(xlabel="Evaluator call", ylabel="R", title="Online batch minimum and running best")
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(output / "online_best_curve.png", dpi=180); plt.close(fig)
    summary = {"calls": int(len(calls)), "points": len(rows), "best_R": float(running_best[-1]),
               "best_call": int(calls[np.argmin(batch_min)])}
    if prediction_path is not None:
        predictions = read_predictions(prediction_path)
        actual, predicted = [], []
        for row in rows:
            key = (row.call, row.point)
            if key not in predictions:
                raise ValueError(f"missing prediction for call {key[0]}, point {key[1]}")
            actual.append(row.R); predicted.append(predictions[key])
        actual, predicted = np.asarray(actual), np.asarray(predicted)
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        ax.scatter(predicted, actual, s=14, alpha=0.7)
        lo, hi = float(min(actual.min(), predicted.min())), float(max(actual.max(), predicted.max()))
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set(xlabel="Predicted R", ylabel="Measured R", title="Prediction versus measurement")
        ax.grid(alpha=0.25); fig.tight_layout()
        fig.savefig(output / "prediction_vs_measurement.png", dpi=180); plt.close(fig)
        summary.update({"prediction_rows": len(predicted), "mae": float(np.mean(np.abs(predicted - actual))),
                        "bias": float(np.mean(predicted - actual))})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--out", type=Path, default=Path("analysis/online_results/delivery"))
    args = parser.parse_args()
    summary = plot_online_log(args.log, args.out, args.predictions)
    print(summary)


if __name__ == "__main__":
    main()
