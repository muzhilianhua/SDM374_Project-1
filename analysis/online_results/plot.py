"""Generate report figures from the append-only online log (T26).

The optional prediction file must contain ``call,point,predicted_R``. Actual
scores always come from ``online_log.csv``; predictions are never treated as
measurements.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from sdm.online.log import OnlineLog, OnlineLogError
from sdm.runs import git_state


def read_predictions(path: Path) -> dict[tuple[int, int], float]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        expected = {"call", "point", "predicted_R"}
        if len(reader.fieldnames or ()) != 3 or set(reader.fieldnames or ()) != expected:
            raise ValueError(f"prediction CSV must have columns {sorted(expected)}")
        result = {}
        for index, raw in enumerate(reader, 2):
            if None in raw:
                raise ValueError(f"unexpected fields in prediction row {index}")
            try:
                call, point = int(raw["call"]), int(raw["point"])
                prediction = float(raw["predicted_R"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid prediction row {index}") from error
            if call < 1 or point < 1 or not np.isfinite(prediction):
                raise ValueError(f"invalid prediction row {index}")
            key = (call, point)
            if key in result:
                raise ValueError(f"duplicate prediction key {key}")
            result[key] = prediction
        return result


def plot_online_log(log_path: Path, output: Path, prediction_path: Path | None = None, *,
                    offline_baseline: float = 115.68, training_best: float = 115.76) -> dict:
    rows = OnlineLog(log_path).rows()
    if not rows:
        raise OnlineLogError("online log is empty")
    if not np.isfinite([offline_baseline, training_best]).all():
        raise ValueError("reference thresholds must be finite")
    predictions = read_predictions(prediction_path) if prediction_path is not None else None
    if predictions is not None:
        expected = {(row.call, row.point) for row in rows}
        missing, extra = expected - set(predictions), set(predictions) - expected
        if missing:
            raise ValueError(f"missing prediction for {sorted(missing)[:3]}")
        if extra:
            raise ValueError(f"prediction has unknown keys: {sorted(extra)[:3]}")
    by_call = {}
    for row in rows:
        by_call.setdefault(row.call, []).append(row.R)
    calls = np.array(sorted(by_call))
    batch_min = np.array([min(by_call[call]) for call in calls])
    running_best = np.minimum.accumulate(batch_min)
    run_ids = list(dict.fromkeys(row.run_id for row in rows))
    footer = textwrap.fill("Run IDs: " + ", ".join(run_ids), width=105)

    def save(fig, name):
        # Reserve vertical space for every run id, including long multi-run logs.
        footer_height = 0.16 * len(footer.splitlines()) + 0.12
        width, height = fig.get_size_inches()
        fig.set_size_inches(width, height + footer_height)
        fig.text(0.02, 0.02, footer, fontsize=6, va="bottom")
        fig.tight_layout(rect=(0, footer_height / (height + footer_height), 1, 1))
        fig.savefig(output / name, dpi=180)
        plt.close(fig)

    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(calls, batch_min, "o-", label="batch minimum")
    ax.plot(calls, running_best, "s--", label="running best")
    ax.axhline(offline_baseline, color="tab:red", linestyle=":",
               label=f"offline baseline {offline_baseline:g}")
    ax.axhline(training_best, color="tab:green", linestyle="-.",
               label=f"training best {training_best:g}")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set(xlabel="Evaluator call", ylabel="R", title="Online batch minimum and running best")
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    save(fig, "online_best_curve.png")
    summary = {"calls": int(len(calls)), "points": len(rows), "best_R": float(running_best[-1]),
               "best_call": int(calls[np.argmin(batch_min)]), "run_ids": run_ids,
               "git": git_state(),
               "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
               "prediction_sha256": (hashlib.sha256(prediction_path.read_bytes()).hexdigest()
                                     if prediction_path is not None else None),
               "offline_baseline": float(offline_baseline), "training_best": float(training_best),
               "beat_offline_baseline": bool(running_best[-1] < offline_baseline),
               "per_call": [{"call": int(call), "points": len(by_call[call]),
                             "batch_min": float(value), "running_best": float(best)}
                            for call, value, best in zip(calls, batch_min, running_best)]}
    if predictions is not None:
        actual, predicted = [], []
        for row in rows:
            key = (row.call, row.point)
            actual.append(row.R); predicted.append(predictions[key])
        actual, predicted = np.asarray(actual), np.asarray(predicted)
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        colors = ax.scatter(predicted, actual, c=[row.call for row in rows],
                            s=14, alpha=0.7, cmap="viridis")
        colorbar = fig.colorbar(colors, ax=ax, label="Evaluator call")
        colorbar.locator = MaxNLocator(integer=True)
        colorbar.update_ticks()
        lo, hi = float(min(actual.min(), predicted.min())), float(max(actual.max(), predicted.max()))
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set(xlabel="Predicted R", ylabel="Measured R", title="Prediction versus measurement")
        ax.grid(alpha=0.25); fig.tight_layout()
        save(fig, "prediction_vs_measurement.png")
        summary.update({"prediction_rows": len(predicted), "mae": float(np.mean(np.abs(predicted - actual))),
                        "bias": float(np.mean(predicted - actual))})
        for record in summary["per_call"]:
            mask = np.array([row.call == record["call"] for row in rows])
            errors = predicted[mask] - actual[mask]
            record.update(mae=float(np.mean(np.abs(errors))), bias=float(np.mean(errors)))
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n",
                                         encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--out", type=Path, default=Path("analysis/online_results/results"))
    parser.add_argument("--offline-baseline", type=float, default=115.68)
    parser.add_argument("--training-best", type=float, default=115.76)
    args = parser.parse_args()
    summary = plot_online_log(args.log, args.out, args.predictions,
                              offline_baseline=args.offline_baseline, training_best=args.training_best)
    print(summary)


if __name__ == "__main__":
    main()
