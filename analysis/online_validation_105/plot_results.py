"""Validate an online run and create its three figures and JSON summaries."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from run_online import DIM, WEIGHT_COLUMNS, duplicate_mask, load_training, valid_weight_mask


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def load_and_validate(run_dir: Path) -> tuple[dict, pd.DataFrame, list[dict], dict]:
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    train_weights, train_actual, _ = load_training(config)
    evaluations_path = run_dir / "evaluations.csv"
    evaluations = pd.read_csv(evaluations_path) if evaluations_path.exists() else pd.DataFrame(columns=WEIGHT_COLUMNS)
    calls = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((run_dir / "calls").glob("call_*.json"))]
    calls = [record for record in calls if record.get("status") in ("ok", "failed")]
    successful = [record for record in calls if record["status"] == "ok"]
    failed = [record for record in calls if record["status"] == "failed"]
    if len(evaluations) != sum(len(record["actual_R"]) for record in successful):
        raise RuntimeError("evaluation rows do not match successful call payloads")
    if len(evaluations):
        weights = evaluations[WEIGHT_COLUMNS].to_numpy(dtype=float)
        if not valid_weight_mask(weights).all() or not np.isfinite(evaluations["actual_R"]).all():
            raise RuntimeError("online evaluations contain invalid weights or targets")
        all_sent = []
        attempts_path = run_dir / "attempts.jsonl"
        if attempts_path.exists():
            with attempts_path.open(encoding="utf-8") as handle:
                all_sent = [row for line in handle if line.strip()
                            for row in (json.loads(line)["weights"] if json.loads(line)["status"] == "started" else [])]
        excluded = np.vstack([train_weights, np.asarray(all_sent).reshape(-1, DIM)]) if all_sent else train_weights
        # Successful points are in the sent set, so check training and within-online uniqueness separately.
        if duplicate_mask(weights, train_weights, config["selection"]["duplicate_atol"]).any():
            raise RuntimeError("an online point duplicates an original training row")
        if len(all_sent) and duplicate_mask(np.asarray(all_sent).reshape(-1, DIM), train_weights,
                                            config["selection"]["duplicate_atol"]).any():
            raise RuntimeError("a sent point duplicates an original training row")
        sent_array = np.asarray(all_sent).reshape(-1, DIM)
        for index in range(len(sent_array)):
            if (np.max(np.abs(sent_array[index + 1:] - sent_array[index]), axis=1) <=
                    config["selection"]["duplicate_atol"]).any():
                raise RuntimeError("previously sent points are not unique")
        for record in successful:
            predicted = np.asarray(record["predicted_mean"], dtype=float)
            actual = np.asarray(record["actual_R"], dtype=float)
            rows = evaluations[evaluations["call"] == record["call"]].sort_values("point")
            if not np.allclose(predicted, rows["predicted_mean"], rtol=0, atol=1e-10):
                raise RuntimeError(f"call {record['call']} predictions differ between logs")
            if not np.allclose(actual, rows["actual_R"], rtol=0, atol=1e-10):
                raise RuntimeError(f"call {record['call']} actual values differ between logs")
    return config, evaluations, calls, {"train_weights": train_weights, "train_actual": train_actual,
                                        "successful": successful, "failed": failed}


def create_outputs(run_dir: Path) -> dict:
    config, evaluations, calls, context = load_and_validate(run_dir)
    successful = context["successful"]
    train_best = float(context["train_actual"].min())
    call_numbers = [record["call"] for record in successful]
    batch_minima = [float(record["batch_min"]) for record in successful]
    running = np.minimum.accumulate(batch_minima).tolist() if batch_minima else []
    metrics = [{key: record.get(key) for key in
                ("call", "train_rows", "calibration", "batch_min", "running_best_online", "metrics",
                 "requested_quotas", "slot_counts")} for record in successful]

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})
    fig, ax = plt.subplots(figsize=(9, 5.2))
    if call_numbers:
        ax.plot(call_numbers, batch_minima, "o-", label="Batch minimum")
        ax.plot(call_numbers, running, "s-", label="Running best (online)")
    ax.axhline(train_best, color="black", linestyle="--", linewidth=1.2,
               label=f"Original training best = {train_best:.6f}")
    ax.set(xlabel="Online call", ylabel="R (lower is better)", title="Independent online R descent")
    ax.set_xticks(call_numbers)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "r_descent.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for key, label in (("mae", "MAE"), ("rmse", "RMSE"), ("bias_predicted_minus_actual", "Bias (predicted - actual)")):
        values = [record["metrics"][key] for record in successful]
        ax.plot(call_numbers, values, "o-", label=label)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Online call", ylabel="Prediction error in R", title="Pre-evaluation prediction error by call")
    ax.set_xticks(call_numbers)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "learning_curve.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    if len(evaluations):
        scatter = ax.scatter(evaluations["actual_R"], evaluations["predicted_mean"],
                             c=evaluations["call"], cmap="viridis", s=22, alpha=0.8, edgecolors="none")
        fig.colorbar(scatter, ax=ax, label="Online call")
        lower = min(float(evaluations["actual_R"].min()), float(evaluations["predicted_mean"].min()))
        upper = max(float(evaluations["actual_R"].max()), float(evaluations["predicted_mean"].max()))
        margin = max((upper - lower) * 0.04, 1e-6)
        ax.plot([lower - margin, upper + margin], [lower - margin, upper + margin], "k--", label="Ideal")
        ax.set_xlim(lower - margin, upper + margin)
        ax.set_ylim(lower - margin, upper + margin)
        ax.legend()
    ax.set(xlabel="Actual R", ylabel="Predicted R (before evaluation)", title="Online prediction vs. actual")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(run_dir / "prediction_vs_actual.png", dpi=180)
    plt.close(fig)

    summary = {
        "experiment": config["experiment"], "run_id": run_dir.name,
        "successful_calls": len(successful), "failed_calls": [record["call"] for record in context["failed"]],
        "successful_online_points": len(evaluations), "expected_points_if_no_failures": config["rounds"] * config["batch_size"],
        "original_training_best_R": train_best,
        "online_best_R": float(evaluations["actual_R"].min()) if len(evaluations) else None,
        "online_best_call": int(evaluations.loc[evaluations["actual_R"].idxmin(), "call"]) if len(evaluations) else None,
        "online_best_point": int(evaluations.loc[evaluations["actual_R"].idxmin(), "point"]) if len(evaluations) else None,
        "calls": metrics,
    }
    if len(evaluations):
        best_row = evaluations.loc[evaluations["actual_R"].idxmin()]
        best = {"R": float(best_row["actual_R"]), "call": int(best_row["call"]), "point": int(best_row["point"]),
                "weights": [float(best_row[name]) for name in WEIGHT_COLUMNS], "source": "this run's online evaluations"}
        write_json(run_dir / "best_weights.json", best)
    write_json(run_dir / "summary.json", summary)
    manifest = {name: sha256_file(run_dir / name) for name in
                ("evaluations.csv", "metrics_by_call.csv", "summary.json", "r_descent.png",
                 "learning_curve.png", "prediction_vs_actual.png") if (run_dir / name).exists()}
    write_json(run_dir / "output_manifest.json", manifest)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="run directory or run id under runs/")
    args = parser.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / "runs" / args.run_dir
    summary = create_outputs(run_dir.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
