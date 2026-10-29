"""Select a submission from CLI proposals using a fixed two-model pessimistic consensus rule.

This is a diagnostic study, not a new evaluation protocol or a real-controller score.
All proposal generation and model evaluation must already exist as recorded CLI runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from sdm.data import load_dataset, sha256_of
from sdm.evaluate import median_spacing, nearest_distances
from sdm.frozen_v32 import load_outer_splits
from sdm.paths import REPO_ROOT
from sdm.runs import PACKAGES, git_state, resolve_run
from sdm.simplex import check_weights
from importlib.metadata import version

JUDGES = ("histgb", "xgboost")
DISTANCE_LIMIT = 0.75


def rounded_weights(points):
    """Eight-decimal JSON weights with a sum correction on the largest entry in each building."""
    rows = np.round(check_weights(points).reshape(-1, 7), 8)
    rows[np.arange(len(rows)), rows.argmax(axis=1)] += 1 - rows.sum(axis=1)
    return check_weights(rows.reshape(-1, 21))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def select(proposal_refs, judge_refs, evaluation_refs, out):
    dataset = load_dataset(REPO_ROOT / "dataset_same8")
    spacing = median_spacing(dataset.X)
    judges = {name: resolve_run(ref) for name, ref in judge_refs.items()}
    models = {name: run.load_model() for name, run in judges.items()}
    proposals = [resolve_run(ref) for ref in proposal_refs]
    all_sources = [*proposals, *judges.values(), *map(resolve_run, evaluation_refs.values())]
    for run in all_sources:
        if run.read_json("meta.json")["dataset"]["sha256"] != dataset.sha256:
            raise ValueError(f"{run.id}: dataset mismatch")
    rows, points = [], []
    for run in proposals:
        for rank, proposal in enumerate(run.read_json("proposals.json"), 1):
            rows.append({"run": run.id, "rank": rank, "kind": "proposal"})
            points.append(np.asarray(proposal["weights"]).reshape(21))
    rows.append({"run": "dataset:" + dataset.ids[int(dataset.y.argmin())], "rank": 1, "kind": "observed"})
    points.append(dataset.X[dataset.y.argmin()])
    raw = check_weights(np.asarray(points))
    weights = rounded_weights(raw)
    table = pd.DataFrame(rows)
    distance = nearest_distances(weights, dataset.X)
    table["distance_ratio"] = distance / spacing
    for name, model in models.items():
        mean, std = model.predict(weights)
        table[name + "_mean"], table[name + "_std"] = mean, std
    table["selection_score"] = np.maximum.reduce([
        table[name + "_mean"] + table[name + "_std"] for name in JUDGES])
    table["boosted_disagreement"] = abs(table.histgb_mean - table.xgboost_mean)
    table["eligible"] = ((table.kind == "proposal") & (distance > 1e-6)
                         & (table.distance_ratio <= DISTANCE_LIMIT))
    # Fixed tie breaks; no actual label from a proposed point is available or used.
    ordered = table[table.eligible].sort_values(
        ["selection_score", "distance_ratio", "run", "rank"], kind="stable")
    if ordered.empty:
        raise ValueError("no novel candidates pass the distance limit")
    best = int(ordered.index[0])
    out.mkdir(parents=True, exist_ok=False)
    table.to_csv(out / "candidate_comparison.csv", index=False)
    write_json(out / "weights.json", weights[best].reshape(3, 7).tolist())
    write_json(out / "best_observed_reference.json", weights[-1].reshape(3, 7).tolist())
    # Probe nearby simplex points, using a fixed seed; these are sensitivity checks, not truth.
    rng = np.random.default_rng(2026)
    draws = rng.gamma(2000 * weights[best].reshape(1, 3, 7) + 0.01, size=(128, 3, 7))
    nearby = (draws / draws.sum(axis=2, keepdims=True)).reshape(-1, 21)
    sensitivity = {}
    for name in JUDGES:
        predictions = models[name].predict(nearby)[0]
        sensitivity[name] = {"median": float(np.median(predictions)),
                             "q10": float(np.quantile(predictions, 0.1)),
                             "q90": float(np.quantile(predictions, 0.9))}
    sensitivity["median_perturbation_distance"] = float(np.median(np.linalg.norm(nearby - weights[best], axis=1)))

    # Check the fixed selection rule on existing frozen held-out predictions. Reused folds are
    # selection-aware diagnostics, not independent evidence for a new optimized weight vector.
    evaluations = {name: resolve_run(ref) for name, ref in evaluation_refs.items()}
    prediction_tables = {name: pd.read_csv(run.path / "predictions.csv") for name, run in evaluations.items()}
    diagnostics = []
    for split in load_outer_splits(dataset):
        ids = np.asarray(dataset.ids)[split.test]
        held = {name: t[t.split_id == split.split_id].set_index("candidate_id").loc[ids]
                for name, t in prediction_tables.items()}
        ratio = nearest_distances(dataset.X[split.test], dataset.X[split.train]) / median_spacing(dataset.X[split.train])
        eligible = ratio <= DISTANCE_LIMIT
        values = {name: held[name].predicted_mean.to_numpy() + held[name].predicted_std.to_numpy()
                  for name in JUDGES}
        values["consensus"] = np.maximum(values["histgb"], values["xgboost"])
        for name, acquisition in values.items():
            if not eligible.any():
                continue
            chosen = int(np.argmin(np.where(eligible, acquisition, np.inf)))
            actual = float(dataset.y[split.test[chosen]])
            diagnostics.append({"split": split.split_id, "rule": name, "candidate_id": ids[chosen],
                                "actual": actual, "regret_vs_full_test_best": actual - float(dataset.y[split.test].min()),
                                "eligible_count": int(eligible.sum())})
    diagnostic_table = pd.DataFrame(diagnostics)
    diagnostic_table.to_csv(out / "selection_diagnostics.csv", index=False)
    selection = table.loc[best].to_dict()
    selection.update({"source_index_zero_based": int(table.loc[best, "rank"]) - 1,
                      "raw_to_export_max_abs_change": float(np.max(np.abs(weights[best] - raw[best]))),
                      "best_observed_actual": float(dataset.y.min()),
                      "measured_score": None, "sensitivity": sensitivity,
                      "selection_rule": "min max(histgb mean + std, xgboost mean + std), distance_ratio <= 0.75",
                      "warning": "Heuristic risk penalties, not calibrated confidence bounds for optimized points. "
                                 "Model agreement is not independent validation. ExtraTrees is an additional diagnostic."})
    write_json(out / "selection.json", selection)
    provenance = {"git": git_state(), "python": platform.python_version(),
                  "packages": {name: version(name) for name in PACKAGES},
                  "dataset_sha256": dataset.sha256, "seed": 2026,
                  "proposal_runs": proposal_refs, "judge_runs": judge_refs, "evaluate_runs": evaluation_refs,
                  "inputs": {str(run.path / filename): sha256_of(run.path / filename)
                             for run in all_sources for filename in ("meta.json", "metrics.json", "proposals.json", "predictions.csv", "model.pkl")
                             if (run.path / filename).exists()},
                  "weights_sha256": hashlib.sha256((out / "weights.json").read_bytes()).hexdigest()}
    write_json(out / "provenance.json", provenance)
    summary = table.groupby("run").agg(candidates=("rank", "size"), eligible=("eligible", "sum"),
                                       best_consensus_score=("selection_score", "min"))
    summary.to_csv(out / "method_summary.csv")
    print(json.dumps(selection, indent=2))
    print(diagnostic_table.groupby("rule")["regret_vs_full_test_best"].mean().to_string())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", nargs="+", required=True)
    parser.add_argument("--judges", nargs="+", required=True, help="name=run-id (include histgb and xgboost)")
    parser.add_argument("--evaluations", nargs="+", required=True, help="histgb=run-id xgboost=run-id")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    select(args.proposals, dict(item.split("=", 1) for item in args.judges),
           dict(item.split("=", 1) for item in args.evaluations), args.out)


if __name__ == "__main__":
    main()
