"""Summarize recorded CLI evidence, cross-score candidates and export a traceable study package."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform

from sdm.data import load_dataset, sha256_of
from sdm.evaluate import median_spacing, nearest_distances
from sdm.frozen_v32 import load_outer_splits
from sdm.paths import REPO_ROOT
from sdm.runs import PACKAGES, git_state, resolve_run
from sdm.simplex import check_weights

import numpy as np
import pandas as pd


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def rounded(points):
    rows = check_weights(points).reshape(-1, 7)
    units = np.rint(rows * 1e8).astype(np.int64)
    units[np.arange(len(rows)), rows.argmax(1)] += 100_000_000 - units.sum(1)
    return check_weights((units / 1e8).reshape(-1, 21))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference-histgb", default="20261029-125052-optimize-histgb-v32")
    parser.add_argument("--reference-xgboost", default="20261029-125052-optimize-xgboost-v32")
    parser.add_argument("--reference-extratrees", default="20261029-125052-optimize-extratrees-default")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    dataset = load_dataset(REPO_ROOT / "dataset_same8")
    splits = load_outer_splits(dataset)
    cases = json.loads(args.manifest.read_text())
    inputs = [args.manifest]
    evidence, scores, folds, bands, full, models, params = {}, [], [], [], {}, {}, {}
    for case in cases:
        label = f"{case['family']}-{case['strategy']}"
        erun, trun = resolve_run(case["evaluate"]), resolve_run(case["train"])
        for run_id in [case["evaluate"], case["train"], *case["optimize"]]:
            run = resolve_run(run_id)
            meta = run.read_json("meta.json")
            if meta["dataset"]["sha256"] != dataset.sha256 or meta["git"]["dirty"]:
                raise ValueError(f"unclean or wrong-dataset evidence: {run_id}")
            evidence[run_id] = {"meta": meta, "metrics": run.read_json("metrics.json"),
                                "config": run.config_path.read_text()}
            inputs.extend([run.path / "meta.json", run.path / "metrics.json", run.config_path])
        scores.append({"label": label, **erun.read_json("metrics.json"), "run": erun.id})
        frame = pd.read_csv(erun.path / "fold_scores.csv")
        frame.insert(0, "label", label)
        folds.append(frame)
        pred = pd.read_csv(erun.path / "predictions.csv")
        # Preserve the complete outer predictions for rescoring on a fresh clone.
        pred.to_csv(args.out / f"outer_predictions-{label}.csv.gz", index=False, compression="gzip")
        inputs.extend([erun.path / "fold_scores.csv", erun.path / "predictions.csv", trun.path / "model.pkl"])
        indexed = pred.set_index(["split_id", "candidate_id"])
        for split in splits:
            part = indexed.loc[split.split_id].loc[np.asarray(dataset.ids)[split.test]]
            distance = nearest_distances(dataset.X[split.test], dataset.X[split.train])
            thirds = np.array_split(np.argsort(distance, kind="stable"), 3)
            for band, indices in zip(("near", "middle", "far"), thirds):
                p = part.iloc[indices]
                error = np.abs(p.predicted_mean - p.actual)
                low = p.actual <= 132.8
                bands.append({"label": label, "split_id": split.split_id, "band": band,
                    "n": len(p), "low_n": int(low.sum()), "mae": float(error.mean()),
                    "low_mae": float(error[low].mean()) if low.any() else None,
                    "coverage": float((error <= p.predicted_std).mean()),
                    "distance_min": float(distance[indices].min()), "distance_max": float(distance[indices].max())})
        model = trun.load_model()
        models[label] = model
        full[label] = model.metrics
        estimator_params = {key: ("NaN" if isinstance(value, float) and not np.isfinite(value) else value)
                            for key, value in model.estimator.get_params().items()}
        params[label] = {"run": trun.id, "selected": model.selected_, "inner_trials": model.trials_,
                         "estimator": estimator_params, "metrics": model.metrics}

    summary = pd.DataFrame(scores).set_index("label")
    summary.to_csv(args.out / "model_scores.csv")
    fold_table = pd.concat(folds, ignore_index=True)
    fold_table.to_csv(args.out / "fold_scores.csv", index=False)
    pd.DataFrame(bands).to_csv(args.out / "distance_diagnostics.csv", index=False)
    selection, paired = {}, []
    for family in ("histgb", "xgboost"):
        base, soft = summary.loc[f"{family}-unweighted"], summary.loc[f"{family}-soft"]
        passes = bool(soft.v32_score >= base.v32_score and soft.mae <= 1.05 * base.mae
                      and soft.v32_local_mae <= 1.10 * base.v32_local_mae)
        selection[family] = {"soft_passes_outer_guard": passes,
            "soft_vs_uniform_choice": f"{family}-soft" if passes else f"{family}-unweighted",
            "highest_mean_full_support_workflow": max(
                [f"{family}-{s}" for s in ("unweighted", "step", "soft")],
                key=lambda label: summary.loc[label, "v32_score"]),
            "soft_minus_uniform_score": float(soft.v32_score - base.v32_score)}
        for reference in ("unweighted", "step", "trimmed-step"):
            left = fold_table[fold_table.label == f"{family}-soft"].set_index("split_id")
            right = fold_table[fold_table.label == f"{family}-{reference}"].set_index("split_id")
            delta = left.score - right.score
            paired.append({"family": family, "reference": reference,
                "mean_score_delta": float(delta.mean()), "score_delta_sd": float(delta.std(ddof=1)),
                "improved_folds": int((delta > 0).sum()), "tied_folds": int((delta == 0).sum()),
                "repeat1_delta": float(delta.iloc[:5].mean()), "repeat2_delta": float(delta.iloc[5:].mean()),
                "deltas": delta.to_dict()})
    write_json(args.out / "model_selection.json", selection)
    write_json(args.out / "paired_comparisons.json", paired)
    write_json(args.out / "full_data_parameters.json", params)

    # Fixed references go first so duplicate proposals cannot displace their provenance.
    points, entries = [], []
    for label, path in [("incumbent", REPO_ROOT / "analysis/offline_submission/submission/weights.json"),
                        ("previous_challenger", REPO_ROOT / "analysis/robust_search/delivery/challenger_weights.json")]:
        points.append(np.asarray(json.loads(path.read_text())).reshape(21))
        entries.append({"label": label, "source_run": "tracked-reference", "rank": 0})
        inputs.append(path)
    points.append(dataset.X[dataset.y.argmin()])
    entries.append({"label": "best_observed", "source_run": "dataset:p01969", "rank": 0})
    for case in cases:
        label = f"{case['family']}-{case['strategy']}"
        for run_id in case["optimize"]:
            run = resolve_run(run_id)
            for rank, row in enumerate(run.read_json("proposals.json"), 1):
                points.append(np.asarray(row["weights"]).reshape(21))
                entries.append({"label": label, "source_run": run.id, "rank": rank})
            inputs.append(run.path / "proposals.json")
    points = rounded(np.asarray(points))
    _, first = np.unique(points, axis=0, return_index=True)
    first = np.sort(first)
    points = points[first]
    table = pd.DataFrame([entries[i] for i in first])
    distance = nearest_distances(points, dataset.X)
    table["distance_ratio"] = distance / median_spacing(dataset.X)
    table["eligible"] = (distance > 1e-6) & (table.distance_ratio <= .75)
    for family in ("histgb", "xgboost"):
        for strategy in ("unweighted", "step", "trimmed-step", "soft"):
            label = f"{family}-{strategy}"
            table[label + "_mean"], table[label + "_std"] = models[label].predict(points)
    judges = []
    for family in ("histgb", "xgboost"):
        label = family + "-soft"
        judges.append(label)
        run = resolve_run(getattr(args, "reference_" + family))
        key = "original-" + family
        table[key + "_mean"], table[key + "_std"] = run.load_model().predict(points)
        judges.append(key)
        inputs.extend([run.path / "model.pkl", run.path / "meta.json"])
    extra_run = resolve_run(args.reference_extratrees)
    table["extratrees_mean"], table["extratrees_std"] = extra_run.load_model().predict(points)
    inputs.extend([extra_run.path / "model.pkl", extra_run.path / "meta.json"])
    for ref in (args.reference_histgb, args.reference_xgboost, args.reference_extratrees):
        run = resolve_run(ref)
        meta = run.read_json("meta.json")
        if meta["dataset"]["sha256"] != dataset.sha256 or meta["git"]["dirty"]:
            raise ValueError("reference model must have clean, matching-dataset provenance")
        evidence[run.id] = {"meta": meta, "metrics": run.read_json("metrics.json"),
                            "config": run.config_path.read_text()}
    table["joint_risk"] = np.maximum.reduce([table[j + "_mean"] + table[j + "_std"] for j in judges])
    chosen = int(table[table.eligible].sort_values(["joint_risk", "distance_ratio"], kind="stable").index[0])
    table.to_csv(args.out / "candidate_scores.csv", index_label="point_index")
    write_json(args.out / "candidate_weights.json", points.reshape(-1, 3, 7).tolist())
    write_json(args.out / "study_weights.json", points[chosen].reshape(3, 7).tolist())
    rng = np.random.default_rng(3127)
    draws = rng.gamma(2000 * points[chosen].reshape(1, 3, 7) + .01, size=(128, 3, 7))
    nearby = (draws / draws.sum(-1, keepdims=True)).reshape(-1, 21)
    sensitivity = {"seed": 3127, "n": 128, "concentration": 2000, "judges": {}}
    for label in ("histgb-soft", "xgboost-soft"):
        mean, _ = models[label].predict(nearby)
        sensitivity["judges"][label] = dict(zip(["q10", "median", "q90"], np.quantile(mean, [.1,.5,.9]).tolist()))
    write_json(args.out / "local_sensitivity.json", sensitivity)
    write_json(args.out / "candidate_selection.json", {"index": chosen, "judges": judges,
        "source": table.loc[chosen].to_dict(), "incumbent": table.iloc[0].to_dict(),
        "model_guard": selection, "measured_score": None,
        "status": "Exploratory study candidate; original delivery is preserved pending real evaluation."})
    write_json(args.out / "run_evidence.json", evidence)
    write_json(args.out / "provenance.json", {"git": git_state(), "python": platform.python_version(),
        "packages": {p: version(p) for p in PACKAGES}, "dataset_sha256": dataset.sha256,
        "manifest": cases, "input_hashes": {str(p.resolve().relative_to(REPO_ROOT)): sha256_of(p) for p in inputs},
        "limitation": "Reused frozen benchmark and surrogate-only candidate search; no simulator measurements."})
    print(summary[["v32_score", "mae", "v32_local_mae", "v32_top1_regret", "std_coverage"]].to_string())
    print(json.dumps(selection, indent=2))
    print(table.loc[[0, 1, chosen], ["label", "joint_risk", "distance_ratio", "extratrees_mean"]].to_string())


if __name__ == "__main__":
    main()
