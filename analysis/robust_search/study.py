"""Screen a frozen CLI candidate pool, then separately audit one selected challenger.

Training subsample stress tests use no simulator and provide no new measured labels.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform

from sdm.data import load_dataset, sha256_of
from sdm.evaluate import median_spacing, nearest_distances
from sdm.paths import REPO_ROOT
from sdm.registry import build_model
from sdm.runs import PACKAGES, git_state, resolve_run
from sdm.simplex import check_weights

import numpy as np
import pandas as pd

MEMBERS = ("histgb", "xgboost")
SEEDS = {"screen": tuple(range(27101, 27106)), "audit": tuple(range(28101, 28106))}
INCUMBENT = REPO_ROOT / "analysis/offline_submission/submission/weights.json"
BASELINE_RUN = "20261029-125052-optimize-baseline"
XGBOOST_RUN = "20261029-125052-optimize-xgboost-v32"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def export_weights(points):
    rows = check_weights(points).reshape(-1, 7)
    units = np.rint(rows * 100_000_000).astype(np.int64)
    units[np.arange(len(units)), rows.argmax(axis=1)] += 100_000_000 - units.sum(axis=1)
    return check_weights((units / 100_000_000).reshape(-1, 21))


def panel(dataset, points, phase):
    records, subsets = [], {}
    for seed in SEEDS[phase]:
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(len(dataset.y), size=int(0.9 * len(dataset.y)), replace=False))
        subsets[str(seed)] = [dataset.ids[i] for i in rows]
        predictions = {}
        for name in MEMBERS:
            model = build_model(name, {"seed": 2026}).fit(dataset.X[rows], dataset.y[rows])
            predictions[name] = model.predict(points)
        for i in range(len(points)):
            records.append({"point_index": i, "seed": seed,
                            **{name + "_mean": float(predictions[name][0][i]) for name in MEMBERS},
                            **{name + "_std": float(predictions[name][1][i]) for name in MEMBERS},
                            "score": max(float(predictions[name][0][i] + predictions[name][1][i])
                                         for name in MEMBERS)})
        print(f"{phase}: completed subsample seed {seed}", flush=True)
    return pd.DataFrame(records), subsets


def provenance(dataset, phase, inputs, subsets):
    return {"git": git_state(), "phase": phase, "python": platform.python_version(),
            "packages": {name: version(name) for name in PACKAGES},
            "dataset_sha256": dataset.sha256, "subsample_fraction": 0.9,
            "subsample_seeds": list(SEEDS[phase]), "model_seed": 2026,
            "subsample_candidate_ids": subsets,
            "inputs": {str(Path(p).resolve()): sha256_of(Path(p)) for p in inputs},
            "caveat": "Overlapping training subsets from the same dataset; model sensitivity only, not new truth."}


def screen(run_refs, committee_ref, out):
    dataset = load_dataset(REPO_ROOT / "dataset_same8")
    run = resolve_run(committee_ref)
    if run.read_json("meta.json")["dataset"]["sha256"] != dataset.sha256:
        raise ValueError("committee dataset mismatch")
    committee = run.load_model()
    if set(committee.members) != set(MEMBERS):
        raise ValueError("expected the HistGB/XGBoost committee")
    points = [np.asarray(json.loads(INCUMBENT.read_text())).reshape(21)]
    entries = [{"label": "incumbent", "source_run": "20261029-125052-optimize-histgb-v32",
                "source_rank": 16, "selectable": True}]
    inputs = [INCUMBENT, run.path / "model.pkl", run.path / "meta.json"]
    for ref, label in [(BASELINE_RUN, "baseline_prediction_min"), (XGBOOST_RUN, "xgboost_prediction_min")]:
        source = resolve_run(ref)
        points.append(np.asarray(source.read_json("proposals.json")[0]["weights"]).reshape(21))
        entries.append({"label": label, "source_run": ref, "source_rank": 1, "selectable": False})
        inputs.append(source.path / "proposals.json")
    points.append(dataset.X[dataset.y.argmin()])
    entries.append({"label": "best_observed", "source_run": "dataset:p01969", "source_rank": 1, "selectable": False})
    for ref in run_refs:
        source = resolve_run(ref)
        if source.read_json("meta.json")["dataset"]["sha256"] != dataset.sha256:
            raise ValueError("proposal dataset mismatch")
        for rank, proposal in enumerate(source.read_json("proposals.json"), 1):
            points.append(np.asarray(proposal["weights"]).reshape(21))
            entries.append({"label": "joint_search", "source_run": source.id,
                            "source_rank": rank, "selectable": True})
        inputs.extend([source.path / "proposals.json", source.path / "meta.json"])
    points = export_weights(np.asarray(points))
    # Preserve incumbent and fixed references first; duplicates across search runs are one candidate.
    _, first = np.unique(points, axis=0, return_index=True)
    first = np.sort(first)
    points, entries = points[first], [entries[i] for i in first]
    table = pd.DataFrame(entries)
    table["distance_ratio"] = nearest_distances(points, dataset.X) / median_spacing(dataset.X)
    table["eligible"] = (table.selectable & (table.distance_ratio <= 0.75)
                         & (nearest_distances(points, dataset.X) > 1e-6))
    for name, member in committee.members.items():
        mean, std = member.predict(points)
        table[name + "_full_mean"], table[name + "_full_std"] = mean, std
    table["full_score"] = np.maximum.reduce([
        table[name + "_full_mean"] + table[name + "_full_std"] for name in MEMBERS])
    result, subsets = panel(dataset, points, "screen")
    grouped = result.groupby("point_index")
    table["screen_q90"] = grouped.score.quantile(0.9)
    table["screen_mean"] = grouped.score.mean()
    eligible = table[table.eligible]
    chosen = int(eligible.sort_values(["screen_q90", "full_score", "distance_ratio"], kind="stable").index[0])
    out.mkdir(parents=True, exist_ok=False)
    table.to_csv(out / "candidate_summary.csv", index_label="point_index")
    result.to_csv(out / "screen_predictions.csv", index=False)
    write_json(out / "candidate_weights.json", points.reshape(-1, 3, 7).tolist())
    selection = {"analysis_id": out.name, "chosen_index": chosen, "incumbent_index": 0,
                 "committee_run": run.id, "source": table.loc[chosen].to_dict(),
                 "candidate_count": len(table), "eligible_count": int(table.eligible.sum()),
                 "audit_seeds_frozen_before_audit": list(SEEDS["audit"])}
    write_json(out / "selection.json", selection)
    write_json(out / "provenance.json", provenance(dataset, "screen", inputs, subsets))
    print(json.dumps(selection, indent=2), flush=True)


def passes_replacement_gate(full_challenger, full_incumbent, screen_challenger, screen_incumbent, audit_delta):
    delta = np.asarray(audit_delta, dtype=float)
    return bool(len(delta) == 5 and np.isfinite(delta).all() and full_challenger < full_incumbent
                and screen_challenger < screen_incumbent and np.all(delta < 0) and delta.mean() <= -1.0)


def audit(screen_dir, out):
    dataset = load_dataset(REPO_ROOT / "dataset_same8")
    source = json.loads((screen_dir / "selection.json").read_text())
    source_prov = json.loads((screen_dir / "provenance.json").read_text())
    if source_prov["dataset_sha256"] != dataset.sha256 or source["audit_seeds_frozen_before_audit"] != list(SEEDS["audit"]):
        raise ValueError("audit dataset or predefined seed mismatch")
    points = check_weights(json.loads((screen_dir / "candidate_weights.json").read_text()))
    table = pd.read_csv(screen_dir / "candidate_summary.csv", index_col="point_index")
    chosen = source["chosen_index"]
    indices = list(dict.fromkeys([0, chosen, *table.index[~table.selectable].tolist()]))
    shortlist = points[indices]
    predictions, subsets = panel(dataset, shortlist, "audit")
    predictions["original_index"] = predictions.point_index.map(dict(enumerate(indices)))
    predictions["label"] = predictions.original_index.map(table.label)
    predictions.loc[predictions.original_index == chosen, "label"] = "challenger" if chosen != 0 else "incumbent"
    pivot = predictions.pivot(index="seed", columns="original_index", values="score")
    delta = (pivot[chosen] - pivot[0]).to_numpy()
    replace = passes_replacement_gate(table.loc[chosen, "full_score"], table.loc[0, "full_score"],
                                     table.loc[chosen, "screen_q90"], table.loc[0, "screen_q90"], delta)
    recommended = chosen if replace else 0
    out.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(out / "audit_predictions.csv", index=False)
    summary = predictions.groupby(["original_index", "label"]).agg(
        score_mean=("score", "mean"), score_q90=("score", lambda x: x.quantile(0.9)),
        histgb_mean=("histgb_mean", "mean"), xgboost_mean=("xgboost_mean", "mean"))
    summary.to_csv(out / "audit_summary.csv")
    for filename, index in [("recommended_weights.json", recommended), ("challenger_weights.json", chosen)]:
        write_json(out / filename, points[index].reshape(3, 7).tolist())
    decision = {"analysis_id": out.name, "screen_analysis_id": source["analysis_id"],
                "replace_incumbent": replace, "chosen_index": chosen, "recommended_index": recommended,
                "challenger_source": table.loc[chosen].to_dict(), "incumbent": table.loc[0].to_dict(),
                "audit_challenger_minus_incumbent": delta.tolist(),
                "audit_mean_delta": float(delta.mean()), "measured_score": None,
                "gate": "better full-data and screening scores; all five audit paired deltas < 0; mean delta <= -1",
                "limitation": "Training subsample stress test only. No real measurements or independent validation."}
    write_json(out / "decision.json", decision)
    inputs = [screen_dir / name for name in ["selection.json", "candidate_weights.json", "candidate_summary.csv", "provenance.json"]]
    write_json(out / "provenance.json", provenance(dataset, "audit", inputs, subsets))
    print(json.dumps(decision, indent=2), flush=True)
    print(summary.to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="phase", required=True)
    a = commands.add_parser("screen")
    a.add_argument("--runs", nargs="+", required=True)
    a.add_argument("--committee", required=True)
    a.add_argument("--out", type=Path, required=True)
    b = commands.add_parser("audit")
    b.add_argument("--screen-dir", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "screen":
        screen(args.runs, args.committee, args.out)
    else:
        audit(args.screen_dir, args.out)


if __name__ == "__main__":
    main()
