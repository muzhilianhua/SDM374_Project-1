"""Does the optimizer's result depend on the surrogate it searches? A model x optimizer study.

Three parts, all written to one results folder with provenance:
  compatibility.csv   which (model, optimizer) pairs can run at all (pgd needs gradients)
  cross_judge.csv     W* found on the real data, scored by every model, with distance to the data
  semi_synthetic.csv  a fitted model plays the "true" function; labels are truth + resampled residuals,
                      so the true value of any W* is known and regret can be measured exactly

Usage: uv run python analysis/coupling/study.py [--seeds 3] [--out DIR]
"""
from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime
from importlib import metadata
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sdm.data import load_dataset
from sdm.evaluate import median_spacing, nearest_distances, out_of_fold
from sdm.paths import REPO_ROOT
from sdm.registry import build_model, build_optimizer, models, optimizers
from sdm.runs import PACKAGES, git_state

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_TRUTHS = ("ridge2", "extratrees")


def _build(builder, name: str, seed: int, overrides: dict | None):
    """Build a plugin with its defaults, its seed set to `seed` when it has one, plus overrides."""
    registry_table = models() if builder is build_model else optimizers()
    params = dict(overrides or {})
    if "seed" in {field.name for field in fields(registry_table[name].Params)}:
        params.setdefault("seed", seed)
    return builder(name, params)


def _propose(model, optimizer_name, seed, X, y, optimizer_params):
    """W* (21,) or None when the optimizer needs something the model does not provide."""
    optimizer = _build(build_optimizer, optimizer_name, seed, optimizer_params.get(optimizer_name))
    try:
        return optimizer.propose(model, X, y, n=1)[0], ""
    except NotImplementedError as error:
        return None, str(error)


def run_study(*, model_names, optimizer_names, truth_names, seeds, optimizer_params=None, base_seed=2026):
    optimizer_params = optimizer_params or {}
    dataset = load_dataset(REPO_ROOT / "dataset_same8")
    X, y = dataset.X, dataset.y
    spacing = median_spacing(X)
    pairs = list(itertools.product(model_names, optimizer_names))

    # Cross-judging on the real table: every model scores every W*.
    judge_rows = []
    for repeat in range(seeds):
        seed = base_seed + repeat
        fitted = {name: _build(build_model, name, seed, None).fit(X, y) for name in model_names}
        for model_name, optimizer_name in pairs:
            w, reason = _propose(fitted[model_name], optimizer_name, seed, X, y, optimizer_params)
            row = {"seed": seed, "model": model_name, "optimizer": optimizer_name, "compatible": w is not None,
                   "reason": reason}
            if w is not None:
                row["distance_ratio"] = float(nearest_distances(w[None], X)[0] / spacing)
                row.update({f"judged_by_{judge}": float(m.predict(w[None])[0][0]) for judge, m in fitted.items()})
                row["weights"] = json.dumps(np.round(w, 6).tolist())
            judge_rows.append(row)
    cross_judge = pd.DataFrame(judge_rows)
    compatibility = (cross_judge.groupby(["model", "optimizer"], sort=False)
                     .agg(compatible=("compatible", "all"), reason=("reason", "first")).reset_index())

    # Semi-synthetic benchmark: known truth, realistic noise, exact regret.
    synthetic_rows = []
    for truth_name in truth_names:
        truth = _build(build_model, truth_name, base_seed, None).fit(X, y)
        truth_on_table = truth.predict(X)[0]
        oof_mean, _ = out_of_fold(lambda: _build(build_model, truth_name, base_seed, None), X, y)
        residuals = y - oof_mean
        best_row_truth = float(truth_on_table.min())
        for repeat in range(seeds):
            seed = base_seed + repeat
            rng = np.random.default_rng(seed)
            y_synthetic = truth_on_table + rng.choice(residuals, size=len(y), replace=True)
            fitted = {name: _build(build_model, name, seed, None).fit(X, y_synthetic) for name in model_names}
            for model_name, optimizer_name in pairs:
                w, _ = _propose(fitted[model_name], optimizer_name, seed, X, y_synthetic, optimizer_params)
                row = {"truth": truth_name, "seed": seed, "model": model_name, "optimizer": optimizer_name,
                       "compatible": w is not None}
                if w is not None:
                    true_value = float(truth.predict(w[None])[0][0])
                    believed = float(fitted[model_name].predict(w[None])[0][0])
                    row.update({"true_value": true_value, "best_row_true_value": best_row_truth,
                                "regret": true_value - best_row_truth, "believed_value": believed,
                                "optimism": true_value - believed,
                                "distance_ratio": float(nearest_distances(w[None], X)[0] / spacing)})
                synthetic_rows.append(row)
    semi_synthetic = pd.DataFrame(synthetic_rows)

    provenance = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "git": git_state(),
        "dataset": {"version": dataset.version, "sha256": dataset.sha256},
        "packages": {package: metadata.version(package) for package in PACKAGES},
        "models": list(model_names), "optimizers": list(optimizer_names), "truths": list(truth_names),
        "seeds": [base_seed + repeat for repeat in range(seeds)], "optimizer_params": optimizer_params,
        "spacing": spacing,
    }
    return {"compatibility": compatibility, "cross_judge": cross_judge, "semi_synthetic": semi_synthetic,
            "provenance": provenance}


def summarize(results) -> dict[str, pd.DataFrame]:
    judge = results["cross_judge"]
    judge = judge[judge.compatible]
    judged = [column for column in judge.columns if column.startswith("judged_by_")]

    def stability(group):
        points = np.array([json.loads(w) for w in group.weights]).reshape(len(group), -1)
        if len(points) < 2:
            return 0.0
        return float(np.mean([np.linalg.norm(a - b) for a, b in itertools.combinations(points, 2)]))

    cross = judge.groupby(["model", "optimizer"])[judged + ["distance_ratio"]].mean()
    cross["seed_spread"] = judge.groupby(["model", "optimizer"]).apply(stability)

    synthetic = results["semi_synthetic"]
    synthetic = synthetic[synthetic.compatible]
    regret = synthetic.pivot_table(index=["model", "optimizer"], columns="truth", values="regret",
                                   aggfunc=["mean", "std"])
    optimism = synthetic.pivot_table(index=["model", "optimizer"], columns="truth", values="optimism", aggfunc="mean")
    return {"cross_judge": cross, "regret": regret, "optimism": optimism}


def write(results, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    for name in ("compatibility", "cross_judge", "semi_synthetic"):
        results[name].to_csv(out / f"{name}.csv", index=False)
    (out / "provenance.json").write_text(json.dumps(results["provenance"], indent=2) + "\n", encoding="utf-8")
    tables = summarize(results)
    text = ["# Coupling study summary", "", f"Provenance: `{(out / 'provenance.json').name}`", ""]
    text += ["## Compatibility", "", "```", results["compatibility"].to_string(index=False), "```", ""]
    for title, key in (("Cross-judging on real data (mean over seeds)", "cross_judge"),
                       ("Semi-synthetic regret: true value minus best training row (lower is better)", "regret"),
                       ("Optimism: true minus believed value (positive = model over-promised)", "optimism")):
        text += [f"## {title}", "", "```", tables[key].round(2).to_string(), "```", ""]
    (out / "summary.md").write_text("\n".join(text), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--models", nargs="+", default=None, help="default: every registered model")
    parser.add_argument("--optimizers", nargs="+", default=None, help="default: every registered optimizer")
    parser.add_argument("--truths", nargs="+", default=list(DEFAULT_TRUTHS))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    results = run_study(model_names=args.models or sorted(models()),
                        optimizer_names=args.optimizers or sorted(optimizers()),
                        truth_names=args.truths, seeds=args.seeds)
    out = write(results, args.out or RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}")
    print((out / "summary.md").read_text(encoding="utf-8"))
    print(f"results: {out}")


if __name__ == "__main__":
    main()
