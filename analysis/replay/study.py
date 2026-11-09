"""T17-prototype replay of the online loop on the 2048 offline rows.

The online phase is limited to 20 evaluator calls with at most 50 weight vectors per call.
This harness replays that budget against two truth sources and several policy settings:

  truth = "table"          a held-out slice of the 2048 rows is the candidate pool; a proposal is
                           scored by looking the row up in the table
  truth = "semi-synthetic" a surrogate fitted on all 2048 rows answers arbitrary weight vectors,
                           so the policy may propose continuous points

Policy settings (the T18 question):
  exploit share   0.0 (pure exploration), 0.5 (hybrid), 1.0 (pure acquisition)
  trust radius    max distance ratio of a proposal from the known data, 0.5 or 1.0
  surrogate       ridge2 or histgb

For every setting the harness records the best known R after each call, so the comparison is on the
same 20-call budget the team will spend online. Nothing here calls the evaluator.

The pool is larger than the 1000-point budget on purpose: if it were smaller, every policy would
eventually see every row and the curves would say nothing about selection. The comparison is still
weak when a policy finds its best point early (see README), so the report also carries
`calls_to_final_best`, the call at which a run stops improving.

Usage: uv run python analysis/replay/study.py [--seeds 5] [--out DIR]
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from importlib import metadata
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sdm.data import load_dataset
from sdm.evaluate import median_spacing, nearest_distances
from sdm.paths import REPO_ROOT
from sdm.registry import build_model, build_optimizer
from sdm.runs import PACKAGES, git_state
from sdm.simplex import FEATURES, project

OUT_DEFAULT = Path(__file__).resolve().parent / "results"
CALLS = 20
BATCH = 50
# The 20-call budget is 1000 points, so the pool must be larger than that; otherwise every policy
# ends up seeing the whole pool and the comparison says nothing.
POOL_SIZE = 1600
INIT_KNOWN = 2048 - POOL_SIZE
EXPLOIT_SHARES = (0.0, 0.5, 1.0)
SURROGATES = ("ridge2", "histgb")

# The full grid is 2 x 3 x 2; to stay inside a useful runtime the radius comparison is run for the
# strongest surrogate only, so radius is not confounded with surrogate choice.
CONFIGS = ([(surrogate, exploit, 1.0) for surrogate in SURROGATES for exploit in EXPLOIT_SHARES]
           + [("histgb", 0.5, 0.5)])

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _make_model(name: str, seed: int):
    """Build a registered surrogate with a replay-cheap configuration."""
    if name == "ridge2":
        params = {"seed": seed, "folds": 2, "alphas": (1.0,)}
    elif name == "histgb":
        params = {"max_iter": 60, "calib_folds": 2}
    else:
        params = {}
    return build_model(name, params)


def _explore(rng, anchors, n):
    """Random weight vectors concentrated around the current best anchors."""
    if n <= 0:
        return np.empty((0, len(FEATURES)))
    center = anchors[rng.integers(len(anchors), size=n)]
    concentration = rng.choice([300.0, 1000.0, 3000.0], size=(n, 1, 1))
    gamma = rng.gamma(center.reshape(-1, 3, 7) * concentration + 0.03)
    points = (gamma / gamma.sum(axis=2, keepdims=True)).reshape(n, -1)
    return project(points)


def _pool_policy(model, X_known, y_known, pool_X, n, exploit, radius, rng):
    """Choose n rows from the fixed pool; exploit share is ranked by predicted mean."""
    spacing = median_spacing(X_known)
    distance = nearest_distances(pool_X, X_known)
    eligible = np.flatnonzero(distance <= spacing * radius)
    if len(eligible) < n:
        eligible = np.argsort(distance, kind="stable")[:n]
    mean, spread = model.predict(pool_X[eligible])
    k = int(round(exploit * n))
    ranked = eligible[np.argsort(mean, kind="stable")]
    chosen = list(ranked[:k])
    rest = np.setdiff1d(eligible, chosen)
    if n - k > 0:
        chosen.extend(rng.choice(rest, size=n - k, replace=False))
    return np.asarray(chosen, dtype=int)


def _surrogate_truth(X_all, y_all, seed):
    model = _make_model("histgb", seed).fit(X_all, y_all)
    return model


def run_config(truth_name, surrogate, exploit, radius, seed, X_all, y_all, pool_idx, init_idx):
    rng = np.random.default_rng(seed)
    known_idx = np.sort(rng.choice(len(init_idx), size=INIT_KNOWN, replace=False))
    known_idx = init_idx[known_idx]

    if truth_name == "table":
        pool_X, pool_y = X_all[pool_idx], y_all[pool_idx]
        truth_model = None
        y_known = y_all[known_idx]
    else:
        truth_model = _surrogate_truth(X_all, y_all, seed)
        pool_X = X_all[pool_idx]
        pool_y = truth_model.predict(pool_X)[0]
        y_known = truth_model.predict(X_all[known_idx])[0]

    X_known = X_all[known_idx].copy()
    best_curve = [float(np.min(y_known))]
    used_pool = np.zeros(len(pool_X), dtype=bool)

    for _call in range(CALLS):
        model = _make_model(surrogate, seed).fit(X_known, y_known)
        if truth_name == "table":
            candidate_pool = np.flatnonzero(~used_pool)
            picks = _pool_policy(model, X_known, y_known, pool_X[candidate_pool], BATCH,
                                 exploit, radius, rng)
            picks = candidate_pool[picks]
            new_X, new_y = pool_X[picks], pool_y[picks]
            used_pool[picks] = True
        else:
            anchors = X_known[np.argsort(y_known, kind="stable")[:32]]
            k = int(round(exploit * BATCH))
            if k > 0:
                optimizer = build_optimizer("bayes-pool", {"acquisition": "mean",
                                                           "max_distance_ratio": radius,
                                                           "samples": 3000, "seed": int(seed)})
                exploit_points = optimizer.propose(model, X_known, y_known, n=k)
            else:
                exploit_points = np.empty((0, len(FEATURES)))
            new_X = np.vstack([exploit_points, _explore(rng, anchors, BATCH - k)])
            new_y = pool_y[:0] if truth_model is None else truth_model.predict(new_X)[0]
        X_known = np.vstack([X_known, new_X])
        y_known = np.concatenate([y_known, new_y])
        best_curve.append(float(np.min(y_known)))

    return {"curve": best_curve, "final_best": best_curve[-1],
            "first_call_best": best_curve[1],
            "calls_to_final_best": next(i for i, value in enumerate(best_curve)
                                        if value <= best_curve[-1] + 1e-9)}


def finish(out: Path, rows: list[dict], dataset, seeds: int, git: dict | None = None) -> None:
    """Aggregate runs, write results.json / provenance.json / figures / summary.md."""
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    groups = {}
    for row in rows:
        key = (row["truth"], row["surrogate"], row["exploit"], row["radius"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (truth_name, surrogate, exploit, radius), members in groups.items():
        finals = [row["final_best"] for row in members]
        summary.append({"truth": truth_name, "surrogate": surrogate, "exploit": exploit,
                        "radius": radius, "seeds": len(members),
                        "final_mean": float(np.mean(finals)), "final_std": float(np.std(finals)),
                        "final_min": float(np.min(finals)), "final_max": float(np.max(finals)),
                        "first_call_mean": float(np.mean([r["first_call_best"] for r in members])),
                        "first_call_std": float(np.std([r["first_call_best"] for r in members])),
                        "calls_to_final_median": float(np.median(
                            [r["calls_to_final_best"] for r in members])),
                        "curve_mean": [float(v) for v in np.mean(
                            [r["curve"] for r in members], axis=0)]})
    summary.sort(key=lambda item: (item["truth"], item["calls_to_final_median"],
                                   item["first_call_mean"]))

    result = {"protocol": {"calls": CALLS, "batch": BATCH, "pool_size": POOL_SIZE,
                           "initial_known": INIT_KNOWN, "seeds": seeds},
              "dataset": {"version": dataset.version, "sha256": dataset.sha256},
              "runs": summary}
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "git": git or git_state(),
        "dataset": {"version": dataset.version, "sha256": dataset.sha256,
                    "path": "dataset_same8/train.csv"},
        "packages": {package: metadata.version(package) for package in PACKAGES},
        "protocol": result["protocol"],
        "code": "analysis/replay/study.py",
        "note": "table truth scores proposals by lookup in the held-out rows; semi-synthetic truth is a "
                "histgb fitted on all 2048 rows. No evaluator calls.",
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    for truth_name in ("table", "semi-synthetic"):
        fig, ax = plt.subplots(figsize=(9, 4.8))
        shown = [item for item in summary if item["truth"] == truth_name]
        for item in shown:
            ax.plot(range(len(item["curve_mean"])), item["curve_mean"], marker="o", ms=3,
                    lw=1.4, alpha=0.85,
                    label=f"{item['surrogate']} exploit={item['exploit']:.1f} r={item['radius']:.1f}")
        ax.set_xlabel("evaluator call")
        ax.set_ylabel("best R so far")
        ax.set_title(f"T18 replay ({truth_name} truth, mean over seeds)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(figs / f"replay_{truth_name.replace('-', '_')}.png", dpi=150)
        plt.close(fig)

    # Convergence is the discriminating quantity: once a policy has seen the best it will ever see,
    # later calls add nothing, and the final value is often identical across settings.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=False)
    for ax, truth_name in zip(axes, ("table", "semi-synthetic")):
        shown = [item for item in summary if item["truth"] == truth_name]
        labels = [f"{i['surrogate']}\ne={i['exploit']:.1f} r={i['radius']:.1f}" for i in shown]
        ax.barh(range(len(shown)), [i["calls_to_final_median"] for i in shown], color="#4C78A8",
                alpha=0.85)
        ax.set_yticks(range(len(shown)))
        ax.set_yticklabels(labels, fontsize=6)
        ax.invert_yaxis()
        ax.set_xlabel("calls until the run's best R is reached (median over seeds)")
        ax.set_title(truth_name)
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figs / "calls_to_final_best.png", dpi=150)
    plt.close(fig)

    lines = ["# T18 replay study", "",
             f"{CALLS} calls x {BATCH} points, {POOL_SIZE}-row pool, {INIT_KNOWN} known rows, "
             f"{seeds} seeds, two truth sources.", "",
             "| truth | surrogate | exploit | radius | best after call 1 | median calls to the "
             "run's best | mean final R | std |",
             "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {i['truth']} | {i['surrogate']} | {i['exploit']:.1f} | {i['radius']:.1f} | "
              f"{i['first_call_mean']:.3f} | {i['calls_to_final_median']:.0f} | "
              f"{i['final_mean']:.3f} | {i['final_std']:.3f} |"
              for i in summary]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nresults: {out}")


def run(out: Path, seeds: int) -> None:
    dataset = load_dataset(REPO_ROOT / "dataset_same8")
    X_all, y_all = dataset.X, dataset.y
    rows = []
    for seed in range(seeds):
        rng = np.random.default_rng(10_000 + seed)
        order = rng.permutation(len(X_all))
        pool_idx, init_idx = np.sort(order[:POOL_SIZE]), np.sort(order[POOL_SIZE:])
        for truth_name in ("table", "semi-synthetic"):
            for surrogate, exploit, radius in CONFIGS:
                outcome = run_config(truth_name, surrogate, exploit, radius, seed,
                                     X_all, y_all, pool_idx, init_idx)
                rows.append({"truth": truth_name, "surrogate": surrogate, "exploit": exploit,
                             "radius": radius, "seed": seed, **outcome})
                with (out / "runs.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(rows[-1]) + "\n")
                print(f"{truth_name:14s} {surrogate:7s} exploit={exploit:.1f} "
                      f"radius={radius:.1f} seed={seed} final={outcome['final_best']:.3f}", flush=True)
    finish(out, rows, dataset, seeds)
    (out / "runs.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n",
                                    encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", type=Path,
                        default=OUT_DEFAULT / datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--from-runs", type=Path,
                        help="re-aggregate an existing runs.jsonl without refitting")
    args = parser.parse_args()
    dataset = load_dataset(REPO_ROOT / "dataset_same8")
    if args.from_runs is not None:
        out = args.from_runs.parent
        rows = [json.loads(line) for line in
                args.from_runs.read_text(encoding="utf-8").splitlines() if line.strip()]
        finish(out, rows, dataset, len({row["seed"] for row in rows}))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    run(args.out, args.seeds)


if __name__ == "__main__":
    main()
