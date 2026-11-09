"""T20 mechanism study: how the 21 weights move R, using only the 2048 offline rows.

Three questions, each with the same folds and seed as the notebook:
  1. within-building ratio effect   (share of each objective vs R, octile medians)
  2. cross-building additivity      (joint vs additive-by-building GBM, and linear models that add
                                     either within-building or cross-building interaction terms)
  3. building sensitivity           (grouped permutation importance, single-building CV R^2)

Nothing here calls the evaluator. Results, figures and provenance go to one results folder.

Usage: uv run python analysis/mechanism/study.py [--out DIR]
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
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sdm.data import load_dataset
from sdm.paths import REPO_ROOT
from sdm.runs import PACKAGES, git_state
from sdm.simplex import FEATURES

OUT_DEFAULT = Path(__file__).resolve().parent / "results"
METRICS = ["cost", "carbon", "peak", "discomfort", "degradation", "ramping", "resilience"]
GROUPS = [list(range(7 * b, 7 * b + 7)) for b in range(3)]
SEED = 0

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _gbm(seed: int = SEED, **kwargs):
    return HistGradientBoostingRegressor(random_state=seed, max_iter=200, **kwargs)


def _cv_r2(make, X, y, kf):
    scores = []
    for train, test in kf.split(X):
        model = make().fit(X[train], y[train])
        scores.append(r2_score(y[test], model.predict(X[test])))
    return float(np.mean(scores))


def _cross_building_features(X):
    """Products between different buildings only (no squares, no within-building terms)."""
    blocks = [X[:, 7 * b:7 * b + 7] for b in range(3)]
    pairs = [(0, 1), (0, 2), (1, 2)]
    return np.hstack([np.einsum("ni,nj->nij", blocks[i], blocks[j]).reshape(len(X), -1)
                      for i, j in pairs])


def _within_building_features(X):
    blocks = [X[:, 7 * b:7 * b + 7] for b in range(3)]
    return np.hstack([PolynomialFeatures(2, include_bias=False).fit_transform(block)[:, 7:]
                      for block in blocks])


def run(out: Path) -> None:
    dataset = load_dataset(REPO_ROOT / "dataset_same8")
    X, y = dataset.X, dataset.y
    df = pd.DataFrame(X, columns=FEATURES)
    df["R"] = y
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    shares = np.column_stack([df[[f"b{b}_{m}" for b in (1, 2, 3)]].mean(axis=1) for m in METRICS])
    result = {
        "dataset": {"rows": int(len(y)), "sha256": dataset.sha256, "version": dataset.version,
                    "R": {"min": float(y.min()), "median": float(np.median(y)), "max": float(y.max())}},
        "seed": SEED, "folds": 5,
    }

    # ---- 1. within-building ratio effect -------------------------------------------------
    share_effect = {}
    for j, metric in enumerate(METRICS):
        share_effect[metric] = float(pd.Series(shares[:, j]).corr(pd.Series(y), method="spearman"))
    ramp_share = shares[:, 5]
    frame = pd.DataFrame({"R": y, "share": ramp_share})
    frame["octile"] = pd.qcut(frame["share"], 8, duplicates="drop")
    octile_table = (frame.groupby("octile", observed=True)["R"]
                    .agg(["count", "median", "mean", "min", "max"]).reset_index())
    octile_mid = [float(interval.mid) for interval in octile_table["octile"]]
    octile_table["share"] = octile_table["octile"].astype(str)
    octile_table = octile_table.drop(columns="octile")
    result["share_effect"] = {
        "spearman_average_share_vs_R": share_effect,
        "ramping_share_octiles": octile_table.round(4).to_dict("records"),
    }

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(ramp_share, y, s=8, alpha=0.35, color="#4C78A8", edgecolors="none")
    ax.plot(octile_mid, octile_table["median"], "o-", color="#E45756", lw=2, label="octile median")
    ax.set_xlabel("mean ramping share over the three buildings")
    ax.set_ylabel("R")
    ax.set_title("T20: higher ramping share, lower R (offline 2048 rows)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / "share_vs_R.png", dpi=150)
    plt.close(fig)

    # ---- 2. cross-building additivity ----------------------------------------------------
    joint = _cv_r2(lambda: _gbm(), X, y, kf)
    additive = _cv_r2(lambda: _gbm(interaction_cst=GROUPS), X, y, kf)

    def ridge(features=None):
        def make():
            steps = [StandardScaler()]
            if features is not None:
                steps.append(features)
            steps.append(Ridge(alpha=1.0))
            return make_pipeline(*steps)
        return make

    additive_linear = _cv_r2(ridge(), X, y, kf)
    within_linear = _cv_r2(ridge(WithinFeatures()), X, y, kf)
    cross_linear = _cv_r2(ridge(CrossFeatures()), X, y, kf)
    result["additivity"] = {
        "gbm_joint_r2": joint,
        "gbm_additive_by_building_r2": additive,
        "ridge_additive_linear_r2": additive_linear,
        "ridge_plus_within_building_terms_r2": within_linear,
        "ridge_plus_cross_building_terms_r2": cross_linear,
    }

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    labels = ["ridge\n(additive)", "ridge + within\nbuilding terms", "ridge + cross\nbuilding terms",
              "GBM\n(joint)", "GBM additive\nby building"]
    values = [additive_linear, within_linear, cross_linear, joint, additive]
    colors = ["#BAB0AC", "#BAB0AC", "#BAB0AC", "#4C78A8", "#F58518"]
    bars = ax.bar(labels, values, color=colors)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("5-fold CV R²")
    ax.set_title("T20: updating finding 2 - additive-by-building fits as well as joint")
    fig.tight_layout()
    fig.savefig(figs / "additivity.png", dpi=150)
    plt.close(fig)

    # ---- 3. building sensitivity ---------------------------------------------------------
    building_drop, single_r2 = {}, {}
    for b, cols in enumerate(GROUPS, start=1):
        drops, single_scores = [], []
        for train, test in kf.split(X):
            model = _gbm().fit(X[train], y[train])
            base = r2_score(y[test], model.predict(X[test]))
            perturbed = X[test].copy()
            rng = np.random.default_rng(SEED + b)
            for col in cols:
                perturbed[:, col] = rng.permutation(perturbed[:, col])
            drops.append(base - r2_score(y[test], model.predict(perturbed)))
            single_scores.append(
                r2_score(y[test], _gbm().fit(X[train][:, cols], y[train]).predict(X[test][:, cols])))
        building_drop[b] = float(np.mean(drops))
        single_r2[b] = float(np.mean(single_scores))
    result["building_sensitivity"] = {"permutation_r2_drop": building_drop, "single_building_cv_r2": single_r2}

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(["building 1", "building 2", "building 3"],
           [building_drop[b] for b in (1, 2, 3)], color=["#4C78A8", "#BAB0AC", "#F58518"])
    for i, b in enumerate((1, 2, 3)):
        ax.text(i, building_drop[b] + 0.02, f"{building_drop[b]:.4f}", ha="center", fontsize=9)
    ax.set_ylabel("R² drop when that building is shuffled")
    ax.set_title("T20: building 2 carries the least information")
    fig.tight_layout()
    fig.savefig(figs / "building_sensitivity.png", dpi=150)
    plt.close(fig)

    # ---- write ---------------------------------------------------------------------------
    provenance = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "git": git_state(),
        "dataset": {"version": dataset.version, "sha256": dataset.sha256,
                    "path": "dataset_same8/train.csv"},
        "packages": {package: metadata.version(package) for package in PACKAGES},
        "seed": SEED, "folds": 5, "code": "analysis/mechanism/study.py",
    }
    (out / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    lines = ["# T20 mechanism study", "", f"Dataset: 2048 offline rows, seed {SEED}, 5 folds.", "",
             "## Share effect (Spearman of mean objective share vs R)", ""]
    lines += [f"- {metric}: {value:+.4f}" for metric, value in share_effect.items()]
    lines += ["", "## Additivity", ""]
    lines += [f"- {name}: {value:.4f}" for name, value in result["additivity"].items()]
    lines += ["", "## Building sensitivity", ""]
    lines += [f"- building {b}: R² drop {building_drop[b]:.4f}, single-building CV R² {single_r2[b]:.4f}"
              for b in (1, 2, 3)]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nresults: {out}")


class WithinFeatures:
    """Transformer: add within-building squares and products, keep the raw weights."""

    def fit(self, X, y=None):
        self._transformers = [PolynomialFeatures(2, include_bias=False).fit(X[:, cols]) for cols in GROUPS]
        return self

    def transform(self, X):
        return np.hstack([X] + [t.transform(X[:, cols]) for t, cols in zip(self._transformers, GROUPS)])


class CrossFeatures:
    """Transformer: raw weights plus products between different buildings (no within-building terms)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.hstack([X, _cross_building_features(X)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path,
                        default=OUT_DEFAULT / datetime.now().strftime("%Y%m%d-%H%M%S"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    run(args.out)


if __name__ == "__main__":
    main()
