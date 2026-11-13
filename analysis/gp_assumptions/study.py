"""Do the GP-ARD surrogate's assumptions hold? Checks on the training table and the local solver log.

The team's GP-ARD is `bayes-search` with `kernel = "additive-ard"`, `lengths = [0.5]`, `noises = [0.03]`
(offline configs `bayes-search-gp-ard-*.toml` at tag `offline-final`): a Matern 5/2 kernel that is additive
over the three buildings with seven length scales shared by the buildings, a 21-dimensional interaction
term of weight 0.1, white noise, a constant mean (standardised target) and 18 L-BFGS iterations.

Two data sources:
  train      the 2048 rows of `dataset_same8` (fit on 1400 random rows, held-out on the other 648)
  solver log the 6,568 local `solver.exe` calls in `analysis/replay/evidence/eval_log.jsonl` (6,548
             distinct points), used only to score predictions; never fitted except in part C

Parts, all written to one results folder with provenance:
  A  determinism and kernel family: repeated points in the log; generic ARD kernels vs the team kernel
  B  residual shape: closed-form leave-one-out residuals of a generic Matern 5/2 ARD fit
  C  stationarity: the same kernel refitted on three regions (training rows, near and far from the
     lowest R in the log)
  D  three changes to the team model: quadratic trend, log target, per-building length scales

Usage: uv run python analysis/gp_assumptions/study.py [--splits 3] [--out DIR]
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
from importlib import metadata
import json
from pathlib import Path
import time
import warnings

import numpy as np
from scipy import stats
from scipy.linalg import cho_solve
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Hyperparameter, Kernel, Matern, RBF, WhiteKernel

import sdm.models.bayes_search as bayes_search
from sdm.data import load_dataset
from sdm.evaluate import median_spacing, nearest_distances
from sdm.paths import REPO_ROOT
from sdm.registry import build_model
from sdm.runs import PACKAGES, git_state

warnings.filterwarnings("ignore", category=ConvergenceWarning)

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
SOLVER_LOG = REPO_ROOT / "analysis" / "replay" / "evidence" / "eval_log.jsonl"
FIT_ROWS = 1400
BAR = 115.765                      # best training row p01969, the offline bar (decision 17)
BASIN_SIZE = 200                   # lowest-R points of the solver log
REGION_ROWS = 800                  # rows per region in part C
BANDS = {"near": (0.0, 1.0), "mid": (1.0, 1.5), "far": (1.5, np.inf)}   # distance ratio to the fit rows
TEAM_PARAMS = {"family": "gp", "kernel": "additive-ard", "lengths": (0.5,), "noises": (0.03,), "seed": 2026}
TeamKernel = bayes_search.BuildingARDKernel


class PerBuildingARDKernel(Kernel):
    """The team kernel with seven length scales per building (21 in all) instead of seven shared ones."""

    def __init__(self, length_scale=(.5,) * 21, global_weight=.1):
        scales = tuple(length_scale)
        self.length_scale = scales * 3 if len(scales) == 7 else scales
        self.global_weight = global_weight

    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter("length_scale", "numeric", (.03, 3.0), 21)

    def __call__(self, X, Y=None, eval_gradient=False):
        scales, weight = np.asarray(self.length_scale, float), self.global_weight
        parts = [Matern(scales[7*b:7*b+7], (.03, 3), nu=2.5)(
            X[:, 7*b:7*b+7], None if Y is None else Y[:, 7*b:7*b+7], eval_gradient=eval_gradient)
            for b in range(3)]
        full = Matern(scales, (.03, 3), nu=2.5)(X, Y, eval_gradient=eval_gradient)
        if not eval_gradient:
            return sum(parts) * (1 - weight) / 3 + weight * full
        matrix = sum(part[0] for part in parts) * (1 - weight) / 3 + weight * full[0]
        gradient = weight * full[1]
        for b in range(3):
            gradient[:, :, 7*b:7*b+7] += parts[b][1] * (1 - weight) / 3
        return matrix, gradient

    def diag(self, X):
        return np.ones(len(X))

    def is_stationary(self):
        return True


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_solver_log():
    """Distinct renormalised points of the solver log, plus the spread of R among repeated points."""
    rows = [json.loads(line) for line in SOLVER_LOG.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda row: row["eval_id"])
    W = np.array([row["weights"] for row in rows], float).reshape(-1, 3, 7)
    X = (W / W.sum(2, keepdims=True)).reshape(-1, 21)
    y = np.array([row["R"] for row in rows], float)
    keys = np.round(X, 9)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    spreads = [float(np.ptp(y[inverse == group])) for group in np.where(counts > 1)[0]]
    _, first = np.unique(keys, axis=0, return_index=True)
    keep = np.sort(first)
    return X[keep], y[keep], {"calls": len(rows), "distinct": int(len(keep)), "repeated_groups": len(spreads),
                              "max_repeat_spread": max(spreads, default=0.0)}


EMPTY_SCORES = ("mae", "spearman", "nlpd", "cover68", "cover95", "z_mean", "z_sd", "z_skew", "z_kurtosis")


def gaussian_scores(mean, std, y) -> dict:
    z = (y - mean) / std
    return {"n": int(len(y)), "mae": float(np.abs(mean - y).mean()), "spearman": float(stats.spearmanr(mean, y)[0]),
            "nlpd": float(np.mean(0.5 * np.log(2 * np.pi * std ** 2) + 0.5 * z ** 2)),
            "cover68": float(np.mean(np.abs(z) < 1)), "cover95": float(np.mean(np.abs(z) < 1.96)),
            "z_mean": float(z.mean()), "z_sd": float(z.std()),
            "z_skew": float(stats.skew(z)), "z_kurtosis": float(stats.kurtosis(z))}


def search_scores(mean, std, Xlog, ylog, ratio) -> dict:
    """How the model would steer a search over the solver-log points."""
    basin = np.argsort(ylog)[:BASIN_SIZE]
    best50 = set(np.argsort(ylog)[:50].tolist())
    order = np.argsort(mean)
    out = {}
    for band, (low, high) in BANDS.items():   # every band is always present; an empty one has n = 0 and None scores
        m = (ratio >= low) & (ratio < high)
        out[band] = gaussian_scores(mean[m], std[m], ylog[m]) if m.any() else dict.fromkeys(EMPTY_SCORES, None) | {"n": 0}
    out.update({"log_spearman": float(stats.spearmanr(mean, ylog)[0]),
                "basin_real": float(ylog[basin].mean()), "basin_pred": float(mean[basin].mean()),
                "basin_std": float(std[basin].mean()),
                "basin_p_below_bar": float(np.mean(stats.norm.cdf((BAR - mean[basin]) / std[basin]))),
                "top1_real": float(ylog[order[0]]), "top10_real_mean": float(ylog[order[:10]].mean()),
                "top50_recall_of_best50": len(best50 & set(order[:50].tolist())) / 50})
    return out


def generic_gp(kind: str) -> GaussianProcessRegressor:
    if kind == "isotropic":
        base = Matern(0.5, (1e-2, 1e2), nu=2.5)
    elif kind == "rbf":
        base = RBF(np.full(21, 0.5), (1e-2, 1e2))
    else:
        base = Matern(np.full(21, 0.5), (1e-2, 1e2), nu={"matern52": 2.5, "matern32": 1.5}[kind])
    kernel = ConstantKernel(1.0, (1e-3, 1e2)) * base + WhiteKernel(1e-2, (1e-6, 1.0))
    return GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=0, random_state=0)


def fit_team(X, y, *, per_building=False, **changes):
    bayes_search.BuildingARDKernel = PerBuildingARDKernel if per_building else TeamKernel
    try:
        return build_model("bayes-search", {**TEAM_PARAMS, **changes}).fit(X, y)
    finally:
        bayes_search.BuildingARDKernel = TeamKernel


def team_summary(model, per_building=False) -> dict:
    metrics = model.metrics
    return {"lml_per_row": metrics["log_marginal_likelihood"] / metrics["train_rows"],
            "noise_R": float(np.sqrt(metrics["noise"]) * model.scale_) if model.params.target == "raw" else None,
            "optimizer_success": metrics.get("optimizer_success"), "optimizer_nit": metrics.get("optimizer_nit"),
            "lengths": [metrics[f"length_{i}"] for i in range(21 if per_building else 7)]}


GENERIC = {"matern52-ard21": "matern52", "matern32-ard21": "matern32", "rbf-ard21": "rbf",
           "matern52-isotropic": "isotropic", "matern52-ard21-sqrt": "matern52"}
CHANGES = {"team": {}, "team+trend": {"trend": True}, "team+log": {"target": "log"},
           "team+per-building": {"per_building": True}, "team+trend+log": {"trend": True, "target": "log"},
           "team+all-three": {"trend": True, "target": "log", "per_building": True}}


def run_study(splits: int) -> dict:
    dataset = load_dataset(REPO_ROOT / "dataset_same8")
    Xt, yt = np.asarray(dataset.X, float), np.asarray(dataset.y, float)
    Xlog, ylog, log_info = load_solver_log()
    spacing = median_spacing(Xt)
    records = []                                  # one per (split, variant)
    loo = None
    for split in range(splits):
        perm = np.random.default_rng(split).permutation(len(yt))
        fit_rows, held = perm[:FIT_ROWS], perm[FIT_ROWS:]
        Xf, yf = Xt[fit_rows], yt[fit_rows]
        ratio = nearest_distances(Xlog, Xf) / spacing
        for name, changes in CHANGES.items():
            start = time.time()
            model = fit_team(Xf, yf, **changes)
            mean_h, std_h = model.predict(Xt[held])
            mean_l, std_l = model.predict(Xlog)
            records.append({"split": split, "variant": name, "part": "A" if name == "team" else "D",
                            "seconds": time.time() - start, **team_summary(model, changes.get("per_building", False)),
                            "holdout": gaussian_scores(mean_h, std_h, yt[held]),
                            "solver_log": search_scores(mean_l, std_l, Xlog, ylog, ratio)})
            print(f"split {split} {name:22s} {records[-1]['seconds']:5.1f}s", flush=True)
        for name, kind in GENERIC.items():
            start = time.time()
            transform = (lambda X: np.sqrt(np.maximum(X, 0))) if name.endswith("sqrt") else (lambda X: X)
            gp = generic_gp(kind).fit(transform(Xf), yf)
            mean_h, std_h = gp.predict(transform(Xt[held]), return_std=True)
            mean_l, std_l = gp.predict(transform(Xlog), return_std=True)
            kernel = gp.kernel_
            records.append({"split": split, "variant": name, "part": "A", "seconds": time.time() - start,
                            "lml_per_row": gp.log_marginal_likelihood_value_ / len(yf),
                            "noise_R": float(np.sqrt(kernel.k2.noise_level) * yf.std()),
                            "amplitude_R": float(np.sqrt(kernel.k1.k1.constant_value) * yf.std()),
                            "lengths": np.atleast_1d(kernel.k1.k2.length_scale).tolist(),
                            "holdout": gaussian_scores(mean_h, std_h, yt[held]),
                            "solver_log": search_scores(mean_l, std_l, Xlog, ylog, ratio)})
            print(f"split {split} {name:22s} {records[-1]['seconds']:5.1f}s", flush=True)
            if split == 0 and name == "matern52-ard21":
                loo = leave_one_out(gp, yf)
    regions = stationarity(Xt, yt, Xlog, ylog, spacing)
    provenance = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "git": git_state(),
        "study_sha256": sha256(Path(__file__).resolve()),
        "dataset": {"version": dataset.version, "sha256": dataset.sha256},
        "solver_log": {"path": str(SOLVER_LOG.relative_to(REPO_ROOT)), "sha256": sha256(SOLVER_LOG), **log_info},
        "packages": {package: metadata.version(package) for package in PACKAGES},
        "splits": {"seeds": list(range(splits)), "fit_rows": FIT_ROWS, "held_out_rows": len(yt) - FIT_ROWS},
        "team_params": {key: list(value) if isinstance(value, tuple) else value for key, value in TEAM_PARAMS.items()},
        "spacing": spacing, "bar": BAR, "basin_size": BASIN_SIZE,
        "evaluator_calls": 0,
    }
    return {"provenance": provenance, "records": records, "loo": loo, "stationarity": regions}


def leave_one_out(gp, y) -> dict:
    """Closed-form LOO residuals of a fitted GP (Rasmussen & Williams eq. 5.12), standardised."""
    inverse = cho_solve((gp.L_, True), np.eye(len(y)))
    # y - mu_loo = alpha / diag(K^-1) and sd_loo = diag(K^-1)^-1/2, so z = alpha / sqrt(diag(K^-1)).
    z = gp.alpha_ / np.sqrt(np.diag(inverse))
    low, high = y <= np.quantile(y, 0.25), y >= np.quantile(y, 0.75)
    return {"rows": int(len(y)), "z_mean": float(z.mean()), "z_sd": float(z.std()), "skew": float(stats.skew(z)),
            "excess_kurtosis": float(stats.kurtosis(z)), "share_abs_z_above_3": float(np.mean(np.abs(z) > 3)),
            "z_sd_lowest_quartile": float(z[low].std()), "z_sd_highest_quartile": float(z[high].std()),
            "z_mean_lowest_quartile": float(z[low].mean()), "z_mean_highest_quartile": float(z[high].mean())}


def stationarity(Xt, yt, Xlog, ylog, spacing) -> list[dict]:
    rng = np.random.default_rng(2026)
    to_best = nearest_distances(Xlog, Xlog[[int(np.argmin(ylog))]])
    far = np.where(to_best / spacing > 2)[0]
    regions = {"training rows": (Xt, yt, rng.permutation(len(yt))[:REGION_ROWS]),
               "solver log near its lowest R": (Xlog, ylog, np.argsort(to_best)[:REGION_ROWS]),
               "solver log > 2 spacings from its lowest R": (Xlog, ylog, rng.permutation(far)[:REGION_ROWS])}
    out = []
    for name, (X, y, rows) in regions.items():
        gp = generic_gp("matern52").fit(X[rows], y[rows])
        kernel = gp.kernel_
        lengths = np.atleast_1d(kernel.k1.k2.length_scale)
        out.append({"region": name, "rows": int(len(rows)), "R_min": float(y[rows].min()), "R_max": float(y[rows].max()),
                    "radius": float(np.median(np.linalg.norm(X[rows] - X[rows].mean(0), axis=1)) / spacing),
                    "amplitude_R": float(np.sqrt(kernel.k1.k1.constant_value) * y[rows].std()),
                    "noise_R": float(np.sqrt(kernel.k2.noise_level) * y[rows].std()),
                    "geo_mean_length": float(np.exp(np.log(lengths).mean())), "lengths": lengths.tolist()})
        print(f"region {name}", flush=True)
    return out


def _mean_sd(values) -> str:
    values = [v for v in values if v is not None]
    return f"{np.mean(values):.3f} ± {np.std(values):.3f}" if values else "—"


def summarize(results) -> str:
    records, prov = results["records"], results["provenance"]
    variants = list(dict.fromkeys(record["variant"] for record in records))

    def column(variant, getter):
        return _mean_sd([getter(r) for r in records if r["variant"] == variant])

    rows = [("held-out MAE", lambda r: r["holdout"]["mae"]),
            ("held-out Spearman", lambda r: r["holdout"]["spearman"]),
            ("held-out NLPD", lambda r: r["holdout"]["nlpd"]),
            ("held-out 95% coverage", lambda r: r["holdout"]["cover95"]),
            ("held-out z kurtosis", lambda r: r["holdout"]["z_kurtosis"]),
            ("log near MAE", lambda r: r["solver_log"]["near"]["mae"]),
            ("log near z sd", lambda r: r["solver_log"]["near"]["z_sd"]),
            ("log near z mean", lambda r: r["solver_log"]["near"]["z_mean"]),
            ("log mid MAE", lambda r: r["solver_log"]["mid"]["mae"]),
            ("log mid 95% coverage", lambda r: r["solver_log"]["mid"]["cover95"]),
            ("log far MAE", lambda r: r["solver_log"]["far"]["mae"]),
            ("log far 95% coverage", lambda r: r["solver_log"]["far"]["cover95"]),
            ("log far z mean", lambda r: r["solver_log"]["far"]["z_mean"]),
            ("log Spearman (all)", lambda r: r["solver_log"]["log_spearman"]),
            ("basin predicted R", lambda r: r["solver_log"]["basin_pred"]),
            ("basin predicted std", lambda r: r["solver_log"]["basin_std"]),
            ("basin P(R < bar)", lambda r: r["solver_log"]["basin_p_below_bar"]),
            ("top-1 pick real R", lambda r: r["solver_log"]["top1_real"]),
            ("top-10 picks mean real R", lambda r: r["solver_log"]["top10_real_mean"]),
            ("top-50 recall of best 50", lambda r: r["solver_log"]["top50_recall_of_best50"]),
            ("fitted noise (R units)", lambda r: r.get("noise_R")),
            ("log marginal likelihood / row", lambda r: r.get("lml_per_row")),
            ("fit seconds", lambda r: r["seconds"])]
    text = ["# GP-ARD assumption study — summary", "",
            f"Provenance: `provenance.json` (commit `{prov['git']['commit']}`, dirty {prov['git']['dirty']}, "
            f"study sha256 `{prov['study_sha256'][:12]}`). Mean ± sd over {len(prov['splits']['seeds'])} splits "
            f"of {prov['splits']['fit_rows']} fit / {prov['splits']['held_out_rows']} held-out training rows. "
            f"Solver log: {prov['solver_log']['distinct']} distinct points; basin = its {BASIN_SIZE} lowest "
            f"(mean real R {records[0]['solver_log']['basin_real']:.2f}); bar {BAR}.", ""]
    for title, part in (("Part A: kernel family and structure", "A"), ("Part D: changes to the team model", "D")):
        shown = [v for v in variants if any(r["variant"] == v and (r["part"] == part or (part == "D" and v == "team"))
                                            for r in records)]
        text += [f"## {title}", "", "| metric | " + " | ".join(shown) + " |", "|---" * (len(shown) + 1) + "|"]
        text += [f"| {label} | " + " | ".join(column(v, getter) for v in shown) + " |" for label, getter in rows]
        text += [""]
    text += ["Optimizer success per split: " + "; ".join(
        f"{v}: {[r.get('optimizer_success') for r in records if r['variant'] == v]}"
        for v in variants if v.startswith("team")), ""]
    loo = results["loo"]
    text += ["## Part B: leave-one-out residuals (matern52-ard21, split 0)", "",
             f"z mean {loo['z_mean']:+.3f}, sd {loo['z_sd']:.3f}, skew {loo['skew']:+.2f}, excess kurtosis "
             f"{loo['excess_kurtosis']:.2f}, share |z| > 3 {loo['share_abs_z_above_3']:.2%} (normal 0.27%); "
             f"lowest-R quartile z sd {loo['z_sd_lowest_quartile']:.2f} / mean {loo['z_mean_lowest_quartile']:+.2f}, "
             f"highest-R quartile z sd {loo['z_sd_highest_quartile']:.2f} / mean {loo['z_mean_highest_quartile']:+.2f}.", ""]
    text += ["## Part C: stationarity (matern52-ard21 refitted per region)", "",
             "| region | rows | median radius (spacings) | R range | amplitude (R) | noise (R) | geo-mean length | "
             "shortest lengths (dims) |", "|---|---|---|---|---|---|---|---|"]
    for region in results["stationarity"]:
        lengths = np.asarray(region["lengths"])
        short = np.argsort(lengths)[:5]
        text.append(f"| {region['region']} | {region['rows']} | {region['radius']:.2f} | "
                    f"{region['R_min']:.1f}–{region['R_max']:.1f} | "
                    f"{region['amplitude_R']:.2f} | {region['noise_R']:.3f} | {region['geo_mean_length']:.3f} | "
                    + ", ".join(f"{lengths[i]:.3f} ({i})" for i in short) + " |")
    log = prov["solver_log"]
    text += ["", f"Determinism: {log['repeated_groups']} repeated points in the solver log, largest R spread "
             f"{log['max_repeat_spread']:.1e}.", ""]
    team = [r for r in records if r["variant"] == "team+per-building"]
    if team:
        text += ["Per-building length scales (split 0; buildings 1–3, objectives in `METRICS` order):", "",
                 "```", *(f"building {b + 1}: " + " ".join(f"{v:.2f}" for v in team[0]["lengths"][7*b:7*b+7])
                          for b in range(3)), "```", ""]
    return "\n".join(text)


def write(results, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    (out / "provenance.json").write_text(json.dumps(results["provenance"], indent=2) + "\n", encoding="utf-8")
    (out / "results.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    (out / "summary.md").write_text(summarize(results) + "\n", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = write(run_study(args.splits), args.out or RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}")
    print((out / "summary.md").read_text(encoding="utf-8"))
    print(f"results: {out}")


if __name__ == "__main__":
    main()
