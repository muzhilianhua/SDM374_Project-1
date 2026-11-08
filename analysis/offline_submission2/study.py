"""Offline submission 2: judge short steps from p01969 with several surrogates (decision 23, no solver).

Three checks, all relative to each judge's own prediction for p01969 ("gain", negative is better):
1. lines:       re-score the PR #9 line searches with extra judges (ARD with the real points, ARD without
                p01969, local GP, ExtraTrees).
2. consensus:   random simplex directions from p01969 that avoid the two measured failures, ranked by the
                worse of the ARD and local-GP gains, with the ExtraTrees veto.
3. robustness:  finalists under judge variants that took no part in the search, plus the worst point of a
                radius-0.01 ball around each finalist.

Usage (repository root): uv run python analysis/offline_submission2/study.py
"""
from __future__ import annotations

import json
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from sdm.config import REPO_ROOT
from sdm.data import load_dataset
from sdm.evaluate import median_spacing, nearest_distances
from sdm.registry import build_model
from sdm.runs import PACKAGES, git_state
from sdm.simplex import as_batch, check_weights

OUT = REPO_ROOT / "analysis/offline_submission2/delivery"
BAR = 115.7648  # decision 17
REAL = {"baseline": 119.0916, "submission1": 119.71206678869203}
ARD = {"family": "gp", "kernel": "additive-ard", "lengths": [0.5], "noises": [0.03], "seed": 2026}
LINES = REPO_ROOT / "analysis/bayesian_search/delivery/current/all_candidate_weights.json"
ARD_OOF = (REPO_ROOT / "analysis/bayesian_search/delivery/cli_evidence/"
           "20261102-195112-evaluate-bayes-study-gp-ard/predictions.csv.gz")
SEED, DIRECTIONS, STEPS = 2026, 20000, (0.03, 0.05, 0.07)
MAX_COS_TO_FAILURE, ET_VETO = 0.5, 4.9


def simplex_directions(n, rng):
    """Unit directions in 21-d whose per-building components sum to zero, so steps keep each sum at 1."""
    u = rng.normal(size=(n, 3, 7))
    u -= u.mean(axis=2, keepdims=True)
    u = u.reshape(n, 21)
    return u / np.linalg.norm(u, axis=1, keepdims=True)


def predict(model, P):
    mean, std = model.predict(P)
    return np.asarray(mean), np.asarray(std)


def main():
    ds = load_dataset(REPO_ROOT / "dataset_same8")
    X, y, ids = ds.X, ds.y, list(ds.ids)
    best = ids.index("p01969")
    p = X[best]
    known = {"baseline": as_batch(json.loads((REPO_ROOT / "weights.json").read_text()))[0],
             "submission1": as_batch(json.loads(
                 (REPO_ROOT / "SUBMISSION_READY/01_FIRST_REAL_TEST/weights.json").read_text()))[0]}
    Xr = np.vstack([X, known["baseline"], known["submission1"]])
    yr = np.concatenate([y, [REAL["baseline"], REAL["submission1"]]])
    loo = np.arange(len(y)) != best

    judges = {"ard": build_model("bayes-search", ARD).fit(X, y),
              "ard_real": build_model("bayes-search", ARD).fit(Xr, yr),
              "ard_loo": build_model("bayes-search", ARD).fit(X[loo], y[loo]),
              "lgp": build_model("local-gp", {}).fit(X, y),
              "lgp_real": build_model("local-gp", {}).fit(Xr, yr),
              "lgp400": build_model("local-gp", {"neighbors": 400}).fit(X, y),
              "lgp1200": build_model("local-gp", {"neighbors": 1200}).fit(X, y),
              "lgp_loo": build_model("local-gp", {}).fit(X[loo], y[loo]),
              "et": build_model("extratrees", {}).fit(X, y)}
    own = {tag: predict(model, p[None])[0][0] for tag, model in judges.items()}

    def score(P, tags):
        table = {}
        for tag in tags:
            mean, std = predict(judges[tag], P)
            table[f"{tag}_gain"] = mean - own[tag]
            if tag != "et":
                table[f"{tag}_mean"], table[f"{tag}_std"] = mean, std
                table[f"{tag}_p_below_bar"] = norm.cdf((BAR - mean) / np.maximum(std, 1e-9))
        return pd.DataFrame(table)

    # 1. Line searches from PR #9, plus the two measured points as sanity rows.
    lines = {name: as_batch(w)[0] for name, w in json.loads(LINES.read_text()).items() if name.startswith("line-")}
    points = {"p01969": p, **known, **lines}
    P = np.stack(list(points.values()))
    check_weights(P)
    line_table = pd.concat([pd.DataFrame({
        "point": list(points), "distance": np.linalg.norm(P - p, axis=1),
        "distance_ratio": nearest_distances(P, X) / median_spacing(X)}),
        score(P, ["ard", "ard_real", "ard_loo", "lgp", "lgp_real", "et"])], axis=1)

    # 2. Consensus directions that avoid the two measured failures.
    rng = np.random.default_rng(SEED)
    u = simplex_directions(DIRECTIONS, rng)
    failure = {k: (v - p) / np.linalg.norm(v - p) for k, v in known.items()}
    cos = pd.DataFrame({f"cos_{k}": u @ d for k, d in failure.items()})
    allowed = (cos < MAX_COS_TO_FAILURE).all(axis=1).to_numpy()
    rows = []
    for step in STEPS:
        W = p + step * u
        ok = allowed & (W >= 0).all(axis=1)
        frame = pd.concat([pd.DataFrame({"step": step, "direction": np.flatnonzero(ok)}),
                           cos[ok].reset_index(drop=True),
                           score(W[ok], ["ard", "lgp", "et"])], axis=1)
        rows.append(frame)
    consensus = pd.concat(rows, ignore_index=True)
    consensus["worst_gain"] = consensus[["ard_gain", "lgp_gain"]].max(axis=1)
    eligible = consensus[consensus.et_gain <= ET_VETO]
    top = eligible.sort_values("worst_gain").groupby("step").head(5)
    agree = eligible.groupby("step").worst_gain.apply(lambda g: float((g < 0).mean()))
    # The recommended step length is 0.05; the same direction is then also judged at 0.03 and 0.07.
    chosen = int(eligible[eligible.step == 0.05].sort_values("worst_gain").iloc[0].direction)

    # 3. Robustness of the finalists.
    finalists = {"p01969": p,
                 **{f"consensus-d{step:.2f}": p + step * u[chosen] for step in STEPS},
                 "r05-d0.07": lines["line-gp-tree-blend-ei-r05-d0.07"],
                 "r06-d0.08": lines["line-gp-tree-blend-ei-r06-d0.08"],
                 "r10-d0.05": lines["line-gp-tree-blend-ei-r10-d0.05"]}
    F = np.stack(list(finalists.values()))
    check_weights(F)
    ball = simplex_directions(64, np.random.default_rng(7))
    Q = np.clip(F[:, None, :] + 0.01 * ball[None], 0, None).reshape(-1, 3, 7)
    Q = (Q / Q.sum(axis=2, keepdims=True)).reshape(-1, 21)
    check_weights(Q)
    robust = {"point": list(finalists),
              "cos_submission1": [float((w - p) @ failure["submission1"] / max(np.linalg.norm(w - p), 1e-12))
                                  for w in F]}
    for tag in ("ard", "ard_loo", "lgp400", "lgp", "lgp1200", "lgp_loo", "et"):
        robust[tag] = predict(judges[tag], F)[0] - own[tag]
        robust[f"{tag}_ball_worst"] = predict(judges[tag], Q)[0].reshape(len(F), -1).max(axis=1) - own[tag]
    robust = pd.DataFrame(robust)
    smooth = ["ard", "ard_loo", "lgp400", "lgp", "lgp1200", "lgp_loo"]
    robust["smooth_worst"] = robust[[f"{t}_ball_worst" for t in smooth]].max(axis=1)

    # Calibration of the canonical ARD out-of-fold predictions where it predicts low.
    oof = pd.read_csv(ARD_OOF)
    calibration = []
    for cut in (118, 120, 122, 125):
        sel = oof[oof.predicted_mean < cut]
        calibration.append({"predicted_below": cut, "rows": len(sel),
                            "mean_predicted": sel.predicted_mean.mean(), "mean_actual": sel.actual.mean(),
                            "actual_minus_predicted": (sel.actual - sel.predicted_mean).mean(),
                            "share_actual_below_bar": float((sel.actual < BAR).mean())})
    calibration = pd.DataFrame(calibration)

    OUT.mkdir(parents=True, exist_ok=True)
    line_table.to_csv(OUT / "lines.csv", index=False)
    top.to_csv(OUT / "consensus_top.csv", index=False)
    robust.to_csv(OUT / "robustness.csv", index=False)
    calibration.to_csv(OUT / "ard_oof_calibration.csv", index=False)
    for name, w in finalists.items():
        if name.startswith("consensus"):
            (OUT / f"{name}.json").write_text(json.dumps(w.reshape(3, 7).tolist(), indent=2) + "\n")
    lgp = judges["lgp"].estimator_.kernel_
    provenance = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "git": git_state(),
        "dataset": {"version": ds.version, "sha256": ds.sha256},
        "packages": {package: metadata.version(package) for package in PACKAGES},
        "judges": {"ard": {"model": "bayes-search", "params": ARD}, "lgp": {"model": "local-gp", "params": {}},
                   "lgp400/lgp1200": {"model": "local-gp", "params": {"neighbors": "400/1200"}},
                   "et": {"model": "extratrees", "params": {}},
                   "*_real": "fit on the 2048 rows plus baseline W* and submission 1 with their real R",
                   "*_loo": "fit without p01969"},
        "own_p01969_prediction": own,
        "consensus": {"seed": SEED, "directions": DIRECTIONS, "steps": list(STEPS),
                      "max_cos_to_failure": MAX_COS_TO_FAILURE, "et_veto": ET_VETO,
                      "allowed_directions": int(allowed.sum()), "scored": len(consensus),
                      "share_both_smooth_gains_negative": {str(k): v for k, v in agree.items()},
                      "chosen_direction": chosen},
        "robustness": {"ball_radius": 0.01, "ball_points": 64, "ball_seed": 7},
        "local_gp_kernel": str(lgp),
        "inputs": {"lines": str(LINES.relative_to(REPO_ROOT)), "ard_oof": str(ARD_OOF.relative_to(REPO_ROOT)),
                   "submission1": "SUBMISSION_READY/01_FIRST_REAL_TEST/weights.json", "baseline": "weights.json"},
        "real_results": REAL, "bar": BAR, "solver_calls": 0,
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2, default=float) + "\n")

    pd.set_option("display.width", 250)
    print(line_table.round(3).to_string(index=False))
    print(top.round(3).to_string(index=False))
    print(robust.round(2).to_string(index=False))
    print(calibration.round(2).to_string(index=False))
    print(json.dumps(provenance["consensus"], indent=1))


if __name__ == "__main__":
    main()
