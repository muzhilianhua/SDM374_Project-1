"""Offline submission 2, follow-up: maximize the probability that a step from p01969 is really better.

Passing needs only R(w) < R(p01969) = 115.7648, so the target is P(gain < 0), not the size of the gain.

1. backtest:  on the 10 frozen outer splits, low-R training rows act as anchors and test rows within
              MAX_PAIR_DISTANCE as candidates. For each pair the ARD GP (fit on the split's training rows)
              and a local GP (800 training rows nearest the anchor) predict the gain with its joint posterior,
              then logistic calibration maps the judges' z-scores to an observed improvement rate. Each
              calibration fold holds out one outer split and drops every training record that shares an
              anchor or candidate with it: the two CV repeats contain the same rows, so a plain split holdout
              would reuse labels. The features are not nested, so the purge reduces but does not remove leakage.
2. scan:      full-data judges score steps 0.01-0.10 along the 19,483 simplex directions of study.py that
              avoid the two measured failures; the calibrated model ranks those whose z-scores lie inside the
              range the backtest observed. The finalists are re-ranked by their worst calibrated probability over
              a radius-0.01 ball, so the choice does not rest on one exact point.

Usage (repository root): uv run python analysis/offline_submission2/sign_study.py [--reuse-backtest]
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from sdm.config import REPO_ROOT
from sdm.data import load_dataset
from sdm.frozen_v32 import load_outer_splits
from sdm.models.local_gp import LocalGaussianProcess
from sdm.registry import build_model
from sdm.runs import PACKAGES, git_state
from sdm.simplex import as_batch, check_weights

OUT = REPO_ROOT / "analysis/offline_submission2/delivery/sign"
BAR = 115.7648
ARD = {"family": "gp", "kernel": "additive-ard", "lengths": [0.5], "noises": [0.03], "seed": 2026}
ANCHORS_PER_SPLIT, MAX_PAIR_DISTANCE, LOCAL_ROWS = 12, 0.45, 800
SEED, DIRECTIONS, MAX_COS_TO_FAILURE, ET_VETO = 2026, 20000, 0.5, 4.9
STEPS = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10)
FEATURES = ("z_ard", "z_lgp")
FINALISTS, BALL_RADIUS, BALL_POINTS = 10, 0.01, 64
AGREE = 0.3


def pair_stats(model, anchor, Q, chunk=4000):
    """Joint-posterior gain of each row of Q over `anchor` for a fitted GP judge (R units).

    Returns gain mean, latent SD of the gain, the judge's white-noise SD (unmodelled roughness at the new
    point), and the marginal predictive mean and SD at Q.
    """
    gp = model.estimator_
    if isinstance(model, LocalGaussianProcess):
        features, location, scale = (lambda P: P), 0.0, 1.0
    else:
        features, location, scale = model._features, model.location_, model.scale_
    to_r = scale * float(gp._y_train_std)
    shift = location + scale * float(gp._y_train_mean)
    signal, noise = gp.kernel_.k1, float(gp.kernel_.k2.noise_level)

    def parts(Z):
        K = gp.kernel_(Z, gp.X_train_)
        V = solve_triangular(gp.L_, K.T, lower=True, check_finite=False)
        return shift + to_r * (K @ gp.alpha_), V, signal.diag(Z) - (V ** 2).sum(axis=0)

    a = features(as_batch(anchor))
    mean_a, V_a, var_a = parts(a)
    frames = []
    for start in range(0, len(Q), chunk):
        Z = features(Q[start:start + chunk])
        mean_q, V_q, var_q = parts(Z)
        cov = signal(a, Z)[0] - V_a[:, 0] @ V_q
        frames.append(pd.DataFrame({
            "gain": mean_q - mean_a[0],
            "gain_sd": to_r * np.sqrt(np.maximum(var_q + var_a[0] - 2 * cov, 0)),
            "noise_sd": to_r * np.sqrt(noise),
            "mean": mean_q, "sd": to_r * np.sqrt(np.maximum(var_q, 0) + noise)}))
    return pd.concat(frames, ignore_index=True)


def z_score(stats):
    """Signed evidence for improvement: positive when the judge expects the new point to be better."""
    return -stats.gain / np.sqrt(stats.gain_sd ** 2 + stats.noise_sd ** 2)


def backtest_split(index):
    ds = load_dataset(REPO_ROOT / "dataset_same8")
    split = load_outer_splits(ds)[index]
    X, y, ids = ds.X, ds.y, np.asarray(ds.ids)
    Xtr, ytr, Xte, yte = X[split.train], y[split.train], X[split.test], y[split.test]
    ard = build_model("bayes-search", ARD).fit(Xtr, ytr)
    et = build_model("extratrees", {}).fit(Xtr, ytr)
    rows, used = [], 0
    for rank, ai in enumerate(np.argsort(ytr, kind="stable")):
        if used == ANCHORS_PER_SPLIT:
            break
        anchor = Xtr[ai]
        distance = np.linalg.norm(Xte - anchor, axis=1)
        near = np.flatnonzero(distance <= MAX_PAIR_DISTANCE)
        if len(near) < 2:
            continue
        used += 1
        local = np.argsort(np.linalg.norm(Xtr - anchor, axis=1), kind="stable")[:LOCAL_ROWS]
        lgp = build_model("local-gp", {}).fit(Xtr[local], ytr[local])
        frame = pd.DataFrame({"split": split.split_id, "anchor": ids[split.train][ai], "anchor_rank": rank,
                              "anchor_y": ytr[ai], "candidate": ids[split.test][near],
                              "distance": distance[near], "candidate_y": yte[near]})
        for tag, model in (("ard", ard), ("lgp", lgp)):
            stats = pair_stats(model, anchor, Xte[near])
            frame[f"{tag}_gain"], frame[f"{tag}_gain_sd"] = stats.gain, stats.gain_sd
            frame[f"{tag}_noise_sd"], frame[f"z_{tag}"] = stats.noise_sd, z_score(stats)
            frame[f"{tag}_p_abs"] = norm.cdf((ytr[ai] - stats["mean"]) / stats["sd"])
        frame["et_gain"] = et.predict(Xte[near])[0] - et.predict(anchor[None])[0][0]
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def purged_training(pairs, split):
    """Calibration-training rows for held-out `split`: the other splits minus every row that shares an
    anchor or a candidate with it, so no held-out outcome label is reused through the second CV repeat."""
    held = (pairs.split == split).to_numpy()
    ids = set(pairs.anchor[held]) | set(pairs.candidate[held])
    return ~held & ~pairs.anchor.isin(ids).to_numpy() & ~pairs.candidate.isin(ids).to_numpy()


def unique_pairs(pairs):
    """One record per (anchor, candidate); the second CV repeat re-creates most pairs."""
    return pairs.drop_duplicates(["anchor", "candidate"])


def calibrate(pairs):
    """Purged leave-one-split-out logistic calibration of improvement on the judges' z-scores."""
    y = (pairs.candidate_y < pairs.anchor_y).astype(int).to_numpy()
    variants = {"ard": ["z_ard"], "lgp": ["z_lgp"], "both": list(FEATURES)}
    oof = pd.DataFrame(index=pairs.index)
    for name, columns in variants.items():
        oof[name] = np.nan
        for split in pairs.split.unique():
            test, train = (pairs.split == split).to_numpy(), purged_training(pairs, split)
            model = LogisticRegression(C=1.0).fit(pairs.loc[train, columns], y[train])
            oof.loc[test, name] = model.predict_proba(pairs.loc[test, columns])[:, 1]
    raw = {"ard_p_gain": norm.cdf(pairs.z_ard), "lgp_p_gain": norm.cdf(pairs.z_lgp),
           "ard_p_abs": pairs.ard_p_abs, "lgp_p_abs": pairs.lgp_p_abs}
    scores = []
    for name, p in {**{f"calibrated_{k}": oof[k] for k in variants}, **raw}.items():
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        scores.append({"probability": name, "log_loss": log_loss(y, p), "brier": brier_score_loss(y, p),
                       "mean_predicted": p.mean(), "observed_rate": y.mean()})
    once = unique_pairs(pairs)
    y_once = (once.candidate_y < once.anchor_y).astype(int)
    final = {k: LogisticRegression(C=1.0).fit(once[c], y_once) for k, c in variants.items()}
    return oof, pd.DataFrame(scores), final


RELIABILITY_BINS = [0, .05, .1, .2, .3, .5, .7, 1.0]


def reliability(pairs, oof):
    y = (pairs.candidate_y < pairs.anchor_y).astype(int)
    bins = pd.cut(oof["both"], RELIABILITY_BINS, include_lowest=True)
    table = pd.DataFrame({"y": y, "p": oof["both"], "bin": bins}).groupby("bin", observed=True)
    return table.agg(pairs=("y", "size"), mean_predicted=("p", "mean"), observed=("y", "mean")).reset_index()


def selection_backtest(pairs, oof):
    """Per anchor, submit the neighbour the calibrated model likes best; how often is it really better?"""
    frame = pairs.assign(p=oof["both"], better=pairs.candidate_y < pairs.anchor_y)
    best = frame.loc[frame.groupby(["split", "anchor"]).p.idxmax()]
    rows = []
    for threshold in (0.0, 0.3, 0.5):
        chosen = best[best.p >= threshold]
        rows.append({"min_calibrated_p": threshold, "anchors": len(chosen),
                     "mean_predicted": chosen.p.mean(), "observed": chosen.better.mean()})
    return pd.DataFrame(rows)


def agreement(pairs):
    """Observed improvement rate when the judges agree or disagree that the candidate is better."""
    pairs = unique_pairs(pairs)
    frame = pairs.assign(better=pairs.candidate_y < pairs.anchor_y,
                         ard_says=np.where(pairs.z_ard > 0.3, "better", np.where(pairs.z_ard <= 0, "worse", "unsure")),
                         lgp_says=np.where(pairs.z_lgp > 0.3, "better", np.where(pairs.z_lgp <= 0, "worse", "unsure")))
    return (frame.groupby(["ard_says", "lgp_says"]).agg(pairs=("better", "size"), observed=("better", "mean"))
            .reset_index())


def scan(final, pairs):
    ds = load_dataset(REPO_ROOT / "dataset_same8")
    X, y, ids = ds.X, ds.y, list(ds.ids)
    p = X[ids.index("p01969")]
    known = {"baseline": as_batch(json.loads((REPO_ROOT / "weights.json").read_text()))[0],
             "submission1": as_batch(json.loads(
                 (REPO_ROOT / "SUBMISSION_READY/01_FIRST_REAL_TEST/weights.json").read_text()))[0]}
    spec = importlib.util.spec_from_file_location("submission2_study",
                                                  REPO_ROOT / "analysis/offline_submission2/study.py")
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    u = study.simplex_directions(DIRECTIONS, np.random.default_rng(SEED))  # same directions as study.py
    failure = np.stack([(v - p) / np.linalg.norm(v - p) for v in known.values()])
    allowed = np.flatnonzero(((u @ failure.T) < MAX_COS_TO_FAILURE).all(axis=1))

    ard = build_model("bayes-search", ARD).fit(X, y)
    lgp = build_model("local-gp", {}).fit(X, y)
    et = build_model("extratrees", {}).fit(X, y)
    et0 = et.predict(p[None])[0][0]
    frames = []
    for step in STEPS:
        W = p + step * u[allowed]
        ok = (W >= 0).all(axis=1)
        W, direction = W[ok], allowed[ok]
        frame = pd.DataFrame({"step": step, "direction": direction})
        for tag, model in (("ard", ard), ("lgp", lgp)):
            stats = pair_stats(model, p, W)
            frame[f"{tag}_gain"], frame[f"z_{tag}"] = stats.gain, z_score(stats)
            frame[f"{tag}_p_abs"] = norm.cdf((BAR - stats["mean"]) / stats["sd"])
        frame["et_gain"] = et.predict(W)[0] - et0
        for name, model in final.items():
            frame[f"p_{name}"] = model.predict_proba(frame[list(model.feature_names_in_)])[:, 1]
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    table = table[table.et_gain <= ET_VETO]
    lo, hi = pairs[list(FEATURES)].min(), pairs[list(FEATURES)].max()
    table["inside_backtest_range"] = ((table[list(FEATURES)] >= lo) & (table[list(FEATURES)] <= hi)).all(axis=1)

    # Finalists: the best directions overall plus the consensus direction of study.py, judged on a ball.
    inside = table[table.inside_backtest_range]
    finalists = pd.concat([inside.nlargest(FINALISTS, "p_both"),
                           inside[(inside.direction == 14840) & inside.step.between(0.03, 0.07)]])
    finalists = finalists.drop_duplicates(["step", "direction"]).reset_index(drop=True)
    ball = study.simplex_directions(BALL_POINTS, np.random.default_rng(7))
    worst = []
    for row in finalists.itertuples():
        w = p + row.step * u[row.direction]
        Q = np.clip(w + BALL_RADIUS * ball, 0, None).reshape(-1, 3, 7)
        Q = (Q / Q.sum(axis=2, keepdims=True)).reshape(-1, 21)
        z = pd.DataFrame({f"z_{tag}": z_score(pair_stats(model, p, Q)) for tag, model in (("ard", ard), ("lgp", lgp))})
        worst.append({"ball_min_p_both": final["both"].predict_proba(z[list(FEATURES)])[:, 1].min(),
                      "ball_max_et_gain": (et.predict(Q)[0] - et0).max(),
                      "cos_submission1": float(u[row.direction] @ failure[1]),
                      "cos_baseline": float(u[row.direction] @ failure[0])})
    finalists = pd.concat([finalists, pd.DataFrame(worst)], axis=1)
    # Eligible: both judges say "better" (z > AGREE, where the backtest's agreeing pairs improved 82 % of
    # the time against about 36 % when they disagreed) and the whole ball stays inside the ExtraTrees veto.
    finalists["eligible"] = ((finalists.z_ard > AGREE) & (finalists.z_lgp > AGREE)
                             & (finalists.ball_max_et_gain <= ET_VETO))
    return table, finalists.sort_values(["eligible", "ball_min_p_both"], ascending=False), p, u


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if "--reuse-backtest" in sys.argv:
        pairs = pd.read_csv(OUT / "backtest_pairs.csv.gz")
    else:
        os.environ.setdefault("OMP_NUM_THREADS", "1")  # inherited by the spawned workers
        with ProcessPoolExecutor(max_workers=5) as pool:
            pairs = pd.concat(pool.map(backtest_split, range(10)), ignore_index=True)
        pairs.to_csv(OUT / "backtest_pairs.csv.gz", index=False)
    oof, scores, final = calibrate(pairs)
    rel, sel = reliability(pairs, oof), selection_backtest(pairs, oof)
    scores.to_csv(OUT / "calibration_scores.csv", index=False)
    rel.to_csv(OUT / "reliability.csv", index=False)
    sel.to_csv(OUT / "selection_backtest.csv", index=False)
    agree = agreement(pairs)
    agree.to_csv(OUT / "agreement.csv", index=False)

    table, finalists, p, u = scan(final, pairs)
    best = (table[table.inside_backtest_range].sort_values("p_both", ascending=False)
            .groupby("step").head(5).sort_values(["step", "p_both"], ascending=[True, False]))
    best.to_csv(OUT / "scan_top.csv", index=False)
    by_step = table[table.inside_backtest_range].groupby("step").p_both.max().rename("max_p_both").reset_index()
    by_step.to_csv(OUT / "scan_by_step.csv", index=False)
    # The calibrated model is fitted on all pairs; read each finalist's probability through the purged
    # out-of-split reliability of its bin, which is the observed rate the backtest supports.
    observed = rel.set_index(rel.bin.astype(str)).observed
    finalists["purged_bin_observed"] = (pd.cut(finalists.p_both, RELIABILITY_BINS, include_lowest=True)
                                        .astype(str).map(observed))
    finalists.to_csv(OUT / "finalists.csv", index=False)
    top = finalists.iloc[0]
    assert top.eligible, "no finalist passes the agreement and veto rules"
    w = (p + top.step * u[int(top.direction)]).reshape(3, 7)
    check_weights(w.reshape(1, 21))
    (OUT / "max-p.json").write_text(json.dumps(w.tolist(), indent=2) + "\n")
    consensus = table[table.direction == 14840].sort_values("step")
    consensus.to_csv(OUT / "consensus_direction_14840.csv", index=False)

    coefficients = {name: {"intercept": float(m.intercept_[0]),
                           **{c: float(v) for c, v in zip(m.feature_names_in_, m.coef_[0])}}
                    for name, m in final.items()}
    provenance = {"created": datetime.now().isoformat(timespec="seconds"), "git": git_state(),
                  "packages": {package: metadata.version(package) for package in PACKAGES},
                  "backtest": {"anchors_per_split": ANCHORS_PER_SPLIT, "max_pair_distance": MAX_PAIR_DISTANCE,
                               "local_rows": LOCAL_ROWS, "pairs": len(pairs),
                               "pair_distance_quantiles": pairs.distance.quantile([0, .5, 1]).tolist(),
                               "improvement_rate": float((pairs.candidate_y < pairs.anchor_y).mean())},
                  "calibration": coefficients,
                  "scan": {"seed": SEED, "directions": DIRECTIONS, "steps": list(STEPS),
                           "max_cos_to_failure": MAX_COS_TO_FAILURE, "et_veto": ET_VETO},
                  "max_p": {"step": float(top.step), "direction": int(top.direction),
                            "p_both": float(top.p_both), "ball_min_p_both": float(top.ball_min_p_both),
                            "purged_bin_observed": float(top.purged_bin_observed)},
                  "calibration_holdout": "leave one outer split out; training rows sharing an anchor or candidate "
                                         "with it are purged; z features are not nested"}
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    pd.set_option("display.width", 250)
    print(json.dumps(provenance["backtest"], indent=1))
    print(scores.round(4).to_string(index=False))
    print(rel.round(3).to_string(index=False))
    print(sel.round(3).to_string(index=False))
    print(json.dumps(coefficients, indent=1))
    print(agree.round(3).to_string(index=False))
    print(by_step.round(3).to_string(index=False))
    print(finalists.round(3).to_string(index=False))
    print(best.round(3).to_string(index=False))
    print(consensus.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
