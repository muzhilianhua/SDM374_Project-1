# -*- coding: utf-8 -*-
"""产生 OOF 留出预测并用外部 v3.2 评分器逐折打分（四个配置对应的入口）。

本脚本**不复制评分器**：它把 `--scoring-dir` 指向仓库根的 `SDM374_scoring_v3_2/` 目录，
从那里 import `surrogate_scoring_v3_2.evaluate / load_final_policy / make_context`。
冻结划分依赖仓库根的 `外推方案/趋势与尾部/frozen/split_manifest.json`（协议
`sdm374.same8.splits.2x5x3.frozen.v1`）。

入口命令（在包根目录执行）：
    python run_oof_and_score.py --model gp-ard
    python run_oof_and_score.py --model tail5opt
    python run_oof_and_score.py --model hp_wlearn
    python run_oof_and_score.py --model blend_xgb_w20x3 ^
        --tree-oof ..\\SDM374_final_models_v1\\evaluation\\eval_predictions.csv

产物写到 --outdir（默认 scores/regen）：
    pred_<model>.csv     逐折留出预测（split_id, candidate_id, y_true, y_pred, sd）
    scores_<model>.csv   逐折 v3.2 五分量与原始量
    summary_<model>.json 汇总（10 折均值/标准差、逐折分数、逐折配对 Δ 与 p 值）

注意：
1) 逐折重拟合很慢（gp-ard 单折约 1 分钟，tail5opt / hp_wlearn 3 起点×80–100 iter 更久）。
2) 混合模型的 tree 分支必须用**逐折留出**的树预测；直接拿全量树模型预测会泄漏。
   历史 OOF 用的是交付包 evaluation/eval_predictions.csv，所以这里要求 --tree-oof。
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import modellib as M  # noqa: E402
import train_four_models as T  # noqa: E402

TARGET = "ramping"
POINTS = ("local_order", "local_mae", "global_order", "global_mae", "top1_regret")


def load_scoring(scoring_dir: Path):
    sys.path.insert(0, str(scoring_dir))
    from surrogate_scoring_v3_2 import evaluate, load_final_policy, make_context  # noqa: E402
    return evaluate, load_final_policy(), make_context


def iter_splits(manifest):
    pos = {c: i for i, c in enumerate(manifest["canonical_candidate_ids"])}
    for row in manifest["outer_splits"]:
        tr = np.array([pos[c] for c in row["outer_train_ids"]], int)
        te = np.array([pos[c] for c in row["outer_test_ids"]], int)
        yield dict(split_id=row["split_id"], train=tr, test=te,
                   test_ids=row["outer_test_ids"], train_ids=row["outer_train_ids"])


def fold_score_row(res, split_id):
    row = {"split_id": split_id, "score": res["score"]}
    row.update({f"p_{k}": res["points"][k] for k in POINTS})
    row.update({"local_mae_raw": res["local_mae"], "global_mae_raw": res["global_mae"],
                "local_pair_acc": res["local_pair_accuracy"], "global_pair_acc": res["global_pair_accuracy"],
                "top1_regret_raw": res["top1_regret"], "top1_id": res["top1_candidate_id"],
                "top1_actual": res["top1_actual"], "oracle_best": res["oracle_best_actual"],
                "train_std": res["train_std"], "low_n": res["low_n"],
                "recall_at_low_count": res["recall_at_low_count"],
                "local_bias": res["local_bias"], "global_bias": res["global_bias"]})
    return row


def block_metrics(y, mu, q01, q90):
    def blk(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return {"n": int(len(a)), "rmse": float(np.sqrt(np.mean((b - a) ** 2))),
                "mae": float(np.mean(np.abs(b - a))), "bias": float(np.mean(b - a))}
    lo = y <= q01
    hi = y > q90
    return {"overall": blk(y, mu), "low_q01": blk(y[lo], mu[lo]), "high_q90": blk(y[hi], mu[hi])}


def fit_fold(name, Xtr, ytr, Xte):
    cfg = T.CONFIGS[name]
    if name == "tail5opt":
        sw = T.tail_weights(ytr, cfg)
        model = T.fit_gp_weighted_opt(Xtr, ytr, cfg, sw, maxiter=cfg.get("maxiter", 80),
                                      n_starts=cfg.get("n_starts", 3), seed=T.SEED,
                                      noise_bounds=cfg.get("noise_bounds", (0.001, 0.3)))
        from scipy.linalg import solve_triangular
        Ks = model["block"](model["X"], Xte)
        mu = model["loc"] + model["scale"] * (Ks.T @ model["a"])
        v = model["block"].diag(Xte) - np.sum(solve_triangular(model["L"], Ks, lower=True) ** 2, axis=0)
        sd = model["scale"] * np.sqrt(np.maximum(v, 1e-12) + model["noise"])
        return mu, sd
    kernel, _ = T.build_kernel(cfg, n_features=Xtr.shape[1])
    opt = T.make_optimizer(maxiter=cfg.get("maxiter", 100), n_starts=cfg.get("n_starts", 4), seed=T.SEED)
    gp, loc, scale = T.fit_gp(Xtr, ytr, kernel, optimizer=opt, seed=T.SEED)
    mu, sd = gp.predict(Xte, return_std=True)
    return loc + scale * mu, scale * np.maximum(sd, 1e-9)


def main():
    ap = argparse.ArgumentParser(description="OOF 生成 + v3.2 逐折评分")
    ap.add_argument("--model", required=True, choices=list(T.CONFIGS))
    ap.add_argument("--data", default=str(HERE.parent / "dataset_same8" / "train.csv"))
    ap.add_argument("--manifest", default=str(HERE.parent / "外推方案" / "趋势与尾部" / "frozen" / "split_manifest.json"))
    ap.add_argument("--scoring-dir", default=str(HERE.parent / "SDM374_scoring_v3_2"),
                    help="外部评分器目录（仓库根的 SDM374_scoring_v3_2/）")
    ap.add_argument("--outdir", default=str(HERE / "scores" / "regen"))
    ap.add_argument("--tree-oof", default=None, help="blend 必需的逐折树模型预测 CSV")
    ap.add_argument("--seed", type=int, default=T.SEED)
    ap.add_argument("--folds", nargs="*", default=None)
    a = ap.parse_args()

    evaluate, policy, make_context = load_scoring(Path(a.scoring_dir))
    df = pd.read_csv(a.data, encoding="utf-8-sig").sort_values("candidate_id").reset_index(drop=True)
    X = df[list(M.FEATURES)].to_numpy(float)
    y = df[TARGET].to_numpy(float)
    ids = df["candidate_id"].to_numpy()
    q01, q90 = float(np.quantile(y, 0.01)), float(np.quantile(y, 0.90))
    manifest = json.loads(Path(a.manifest).read_text(encoding="utf-8"))

    tree = None
    if a.model == "blend_xgb_w20x3":
        if not a.tree_oof:
            raise SystemExit("blend 需要 --tree-oof（逐折留出的树模型预测），否则会泄漏")
        tree = pd.read_csv(a.tree_oof, encoding="utf-8-sig")

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    parts = [p for p in iter_splits(manifest) if a.folds is None or p["split_id"] in a.folds]

    preds, rows = [], []
    for p in parts:
        tr, te = p["train"], p["test"]
        if a.model == "blend_xgb_w20x3":
            mu_gp, sd = fit_fold("blend_xgb_w20x3", X[tr], y[tr], X[te])
            sub = tree[tree.split_id == p["split_id"]].set_index("candidate_id")
            mu_tree = sub.loc[p["test_ids"], "y_pred"].to_numpy(float)
            mu = (1.0 - T.BLEND_ALPHA) * mu_gp + T.BLEND_ALPHA * mu_tree
            sd = (1.0 - T.BLEND_ALPHA) * sd
        else:
            mu, sd = fit_fold(a.model, X[tr], y[tr], X[te])
        ctx = make_context(y[tr], policy)
        res = evaluate(y[te], mu, ctx, policy, np.asarray(p["test_ids"]))
        rows.append(fold_score_row(res, p["split_id"]))
        preds.append(pd.DataFrame({"split_id": p["split_id"], "candidate_id": p["test_ids"],
                                   "y_true": y[te], "y_pred": mu, "sd": sd}))
        print(f"[{a.model}] {p['split_id']} score={res['score']:.4f}")

    pred = pd.concat(preds, ignore_index=True)
    tab = pd.DataFrame(rows)
    pred.to_csv(outdir / f"pred_{a.model}.csv", index=False, encoding="utf-8-sig")
    tab.to_csv(outdir / f"scores_{a.model}.csv", index=False, encoding="utf-8-sig")
    blocks = block_metrics(pred.y_true.to_numpy(), pred.y_pred.to_numpy(), q01, q90)
    summary = {"model": a.model, "protocol": "sdm374.same8.splits.2x5x3.frozen.v1", "n": int(len(pred)),
               "seed": a.seed, "q01": q01, "q90": q90, "pooled": blocks,
               "v32": {"score_mean": float(tab.score.mean()),
                       "score_std": float(tab.score.std(ddof=0))},
               "fold_scores": tab.score.tolist(), "fold_ids": tab.split_id.tolist()}
    for k in POINTS:
        summary["v32"][f"p_{k}_mean"] = float(tab[f"p_{k}"].mean())
        summary["v32"][f"p_{k}_std"] = float(tab[f"p_{k}"].std(ddof=0))
    (outdir / f"summary_{a.model}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 与 gp-ard 的逐折配对（若同目录已有 gp-ard 结果）
    base_path = outdir / "scores_gp-ard.csv"
    if a.model != "gp-ard" and base_path.exists():
        base = pd.read_csv(base_path).set_index("split_id")["score"]
        d = tab.set_index("split_id")["score"] - base
        t, pv = stats.ttest_rel(tab.set_index("split_id")["score"], base)
        pair = {"model": a.model, "delta_mean": float(d.mean()), "delta_sd": float(d.std(ddof=1)),
                "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
                "t_stat": float(t), "p_value": float(pv)}
        (outdir / f"paired_delta_{a.model}_vs_gp-ard.json").write_text(
            json.dumps(pair, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(pair, ensure_ascii=False))

    print(f"[{a.model}] v3.2 {summary['v32']['score_mean']:.4f} ± {summary['v32']['score_std']:.4f} | "
          f"OOF RMSE {blocks['overall']['rmse']:.4f} low {blocks['low_q01']['rmse']:.4f} "
          f"high {blocks['high_q90']['rmse']:.4f}")


if __name__ == "__main__":
    main()
