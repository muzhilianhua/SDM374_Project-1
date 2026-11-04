# -*- coding: utf-8 -*-
"""复算 v3.2 评分 —— 只用包内已保存的 OOF 留出预测 + 外部评分器，不重新训练。

依赖的**外部路径**（不在包内，也不应拷进包）：
    1) 评分器目录：仓库根/SDM374_scoring_v3_2/
       通过 --scoring-dir 指定；脚本只从该目录 import surrogate_scoring_v3_2.py，
       评分规则与协议见该目录的 EVALUATION_PROTOCOL_V3_2.md / EVALUATION_POLICY_V3_2.json。
    2) 数据集：仓库根/dataset_same8/train.csv（提供每折训练标签与候选 id）
       通过 --data 指定。
    3) 冻结划分：仓库根/外推方案/趋势与尾部/frozen/split_manifest.json
       协议 id: sdm374.same8.splits.2x5x3.frozen.v1；通过 --manifest 指定。

入口命令（在包根目录执行）：
    python compute_scores.py
    python compute_scores.py --scoring-dir ..\\SDM374_scoring_v3_2

产物（默认写到 scores/）：
    scores_<model>.csv          逐折五分量与原始量（覆盖同名文件前请确认）
    recomputed_summary.json     四模型总分/分项/低尾/高段/逐折配对 Δ 与 p 值
    recomputed_fold_scores.csv  逐折分数宽表
    recomputed_paired_delta.csv 与 gp-ard 的逐折配对检验

注意：OOF 预测是"逐折留出"的，训练标签只用于构造该折评分上下文，不会泄漏到预测。
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
POINTS = ("local_order", "local_mae", "global_order", "global_mae", "top1_regret")
MODELS = ("gp-ard", "tail5opt", "hp_wlearn", "blend_xgb_w20x3")
TARGET = "ramping"


def load_scoring(scoring_dir: Path):
    sys.path.insert(0, str(scoring_dir))
    from surrogate_scoring_v3_2 import evaluate, load_final_policy, make_context  # noqa: E402
    return evaluate, load_final_policy(), make_context


def fold_row(res, split_id):
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


def main():
    ap = argparse.ArgumentParser(description="基于包内 OOF 预测复算 v3.2 评分")
    ap.add_argument("--data", default=str(HERE.parent / "dataset_same8" / "train.csv"))
    ap.add_argument("--manifest", default=str(HERE.parent / "外推方案" / "趋势与尾部" / "frozen" / "split_manifest.json"))
    ap.add_argument("--scoring-dir", default=str(HERE.parent / "SDM374_scoring_v3_2"))
    ap.add_argument("--outdir", default=str(HERE / "scores" / "recomputed"))
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    a = ap.parse_args()

    evaluate, policy, make_context = load_scoring(Path(a.scoring_dir))
    df = pd.read_csv(a.data, encoding="utf-8-sig").sort_values("candidate_id").reset_index(drop=True)
    y = df[TARGET].to_numpy(float)
    ids = df["candidate_id"].to_numpy()
    pos = {c: i for i, c in enumerate(ids)}
    manifest = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    q01, q90 = float(np.quantile(y, 0.01)), float(np.quantile(y, 0.90))
    frames, summaries = {}, {}
    for model in a.models:
        gz = HERE / "scores" / f"oof_pred_{model}.csv.gz"
        if not gz.exists():
            print(f"[skip] {model}: 缺少 {gz}")
            continue
        with gzip.open(gz, "rt", encoding="utf-8-sig") as f:
            pred = pd.read_csv(f)
        rows = []
        all_pred, all_true = [], []
        for row in manifest["outer_splits"]:
            tr_ids = row["outer_train_ids"]
            te_ids = row["outer_test_ids"]
            tr = np.array([pos[c] for c in tr_ids], int)
            te = np.array([pos[c] for c in te_ids], int)
            sub = pred[pred.split_id == row["split_id"]].set_index("candidate_id")
            mu = sub.loc[te_ids, "y_pred"].to_numpy(float)
            ctx = make_context(y[tr], policy)
            res = evaluate(y[te], mu, ctx, policy, np.asarray(te_ids))
            rows.append(fold_row(res, row["split_id"]))
            all_pred.append(mu)
            all_true.append(y[te])
        tab = pd.DataFrame(rows)
        frames[model] = tab
        tab.to_csv(outdir / f"scores_{model}.csv", index=False, encoding="utf-8-sig")

        yv = np.concatenate(all_true)
        mv = np.concatenate(all_pred)
        def blk(aa, bb):
            return {"n": int(len(aa)), "rmse": float(np.sqrt(np.mean((bb - aa) ** 2))),
                    "mae": float(np.mean(np.abs(bb - aa)), ), "bias": float(np.mean(bb - aa))}
        lo, hi = yv <= q01, yv > q90
        v32 = {"score_mean": float(tab.score.mean()), "score_std": float(tab.score.std(ddof=0))}
        for k in POINTS:
            v32[f"p_{k}_mean"] = float(tab[f"p_{k}"].mean())
            v32[f"p_{k}_std"] = float(tab[f"p_{k}"].std(ddof=0))
        summaries[model] = {"protocol": "sdm374.same8.splits.2x5x3.frozen.v1", "n": int(len(yv)),
                            "v32": v32, "fold_scores": tab.score.tolist(),
                            "fold_ids": tab.split_id.tolist(),
                            "oof_overall": blk(yv, mv), "oof_low_q01": blk(yv[lo], mv[lo]),
                            "oof_high_q90": blk(yv[hi], mv[hi])}
        print(f"[{model}] v3.2={v32['score_mean']:.4f}±{v32['score_std']:.4f} "
              f"RMSE={summaries[model]['oof_overall']['rmse']:.4f}")

    wide = pd.concat([frames[m][["split_id", "score"]].assign(model=m) for m in frames])
    wide.pivot(index="split_id", columns="model", values="score").reset_index().to_csv(
        outdir / "recomputed_fold_scores.csv", index=False, encoding="utf-8-sig")

    if "gp-ard" in frames:
        base = frames["gp-ard"].set_index("split_id")["score"]
        rows = []
        for m in frames:
            s = frames[m].set_index("split_id")["score"]
            d = s - base
            if m == "gp-ard":
                rows.append({"model": m, "delta_mean": 0.0, "delta_sd": 0.0, "wins": 0, "losses": 0,
                             "t_stat": None, "p_value": None})
                continue
            t, p = stats.ttest_rel(s, base)
            rows.append({"model": m, "delta_mean": float(d.mean()), "delta_sd": float(d.std(ddof=1)),
                         "delta_min": float(d.min()), "delta_max": float(d.max()),
                         "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
                         "t_stat": float(t), "p_value": float(p)})
        pd.DataFrame(rows).to_csv(outdir / "recomputed_paired_delta.csv", index=False, encoding="utf-8-sig")

    (outdir / "recomputed_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"复算完成 -> {outdir}")


if __name__ == "__main__":
    main()
