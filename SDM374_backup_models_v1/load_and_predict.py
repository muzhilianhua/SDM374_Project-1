# -*- coding: utf-8 -*-
"""SDM374 备用代理模型包：统一加载 / 预测 / 自检入口。

用法（全部在包根目录执行）：
    python load_and_predict.py --list
    python load_and_predict.py --model gp-ard --weights "w1,w2,...,w21"
    python load_and_predict.py --model gp-ard --weights "1,1,...,1" --json
    python load_and_predict.py --model blend_xgb_w20x3 --weights-file 第一组离线提交.csv
    python load_and_predict.py --self-check

特征顺序有两种路径：
  - 21 维向量 / (n,21) 数组：只能按位置，顺序写死为 FEATURES（建筑1→3，每栋 cost→…→resilience）；
  - 3×7 表格 / CSV（中英文标签均可）：一律按**标签**重排，行列物理顺序无关。
未知/重复/缺失标签、行数≠3、形状不合法一律抛 ContractError，绝不做"猜顺序"。
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
PROJECT = HERE.parent                      # 仓库根（数据集与历史提交仅用于自检，缺失时跳过）
sys.path.insert(0, str(HERE))

import modellib as M  # noqa: E402

DATA = PROJECT / "dataset_same8" / "train.csv"
SUBMISSION = PROJECT / "第一组离线提交.csv"
SUBMISSION_HISTORICAL_GP = 121.488948      # 第一组离线提交_gp预测.csv 里的 gp_ard_ramping
SUBMISSION_TRUTH = 119.712                 # 用户给出的模拟器真值
OOF_GZ = HERE / "scores" / "oof_pred_{model}.csv.gz"


# ─────────────────────────── 权重输入 ───────────────────────────
def read_weights(path: Path):
    """读取 csv / json / npy / txt 形式的权重。"""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(obj, dict):
            for key in ("weights", "candidate_weights", "weight", "w"):
                if key in obj:
                    obj = obj[key]
                    break
        return M.pack_weights(obj)
    if suffix in (".npy",):
        return M.pack_weights(np.load(path))
    if suffix == ".csv":
        df = pd.read_csv(path, encoding="utf-8-sig", index_col=0)
        return M.pack_weights(df)
    txt = path.read_text(encoding="utf-8-sig").strip().replace("\n", ",")
    return M.pack_weights([float(x) for x in txt.split(",") if x.strip()])


def weights_from_string(s: str):
    s = s.strip()
    if s.startswith("["):
        return M.pack_weights(json.loads(s))
    return M.pack_weights([float(x) for x in s.replace("\n", ",").split(",") if x.strip()])


def model_names():
    return sorted(p.name for p in MODELS.iterdir()
                  if p.is_dir() and (p / "model.npz").exists() and (p / "params.json").exists())


# ─────────────────────────── 预测 ───────────────────────────
def predict(model: str, weights, return_std=True):
    bundle = M.load_model(MODELS / model)
    x = M.pack_weights(weights)
    mean, sd = bundle.predict(x) if return_std else (bundle.predict(x, return_std=False), None)
    return mean, sd


# ─────────────────────────── 自检 ───────────────────────────
def _weights_3x7(x21) -> pd.DataFrame:
    return pd.DataFrame(np.asarray(x21).reshape(3, 7),
                        index=["第 1 栋", "第 2 栋", "第 3 栋"],
                        columns=["电费", "碳排", "峰值", "舒适", "损耗", "波动", "韧性"])


def _oof_p01969(model: str):
    path = Path(str(OOF_GZ).format(model=model))
    if not path.exists():
        return []
    df = pd.read_csv(path, encoding="utf-8-sig")
    row = df[df.candidate_id == "p01969"]
    return [float(v) for v in row.y_pred]


def self_check():
    has_data = DATA.exists()
    if has_data:
        df = pd.read_csv(DATA, encoding="utf-8-sig")
        row = df[df.candidate_id == "p01969"].iloc[0]
        x21 = row[list(M.FEATURES)].to_numpy(float)
        truth_p = float(row["ramping"])
    else:
        bundle = M.load_model(MODELS / "gp-ard")
        x21 = bundle.payload["X_train"][0].astype(float)     # 数据缺失时只能做契约检查
        truth_p = float("nan")

    has_sub = SUBMISSION.exists()
    x_sub = None
    if has_sub:
        x_sub = M.pack_weights(pd.read_csv(SUBMISSION, encoding="utf-8-sig", index_col=0))

    rows = []
    for name in model_names():
        bundle = M.load_model(MODELS / name)
        saved = pd.read_csv(MODELS / name / "train_predictions.csv", encoding="utf-8-sig")

        # A) 加载后的全量预测 vs 保存的历史预测（决定性复核，应≈0）
        saved_mean = saved.y_pred.to_numpy(float)
        mean_all = bundle.predict(bundle.payload.get("gp", bundle.payload)["X_train"],
                                  return_std=False)
        d_saved_max = float(np.max(np.abs(mean_all - saved_mean))) if len(saved_mean) == len(mean_all) else float("nan")
        saved_p = float(saved.loc[saved.candidate_id == "p01969", "y_pred"].iloc[0])

        # B) 21 维向量 vs 3×7 中文表头（同值证明契约）
        m_vec = bundle.predict(x21.reshape(1, -1), return_std=False)[0]
        m_tab = bundle.predict(M.pack_weights(_weights_3x7(x21)), return_std=False)[0]

        # C) 与 OOF 留出预测的差（全量 refit vs 折内模型，属正常差异）
        oof_vals = _oof_p01969(name)
        d_oof = [abs(float(m_vec) - v) for v in oof_vals]

        # D) 第一组离线提交
        if x_sub is not None:
            pred_sub, sd_sub = bundle.predict(x_sub)
            sub_pred = float(pred_sub[0])
            d_hist = abs(sub_pred - SUBMISSION_HISTORICAL_GP)
            err_truth = abs(sub_pred - SUBMISSION_TRUTH)
        else:
            sub_pred, sd_sub, d_hist, err_truth = float("nan"), float("nan"), float("nan"), float("nan")

        # E) 反转 21 维顺序，预测必须改变（证明没有静默重排）
        m_rev = bundle.predict(x21[::-1].reshape(1, -1), return_std=False)[0]

        rows.append({
            "model": name,
            "p01969_actual": truth_p,
            "p01969_pred": float(m_vec),
            "p01969_via_3x7_cn": float(m_tab),
            "p01969_saved_insample": saved_p,
            "diff_vs_saved_insample_p01969": abs(float(m_vec) - saved_p),
            "diff_vs_saved_insample_max": d_saved_max,
            "diff_vs_oof": d_oof,
            "submission_pred": sub_pred,
            "submission_sd": float(sd_sub[0]) if x_sub is not None else float("nan"),
            "diff_vs_historical_gp": d_hist,
            "abs_err_vs_truth_119.712": err_truth,
            "permutation_changes_prediction": bool(abs(float(m_rev) - float(m_vec)) > 1e-9),
        })

    summary = {
        "package": "SDM374_backup_models_v1",
        "feature_contract": {
            "n_features": M.N_FEATURES,
            "order": list(M.FEATURES),
            "readable": "建筑1→建筑3；每栋 cost→carbon→peak→discomfort→degradation→ramping→resilience",
        },
        "checks": {
            "A_load_vs_saved_insample_max_abs": max(r["diff_vs_saved_insample_max"] for r in rows),
            "A_load_vs_saved_insample_p01969_max_abs": max(r["diff_vs_saved_insample_p01969"] for r in rows),
            "B_21vec_vs_3x7_chinese_max_abs": max(abs(r["p01969_pred"] - r["p01969_via_3x7_cn"]) for r in rows),
            "C_load_vs_OOF_p01969_max_abs": max((d for r in rows for d in r["diff_vs_oof"]), default=float("nan")),
            "D_load_vs_historical_gp_submission_max_abs": max(r["diff_vs_historical_gp"] for r in rows),
            "E_all_permutation_tests_changed": all(r["permutation_changes_prediction"] for r in rows),
        },
        "references": {"submission_gp": SUBMISSION_HISTORICAL_GP, "submission_truth": SUBMISSION_TRUTH,
                       "dataset_present": has_data, "submission_present": has_sub},
        "rows": rows,
    }
    out = HERE / "self_check.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 104)
    print(f"特征契约: {M.N_FEATURES} 维 | " + " ".join(M.FEATURES))
    print("=" * 104)
    print(f"{'model':22s} {'p01969':>10s} {'保存值':>10s} {'|差|_max':>12s} {'OOF差(两折)':>18s} "
          f"{'提交预测':>10s} {'vs历史':>8s} {'vs真值':>8s}")
    for r in rows:
        oofs = ",".join(f"{d:.4f}" for d in r["diff_vs_oof"]) or "-"
        print(f"{r['model']:22s} {r['p01969_pred']:10.5f} {r['p01969_saved_insample']:10.5f} "
              f"{r['diff_vs_saved_insample_max']:12.3e} {oofs:>18s} {r['submission_pred']:10.4f} "
              f"{r['diff_vs_historical_gp']:8.4f} {r['abs_err_vs_truth_119.712']:8.4f}")
    print("-" * 104)
    c = summary["checks"]
    print(f"A 加载 vs 保存的全量 in-sample 预测   最大绝对差 = {c['A_load_vs_saved_insample_max_abs']:.3e}（应≈0）")
    print(f"B 21 维向量 vs 3×7 中文表头         最大绝对差 = {c['B_21vec_vs_3x7_chinese_max_abs']:.3e}（应≈0）")
    print(f"C 加载 vs p01969 的 OOF 留出预测      最大绝对差 = {c['C_load_vs_OOF_p01969_max_abs']:.4f}")
    print(f"D 加载 vs 历史 gp 提交预测(121.489)   最大绝对差 = {c['D_load_vs_historical_gp_submission_max_abs']:.4f}")
    print(f"E 反转特征顺序是否都改变预测: {c['E_all_permutation_tests_changed']}")
    print(f"自检明细已写入 {out}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="SDM374 备用代理模型包 加载/预测/自检")
    ap.add_argument("--list", action="store_true", help="列出 models/registry.csv")
    ap.add_argument("--model", help="模型名（见 --list）")
    ap.add_argument("--weights", help="21 维逗号分隔，或 JSON 数组")
    ap.add_argument("--weights-file", help="csv/json/npy/txt 权重文件")
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出预测结果")
    a = ap.parse_args()

    if a.list:
        print(M.load_registry().to_string(index=False))
        return
    if a.self_check:
        self_check()
        return
    if not a.model:
        ap.error("需要 --model 或 --self-check 或 --list")
    if a.model not in model_names():
        ap.error(f"未知模型 {a.model}；可选: {', '.join(model_names())}")
    if a.weights_file:
        w = read_weights(Path(a.weights_file))
    elif a.weights:
        w = weights_from_string(a.weights)
    else:
        ap.error("需要 --weights 或 --weights-file")

    mean, sd = predict(a.model, w)
    if a.json:
        print(json.dumps({"model": a.model, "n": int(len(mean)),
                          "mean": mean.tolist(), "std": sd.tolist()}, ensure_ascii=False, indent=2))
    else:
        for i in range(len(mean)):
            print(f"[{a.model}] #{i + 1}  mean={mean[i]:.6f}  std={sd[i]:.6f}")


if __name__ == "__main__":
    main()
