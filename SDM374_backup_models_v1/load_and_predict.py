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
CSV 先按**列名集合**判定布局，再决定是否取某一列作索引/建筑列：
  - 无索引 21 列宽表（列名 = 21 个规范特征，顺序任意）→ 保留全部 21 列；
  - 3×7 + 建筑列（建筑标识可为 1/2/3、b1/b2/b3、building_1、楼1、第 1 栋…）→ 按建筑标签重排。
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
def _csv_layout_error(df, reason):
    raise M.ContractError(
        f"CSV 权重表布局无法识别（{reason}）：识别到 {df.shape[0]} 行 × {df.shape[1]} 列，"
        f"列名 {[str(c) for c in df.columns]}。期望下列之一：\n"
        "  ① 1×21 无索引宽表：列名恰好是 21 个规范特征（b1_cost … b3_resilience，顺序任意）；\n"
        "  ② 3×7 + 建筑列：7 个指标列（中文/英文），外加 1 列可识别的建筑标识"
        "（1/2/3、b1/b2/b3、building_1、楼1、第 1 栋 等）；\n"
        "  ③ n×21：每行一个完整权重向量（可带 1 列建筑/索引列，读取时丢弃该列）。")


def _weights_from_csv(df):
    """按**列名集合**判定 CSV 布局，再决定是否把某一列当作索引/建筑列。

    判定不看“第 0 列是否为数字”，只看列名覆盖情况：
      - 21 个规范特征列（顺序任意）→ 无索引宽表，保留全部 21 列；
      - 21 个规范特征列 + 1 个附加列 → 附加列是索引/建筑标识，丢弃后仍保留 21 列；
      - 7 个指标列 + 1 个可识别建筑列 → 把该列设为索引，按建筑标签重排；
      - 判不清一律抛 ContractError，消息带识别到的形状、列名与期望布局。
    """
    cols = [str(c) for c in df.columns]
    norm = [M._norm_label(c) for c in cols]
    canon = {M._norm_label(f) for f in M.FEATURES}

    feature_positions = [i for i, n in enumerate(norm) if n in canon]
    feature_labels = [norm[i] for i in feature_positions]

    # 布局 ① / ③：21 个规范特征列（顺序任意），可再多 1 列建筑/索引
    if len(feature_positions) == M.N_FEATURES and len(set(feature_labels)) == M.N_FEATURES:
        extra = [c for i, c in enumerate(cols) if i not in feature_positions]
        if len(extra) > 1:
            _csv_layout_error(df, f"21 个规范特征列之外还有 {len(extra)} 个附加列 {extra}")
        if extra:
            df = df.drop(columns=extra)      # 附加列是索引/建筑标识，21 个特征一个都不能丢
        return M.pack_weights(df)

    if feature_positions:
        _csv_layout_error(
            df, f"只识别到 {len(feature_positions)} 个规范特征列"
                f"（缺 {M.N_FEATURES - len(feature_positions)} 个，或部分列名不在规范名内）")

    # 布局 ②：7 个指标列 + 1 个建筑列
    metric_positions = [i for i, c in enumerate(cols) if M._metric_label(c) is not None]
    others = [i for i in range(len(cols)) if i not in metric_positions]
    if len(metric_positions) == M.N_METRICS and len(others) == 1:
        building = df.iloc[:, others[0]]
        labels = [M._building_label(v) for v in building]
        bad = [v for v, label in zip(building, labels) if label is None]
        if bad:
            _csv_layout_error(df, f"附加列 {cols[others[0]]!r} 的值 {bad[:5]} 不是可识别的建筑标签")
        if len(set(labels)) != len(labels):
            _csv_layout_error(df, f"建筑列 {cols[others[0]]!r} 含重复建筑 {list(building)}")
        metric_df = df.iloc[:, metric_positions].copy()
        metric_df.index = labels
        return M.pack_weights(metric_df)

    if len(cols) not in (M.N_METRICS, M.N_FEATURES):
        _csv_layout_error(df, "列数既不是 7（指标）也不是 21（规范特征）")
    return M.pack_weights(df)                # 7/21 列的具体标签问题交给 pack_weights 说明


def read_weights(path: Path):
    """读取 csv / json / npy / txt 形式的权重。

    CSV 不再无条件 `index_col=0`：先读全部列，再按列名集合判定布局（见 `_weights_from_csv`）。
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(obj, dict):
            for key in ("weights", "candidate_weights", "weight", "w"):
                if key in obj:
                    obj = obj[key]
                    break
            else:
                raise M.ContractError(
                    f"{path.name}: JSON 顶层对象缺少权重键（weights/candidate_weights/weight/w），"
                    f"得到键 {sorted(obj)}")
        return M.pack_weights(obj)
    if suffix == ".npy":
        return M.pack_weights(np.load(path))
    if suffix == ".csv":
        return _weights_from_csv(pd.read_csv(path, encoding="utf-8-sig"))
    txt = path.read_text(encoding="utf-8-sig").strip().replace("\n", ",")
    try:
        values = [float(x) for x in txt.split(",") if x.strip()]
    except ValueError as exc:
        raise M.ContractError(f"{path.name}: 文本权重不是合法的逗号分隔数字（{exc}）") from exc
    return M.pack_weights(values)


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
