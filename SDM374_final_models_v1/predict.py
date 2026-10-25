"""SDM374 最终代理模型预测入口（3 个独立全量模型）。

用法：
    python predict.py --input new_test.csv --out preds.csv
    python predict.py --input new_test.csv --out preds.csv --model histgb_low80_full
    python predict.py --input new_test.csv --out preds.csv --id-column candidate_id

输入 CSV 必须包含 MANIFEST.json 中的 21 个特征列（列顺序不限，按列名取值）。
若输入同时包含目标列（默认 ramping，可用 --target-column 指定），会额外输出指标文件。
每个模型输出一列预测；三个模型相互独立，可单独使用。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PKG_DIR = Path(__file__).resolve().parent
MANIFEST = json.loads((PKG_DIR / "MANIFEST.json").read_text(encoding="utf-8"))
FEATURES = MANIFEST["feature_columns"]
TARGET = MANIFEST["target_column"]
MODELS_DIR = PKG_DIR / "models"


def main() -> int:
    ap = argparse.ArgumentParser(description="SDM374 final-model predictor")
    ap.add_argument("--input", required=True, help="测试集 CSV")
    ap.add_argument("--out", required=True, help="预测结果 CSV")
    ap.add_argument("--model", default="all",
                    help="xgb_w20x3_full / histgb_low80_full / histgb_low60_full / all")
    ap.add_argument("--id-column", default=None, help="作为输出首列的标识列（默认生成 row_id）")
    ap.add_argument("--target-column", default=TARGET, help="目标列名，用于计算指标")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        print(f"ERROR: 输入缺少 {len(missing)} 个特征列: {missing}", file=sys.stderr)
        return 1

    X = df[FEATURES].to_numpy(dtype=np.float64)
    n_nan = int(np.isnan(X).sum())
    if n_nan:
        print(f"WARNING: 特征中存在 {n_nan} 个缺失值。HistGB 原生支持 NaN；"
              f"XGBoost 模型对 NaN 会报错。", file=sys.stderr)

    names = sorted(MANIFEST["models"].keys()) if args.model == "all" else [args.model]
    for n in names:
        if n not in MANIFEST["models"]:
            print(f"ERROR: 未知模型 {n}，可选: {sorted(MANIFEST['models'])}", file=sys.stderr)
            return 1

    if args.id_column and args.id_column in df.columns:
        out = pd.DataFrame({args.id_column: df[args.id_column]})
    else:
        out = pd.DataFrame({"row_id": np.arange(len(df))})

    for name in names:
        est = joblib.load(MODELS_DIR / f"{name}.joblib")
        out[name] = np.asarray(est.predict(X), dtype=np.float64)
        print(f"[{name}] {len(df)} 行预测完成")

    out.to_csv(args.out, index=False, float_format="%.17g")
    print(f"已写出预测: {args.out}  ({len(out)} 行 x {out.shape[1] - 1} 个预测列)")

    if args.target_column in df.columns:
        y = df[args.target_column].to_numpy(dtype=np.float64)
        rows = []
        for name in names:
            err = out[name].to_numpy() - y
            rows.append(dict(model=name, n=len(y),
                             rmse=float(np.sqrt(np.mean(err ** 2))),
                             mae=float(np.mean(np.abs(err))),
                             r2=float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2))))
        mpath = Path(args.out).with_name(Path(args.out).stem + "_metrics.csv")
        pd.DataFrame(rows).to_csv(mpath, index=False)
        print(f"已写出指标: {mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
