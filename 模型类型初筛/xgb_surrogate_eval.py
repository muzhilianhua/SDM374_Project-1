"""用 Lab6/surrogate_evaluator.py 评估 XGBoost 代理模型（ramping 预测）。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from xgboost import XGBRegressor

LAB6 = Path(r"D:\HuaweiMoveData\Users\11659\Desktop\机器学习项目\Lab6")
sys.path.insert(0, str(LAB6))

from surrogate_evaluator import SurrogateEvaluator  # noqa: E402

SEED = 2026
METRICS = ("cost", "carbon", "peak", "discomfort", "degradation", "ramping", "resilience")
FEATURES = [f"b{b}_{m}" for b in range(1, 4) for m in METRICS]

# xgb_regression.py 中验证集选出的最优配置
BEST = dict(
    n_estimators=600,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method="hist",
    random_state=SEED,
    n_jobs=-1,
)


def make_xgb():
    return XGBRegressor(**BEST)


df = pd.read_csv("dataset_same8/train.csv", float_precision="round_trip")
X = df[FEATURES].to_numpy(dtype=float)
y = df["ramping"].to_numpy(dtype=float)

print(f"加载数据：{X.shape[0]} 条，{X.shape[1]} 维特征")

# 5 折 OOF 预测，避免用训练集自预测造成指标虚高
oof = cross_val_predict(make_xgb(), X, y, cv=5, method="predict")

model = make_xgb().fit(X, y)

ev = SurrogateEvaluator(X, y, feature_names=FEATURES)
report = ev.evaluate(
    model,
    model_name="XGBoost(depth=5, lr=0.05, n=600)",
    oof_predictions=oof,
    run_subset=True,
    model_factory=make_xgb,
    subset_folds=3,
    subset_topk=10,
)
print(report.summary())

print("\n--- OOF 全样本指标 ---")
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402

print(f"  MAE={mean_absolute_error(y, oof):.3f}  "
      f"RMSE={np.sqrt(mean_squared_error(y, oof)):.3f}  R²={r2_score(y, oof):.4f}")

print("\n--- 多模型对比表 ---")
print(ev.compare_models([report]).to_string(index=False))
