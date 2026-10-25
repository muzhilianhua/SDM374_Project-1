"""用 Lab6/surrogate_evaluator.py 的 SurrogateEvaluator 评估随机森林的寻优适用性。

评估范式（遵循评估器设计）：
  - 全量 2048 数据构造评估器
  - 5 折 OOF 预测 -> 最优5%局部精度 + Spearman 排名一致性（无泄漏）
  - 3 折子集模拟寻优 -> 模型在 holdout 候选池挑 Top-10，看真实分位
  - 与 GBDT 对比，compare_models 汇总

衔接上文：RF 配置与 rf_regression.py 调参结果一致 (n_estimators=800, max_depth=None)。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_predict

# 导入 Lab6 的评估器
LAB6 = Path(r"d:\HuaweiMoveData\Users\11659\Desktop\机器学习项目\Lab6")
sys.path.insert(0, str(LAB6))
from surrogate_evaluator import SurrogateEvaluator

SEED = 2026
np.random.seed(SEED)

METRICS = ("cost", "carbon", "peak", "discomfort", "degradation", "ramping", "resilience")
FEATURES = [f"b{b}_{m}" for b in range(1, 4) for m in METRICS]

df = pd.read_csv("dataset_same8/train.csv", float_precision="round_trip")
X = df[FEATURES].values.astype(float)
y = df["ramping"].values.astype(float)
print(f"数据: {X.shape[0]} 条, {X.shape[1]} 维, ramping ∈ [{y.min():.2f}, {y.max():.2f}]")

ev = SurrogateEvaluator(X, y, feature_names=FEATURES)

# ---------- 模型工厂（与之前调参结果一致） ----------
def rf_factory():
    return RandomForestRegressor(
        n_estimators=800, max_depth=None, min_samples_leaf=2,
        random_state=SEED, n_jobs=-1,
    )

def gbdt_factory():
    return GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.03,
        min_samples_leaf=5, subsample=0.8, random_state=SEED,
    )

reports = []
for name, factory in [("随机森林RF", rf_factory), ("GBDT", gbdt_factory)]:
    print(f"\n>>> 评估 {name} ...")
    # 5 折 OOF（每次新建模型，避免泄漏）
    oof = cross_val_predict(factory(), X, y, cv=5, method="predict")
    model = factory().fit(X, y)
    report = ev.evaluate(
        model, model_name=name,
        oof_predictions=oof,
        run_subset=True, model_factory=factory,
        subset_folds=3, subset_topk=10,
    )
    print(report.summary())
    reports.append(report)

# ---------- 汇总对比 ----------
print("\n" + "=" * 60)
print("  多模型寻优适用性对比")
print("=" * 60)
table = SurrogateEvaluator.compare_models(reports)
print(table.to_string(index=False))
