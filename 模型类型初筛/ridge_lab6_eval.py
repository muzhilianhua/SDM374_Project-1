"""Lab6 SurrogateEvaluator —— 二次岭回归寻优适用性专项评估

评估器：Lab6/surrogate_evaluator.py，四个维度：
    1. 最优 5% 局部精度（MAE / RMSE / R² / 偏差方向）
    2. 排名一致性（Spearman 秩相关）
    3. 子集模拟寻优（模型挑出的 Top-K 在真实 y 里排第几）
    4. 多模型对比

评估设计：
    - 全量 2048 样本 + 5 折 OOF 预测（每样本均为折外预测，无泄漏），
      最优 5% 区域约 102 个样本，比 8:2 holdout 的 21 个统计更稳
    - 子集模拟寻优 5 折（KFold random_state=2026，评估器内部固定）
    - 对比模型：一阶岭回归（degree=1，alpha 同法 CV 选取），衡量二次项带来的增益
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

SEED = 2026
np.random.seed(SEED)

METRICS = ("cost", "carbon", "peak", "discomfort", "degradation", "ramping", "resilience")
FEATURES = [f"b{b}_{m}" for b in range(1, 4) for m in METRICS]

LAB6 = Path(r"D:\HuaweiMoveData\Users\11659\Desktop\机器学习项目\Lab6")
if not (LAB6 / "surrogate_evaluator.py").exists():
    raise FileNotFoundError(f"找不到评估器: {LAB6 / 'surrogate_evaluator.py'}")
sys.path.insert(0, str(LAB6))

from surrogate_evaluator import SurrogateEvaluator  # noqa: E402


def make_poly_ridge(degree, alpha):
    """多项式特征 + 标准化 + 岭回归流水线（所有步骤仅在训练折内拟合，防泄漏）。"""
    return Pipeline(
        [
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, solver="auto")),
        ]
    )


# ──────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────
df = pd.read_csv("dataset_same8/train.csv", float_precision="round_trip")
X = df[FEATURES].to_numpy(dtype=float)
y = df["ramping"].to_numpy(dtype=float)

print("=" * 70)
print("  Lab6 SurrogateEvaluator —— 二次岭回归寻优适用性专项评估")
print("=" * 70)
print(f"\n数据: {X.shape[0]} 条, {X.shape[1]} 维特征, 目标=ramping (越小越好)")

# ──────────────────────────────────────────────
# alpha 调参（全量 5 折 CV，选 RMSE）
# ──────────────────────────────────────────────
alphas = np.logspace(-3, 4, 29)
cv5 = KFold(n_splits=5, shuffle=True, random_state=SEED)

best_alphas = {}
for degree in (2, 1):
    gs = GridSearchCV(
        make_poly_ridge(degree, 1.0),
        param_grid={"ridge__alpha": alphas},
        scoring="neg_root_mean_squared_error",
        cv=cv5,
        n_jobs=-1,
    )
    gs.fit(X, y)
    best_alphas[degree] = float(gs.best_params_["ridge__alpha"])
    print(f"  degree={degree}: 最优 alpha={best_alphas[degree]:.4g}  "
          f"CV RMSE={-gs.best_score_:.4f}")

# ──────────────────────────────────────────────
# 5 折 OOF 预测（全量，逐样本折外）
# ──────────────────────────────────────────────
oof_preds = {}
for degree in (2, 1):
    oof = cross_val_predict(
        make_poly_ridge(degree, best_alphas[degree]), X, y, cv=cv5, n_jobs=-1
    )
    oof_preds[degree] = oof
    print(f"  degree={degree}: 全量 OOF MAE={mean_absolute_error(y, oof):.4f}")

# ──────────────────────────────────────────────
# Lab6 评估（局部精度 / 排名一致性 / 子集寻优）
# ──────────────────────────────────────────────
ev = SurrogateEvaluator(X, y, feature_names=FEATURES)

reports = {}
for degree in (2, 1):
    name = f"QuadraticRidge(degree=2, alpha={best_alphas[2]:.4g})" if degree == 2 \
        else f"LinearRidge(degree=1, alpha={best_alphas[1]:.4g})"
    print(f"\n{'=' * 70}\n  {name} —— Lab6 评估（全量 OOF + 5 折子集寻优）\n{'=' * 70}")
    report = ev.evaluate(
        make_poly_ridge(degree, best_alphas[degree]).fit(X, y),
        model_name=name,
        oof_predictions=oof_preds[degree],
        run_subset=True,
        model_factory=lambda d=degree: make_poly_ridge(d, best_alphas[d]),
        subset_folds=5,
        subset_topk=10,
    )
    print(report.summary())
    reports[degree] = report

# ──────────────────────────────────────────────
# 多模型对比
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("  多模型对比（Lab6 compare_models）")
print("=" * 70)
print(ev.compare_models([reports[2], reports[1]]).to_string(index=False))

# 子集寻优明细
print("\n--- 子集模拟寻优明细（Top-K 真实分位越低越好，通过线=前 10%）---")
for degree in (2, 1):
    for r in reports[degree].subset_runs:
        tag = "✅" if r.pass_topk else "❌"
        print(f"  degree={degree} 折{r.fold}: Top-{r.k} 平均分位={r.mean_true_percentile*100:5.1f}%  "
              f"最优值={r.best_true_value:.1f}  (holdout 最优={r.holdout_best_value:.1f}, "
              f"gap={r.gap:+.1f}, 真实排名={r.best_true_rank}/{r.n_holdout}) {tag}")

print(f"\n复现参数：SEED={SEED}, 全量 5 折 OOF (KFold shuffle, random_state={SEED}), "
      f"degree=2 alpha={best_alphas[2]:.4g}, degree=1 alpha={best_alphas[1]:.4g}, "
      f"subset_folds=5, subset_topk=10")
