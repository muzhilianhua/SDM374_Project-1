"""二次岭回归（Quadratic Ridge Regression）预测 ramping (R)：8:2 分层划分 + 拟合曲线 + Lab6 评估器

模型：二阶多项式特征展开 + 岭回归（L2 正则线性回归）
    对 21 维权重特征 x 构造全部一阶项与二阶交互项（PolynomialFeatures, degree=2），
        y = β₀ + Σ_j β_j x_j + Σ_{j≤k} β_{jk} x_j x_k + ε
    岭回归在 (Φ'Φ + αI)β = Φ'y 中求解，α 通过训练集 5 折 CV 在对数网格上选取。
    特征为组合权重（每栋楼 7 项和为 1），存在强共线，L2 正则正适合该场景。

仅依赖 numpy / pandas / scikit-learn。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# 固定随机种子，保证数据划分与 CV 折分均可复现
SEED = 2026
np.random.seed(SEED)

METRICS = ("cost", "carbon", "peak", "discomfort", "degradation", "ramping", "resilience")
FEATURES = [f"b{b}_{m}" for b in range(1, 4) for m in METRICS]


def make_ridge(alpha, seed=SEED):
    """二次特征 + 标准化 + 岭回归流水线（所有步骤仅在训练折内拟合，防泄漏）。"""
    return Pipeline(
        [
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, random_state=None, solver="auto")),
        ]
    )


# ──────────────────────────────────────────────
# 数据加载 + 8:2 分层划分
# ──────────────────────────────────────────────
df = pd.read_csv("dataset_same8/train.csv", float_precision="round_trip")
X = df[FEATURES].to_numpy(dtype=float)
y = df["ramping"].to_numpy(dtype=float)

# 连续目标按 10 分位分箱做分层标签
strata = pd.qcut(y, q=10, labels=False, duplicates="drop")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=strata, random_state=SEED, shuffle=True
)

print("=" * 70)
print("  二次岭回归 —— ramping (R) 预测 (8:2 分层划分)")
print("=" * 70)
print(f"\n划分结果 (random_state={SEED}, 分层标签=ramping 10 分位):")
print(f"  训练集: {len(y_train)} ({len(y_train)/len(y):.1%})")
print(f"  测试集: {len(y_test)} ({len(y_test)/len(y):.1%})")
print(f"  训练集 ramping: mean={y_train.mean():.2f}  std={y_train.std():.2f}  "
      f"min={y_train.min():.2f}  max={y_train.max():.2f}")
print(f"  测试集 ramping: mean={y_test.mean():.2f}  std={y_test.std():.2f}  "
      f"min={y_test.min():.2f}  max={y_test.max():.2f}")

n_poly = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X[:1]).shape[1]
print(f"\n--- 模型配置 ---")
print(f"  特征展开: degree=2, 21 维 -> {n_poly} 维（含全部二阶交互项）")
print(f"  正则: Ridge(L2), alpha 由训练集 5 折 CV 在对数网格上选取")

# ──────────────────────────────────────────────
# alpha 调参：训练集 5 折 CV
# ──────────────────────────────────────────────
alphas = np.logspace(-3, 4, 29)
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
gs = GridSearchCV(
    make_ridge(alpha=1.0),
    param_grid={"ridge__alpha": alphas},
    scoring="neg_root_mean_squared_error",
    cv=cv,
    n_jobs=-1,
    refit=True,
)
gs.fit(X_train, y_train)
best_alpha = float(gs.best_params_["ridge__alpha"])
cv_rmse = -gs.best_score_
print(f"\n--- alpha 调参 (训练集 5 折 CV, 目标=RMSE) ---")
print(f"  最优 alpha = {best_alpha:.4g}   CV RMSE = {cv_rmse:.4f}")
top5 = sorted(zip(alphas, -gs.cv_results_["mean_test_score"]), key=lambda t: t[1])[:5]
for a, s in top5:
    print(f"    alpha={a:>10.4g}  CV RMSE={s:.4f}")

# ──────────────────────────────────────────────
# 最终模型：全训练集拟合 + 测试集评估
# ──────────────────────────────────────────────
final_model = make_ridge(best_alpha).fit(X_train, y_train)
pred_test = final_model.predict(X_test)

# 训练集 5 折 OOF（每折单独固定种子，与其它脚本范式一致）
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
pred_train_oof = np.zeros_like(y_train)
for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train), start=1):
    fold_model = make_ridge(best_alpha).fit(X_train[tr_idx], y_train[tr_idx])
    pred_train_oof[va_idx] = fold_model.predict(X_train[va_idx])
    print(f"  OOF 折{fold}: MAE={mean_absolute_error(y_train[va_idx], pred_train_oof[va_idx]):.4f}")

train_mae = mean_absolute_error(y_train, pred_train_oof)
train_rmse = np.sqrt(mean_squared_error(y_train, pred_train_oof))
train_r2 = r2_score(y_train, pred_train_oof)

test_mae = mean_absolute_error(y_test, pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, pred_test))
test_r2 = r2_score(y_test, pred_test)
test_rho = spearmanr(y_test, pred_test).statistic

print("\n--- 最终模型评估 ---")
print(f"  训练集 (5 折 OOF): MAE={train_mae:.4f}  RMSE={train_rmse:.4f}  R²={train_r2:.4f}")
print(f"  测试集:            MAE={test_mae:.4f}  RMSE={test_rmse:.4f}  R²={test_r2:.4f}")
print(f"  测试集 Spearman ρ = {test_rho:.4f}")
print(f"  真实值 std={y_test.std():.3f} vs 残差 std={(y_test - pred_test).std():.3f}")

# ──────────────────────────────────────────────
# 拟合曲线可视化
# ──────────────────────────────────────────────
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# 图1: 测试集拟合曲线（按真实值排序 + 滑动平均趋势）
ax = axes[0]
order = np.argsort(y_test)
xs, ys = y_test[order], pred_test[order]
ax.plot(xs, xs, "k--", lw=1.5, label="理想拟合 y=x")
ax.scatter(xs, ys, c="steelblue", s=30, edgecolors="white",
           linewidths=0.4, zorder=3, label="测试集预测")
trend = pd.Series(ys).rolling(window=41, center=True, min_periods=5).mean().to_numpy()
ax.plot(xs, trend, c="tomato", lw=2.2, label="拟合趋势（滑动平均）")
ax.set_xlabel("真实 ramping")
ax.set_ylabel("预测 ramping")
ax.set_title(f"测试集拟合曲线 (R²={test_r2:.4f}, MAE={test_mae:.2f})")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)

# 图2: 训练集 OOF vs 测试集 预测散点
ax = axes[1]
ax.scatter(y_train, pred_train_oof, c="gray", s=10, alpha=0.4,
           label=f"训练集 OOF (R²={train_r2:.4f})")
ax.scatter(y_test, pred_test, c="steelblue", s=30, edgecolors="white",
           linewidths=0.4, zorder=3, label=f"测试集 (R²={test_r2:.4f})")
lims = [y.min() - 5, y.max() + 5]
ax.plot(lims, lims, "k--", lw=1.5, label="y=x")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("真实 ramping")
ax.set_ylabel("预测 ramping")
ax.set_title("训练集 vs 测试集 预测对比")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)

# 图3: 测试集残差
ax = axes[2]
residuals = y_test - pred_test
ax.scatter(pred_test, residuals, c="steelblue", s=30, edgecolors="white", linewidths=0.4)
ax.axhline(0, color="k", ls="--", lw=1.5)
ax.set_xlabel("预测 ramping")
ax.set_ylabel("残差 (真实 - 预测)")
ax.set_title(f"测试集残差分布 (std={residuals.std():.2f})")
ax.grid(alpha=0.3)

plt.tight_layout()
out_png = "ridge_fit_curves_strat82.png"
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"\n✓ 拟合曲线已保存: {out_png}")

# ──────────────────────────────────────────────
# Lab6 评估器
# ──────────────────────────────────────────────
LAB6 = Path(r"D:\HuaweiMoveData\Users\11659\Desktop\机器学习项目\Lab6")
if not (LAB6 / "surrogate_evaluator.py").exists():
    raise FileNotFoundError(f"找不到评估器: {LAB6 / 'surrogate_evaluator.py'}")
sys.path.insert(0, str(LAB6))

from surrogate_evaluator import SurrogateEvaluator  # noqa: E402

model_label = f"QuadraticRidge(degree=2, alpha={best_alpha:.4g})"

print("\n" + "=" * 70)
print("  Lab6 SurrogateEvaluator —— 测试集（8:2 holdout）评估")
print("=" * 70)
ev = SurrogateEvaluator(X_test, y_test, feature_names=FEATURES)
report = ev.evaluate(
    final_model,
    model_name=model_label,
    oof_predictions=pred_test,
    run_subset=True,
    model_factory=lambda: make_ridge(best_alpha),
    subset_folds=3,
    subset_topk=10,
)
print(report.summary())

print("\n--- 多模型对比表 ---")
print(ev.compare_models([report]).to_string(index=False))
print(f"\n复现参数：SEED={SEED}, test_size=0.20, strata=qcut(y, q=10), "
      f"degree=2, alpha={best_alpha:.4g}")
