"""HistGradientBoosting 预测 ramping (R)：8:2 分层划分 + 拟合曲线 + Lab6 评估器"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_predict,
    train_test_split,
)
from scipy.stats import spearmanr

# 固定随机种子，保证划分、交叉验证与模型训练均可复现
SEED = 2026
np.random.seed(SEED)

METRICS = ("cost", "carbon", "peak", "discomfort", "degradation", "ramping", "resilience")
FEATURES = [f"b{b}_{m}" for b in range(1, 4) for m in METRICS]

# ---------- 数据加载 ----------
df = pd.read_csv("dataset_same8/train.csv", float_precision="round_trip")
X = df[FEATURES].to_numpy(dtype=float)
y = df["ramping"].to_numpy(dtype=float)

# ---------- 8 : 2 分层划分 ----------
# 目标 ramping 是连续值，先按分位数切成 10 个区间作为分层标签，
# 再按 8:2 分层抽样，保证训练/测试集的目标分布一致。
strata = pd.qcut(y, q=10, labels=False, duplicates="drop")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=strata, random_state=SEED, shuffle=True
)

print("=" * 68)
print("  HistGradientBoosting 回归 —— ramping (R) 预测 (8:2 分层划分)")
print("=" * 68)
print(f"\n划分结果 (random_state={SEED}, 分层标签=ramping 10 分位):")
print(f"  训练集: {len(y_train)} ({len(y_train)/len(y):.1%})")
print(f"  测试集: {len(y_test)} ({len(y_test)/len(y):.1%})")
print(f"  训练集 ramping: mean={y_train.mean():.2f}  std={y_train.std():.2f}  "
      f"min={y_train.min():.2f}  max={y_train.max():.2f}")
print(f"  测试集 ramping: mean={y_test.mean():.2f}  std={y_test.std():.2f}  "
      f"min={y_test.min():.2f}  max={y_test.max():.2f}")

# ---------- 训练集内 5 折交叉验证选超参 ----------
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

param_grid = {
    "max_depth": [4, 6, None],
    "learning_rate": [0.05, 0.1],
    "max_iter": [300, 600],
}

print("\n--- 训练集 5 折交叉验证超参搜索 (MAE 最小) ---")


def make_histgb(**overrides):
    params = dict(
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=SEED,
    )
    params.update(overrides)
    return HistGradientBoostingRegressor(**params)


gs = GridSearchCV(
    make_histgb(),
    param_grid,
    scoring="neg_mean_absolute_error",
    cv=cv,
    n_jobs=1,
    refit=True,
)
gs.fit(X_train, y_train)

best_params = gs.best_params_
print(f"  ★ 最优配置: {best_params}")
print(f"  CV MAE = {-gs.best_score_:.4f}")


# 固定最优配置的模型工厂，供后续交叉验证与评估器复用
def make_best_histgb():
    return make_histgb(**best_params)


# ---------- 最终模型：在训练集上重训 ----------
model = make_best_histgb().fit(X_train, y_train)

pred_test = model.predict(X_test)
pred_train_oof = cross_val_predict(make_best_histgb(), X_train, y_train, cv=cv, n_jobs=-1)

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

# ---------- 拟合曲线可视化 ----------
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# 图1: 测试集拟合曲线（按真实值排序 + 滑动平均趋势线）
ax = axes[0]
order = np.argsort(y_test)
xs, ys = y_test[order], pred_test[order]
ax.plot(xs, xs, "k--", lw=1.5, label="理想拟合 y=x")
ax.scatter(xs, ys, c="seagreen", s=30, edgecolors="white",
           linewidths=0.4, zorder=3, label="测试集预测")
# 对排序后的预测值做滑动平均，得到平滑的拟合趋势
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
ax.scatter(y_test, pred_test, c="seagreen", s=30, edgecolors="white",
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
ax.scatter(pred_test, residuals, c="seagreen", s=30, edgecolors="white", linewidths=0.4)
ax.axhline(0, color="k", ls="--", lw=1.5)
ax.set_xlabel("预测 ramping")
ax.set_ylabel("残差 (真实 - 预测)")
ax.set_title(f"测试集残差分布 (std={residuals.std():.2f})")
ax.grid(alpha=0.3)

plt.tight_layout()
out_png = "histgb_fit_curves_strat82.png"
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"\n✓ 拟合曲线已保存: {out_png}")

# ---------- Lab6 评估器 ----------
LAB6 = Path(r"D:\HuaweiMoveData\Users\11659\Desktop\机器学习项目\Lab6")
if not (LAB6 / "surrogate_evaluator.py").exists():
    raise FileNotFoundError(f"找不到评估器: {LAB6 / 'surrogate_evaluator.py'}")
sys.path.insert(0, str(LAB6))

from surrogate_evaluator import SurrogateEvaluator  # noqa: E402

print("\n" + "=" * 68)
print("  Lab6 SurrogateEvaluator —— 测试集（8:2 holdout）评估")
print("=" * 68)

ev = SurrogateEvaluator(X_test, y_test, feature_names=FEATURES)
report = ev.evaluate(
    model,
    model_name=f"HistGB(8:2分层, depth={best_params['max_depth']}, "
               f"lr={best_params['learning_rate']}, n={best_params['max_iter']})",
    oof_predictions=pred_test,   # 模型未见过测试集，预测即为样本外结果
    run_subset=True,
    model_factory=make_best_histgb,
    subset_folds=3,
    subset_topk=10,
)
print(report.summary())

print("\n--- 多模型对比表 ---")
print(ev.compare_models([report]).to_string(index=False))
print(f"\n复现参数：SEED={SEED}, test_size=0.20, strata=qcut(y, q=10)")
