"""SPMM 半参数混合模型预测 ramping (R)：8:2 分层划分 + 拟合曲线 + Lab6 评估器

SPMM = Semiparametric Mixed Model：
    参数项   —— 二次多项式 + 岭回归，刻画全局线性/二次趋势
    非参数项 —— RBF 核岭回归拟合参数项残差，刻画局部非线性
最终预测 = 参数项预测 + 非参数残差修正。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from scipy.stats import spearmanr

# 固定随机种子，保证划分、交叉验证与模型训练均可复现
SEED = 2026
np.random.seed(SEED)

METRICS = ("cost", "carbon", "peak", "discomfort", "degradation", "ramping", "resilience")
FEATURES = [f"b{b}_{m}" for b in range(1, 4) for m in METRICS]


# ---------- SPMM：半参数混合模型 ----------
class SemiparametricMixedModel(BaseEstimator, RegressorMixin):
    """参数项（二次岭回归）+ 非参数项（RBF 核岭回归残差修正）。"""

    def __init__(self, alpha_linear=100.0, alpha_kernel=1.0, gamma=0.05, degree=2):
        self.alpha_linear = alpha_linear
        self.alpha_kernel = alpha_kernel
        self.gamma = gamma
        self.degree = degree

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # 参数项：标准化 -> 二次多项式展开 -> 岭回归
        self.scaler_ = StandardScaler().fit(X)
        Z = self.scaler_.transform(X)
        self.poly_ = PolynomialFeatures(self.degree, include_bias=False).fit(Z)
        P = self.poly_.transform(Z)
        self.linear_ = Ridge(alpha=self.alpha_linear).fit(P, y)

        # 非参数项：对参数项的残差做 RBF 核岭回归
        residual = y - self.linear_.predict(P)
        self.kernel_ = KernelRidge(
            kernel="rbf", alpha=self.alpha_kernel, gamma=self.gamma
        ).fit(Z, residual)

        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Z = self.scaler_.transform(X)
        P = self.poly_.transform(Z)
        return self.linear_.predict(P) + self.kernel_.predict(Z)


def make_spmm(**params):
    """模型工厂：供交叉验证与 Lab6 子集模拟复用。"""
    return SemiparametricMixedModel(**{**BEST_PARAMS, **params})


# ---------- 数据加载 ----------
df = pd.read_csv("dataset_same8/train.csv", float_precision="round_trip")
X = df[FEATURES].to_numpy(dtype=float)
y = df["ramping"].to_numpy(dtype=float)

# ---------- 8 : 2 分层划分 ----------
# ramping 为连续目标，先按 10 分位分箱作为分层标签，再按 8:2 分层抽样
strata = pd.qcut(y, q=10, labels=False, duplicates="drop")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=strata, random_state=SEED, shuffle=True
)

print("=" * 70)
print("  SPMM 半参数混合模型 —— ramping (R) 预测 (8:2 分层划分)")
print("=" * 70)
print(f"\n划分结果 (random_state={SEED}, 分层标签=ramping 10 分位):")
print(f"  训练集: {len(y_train)} ({len(y_train)/len(y):.1%})")
print(f"  测试集: {len(y_test)} ({len(y_test)/len(y):.1%})")
print(f"  训练集 ramping: mean={y_train.mean():.2f}  std={y_train.std():.2f}  "
      f"min={y_train.min():.2f}  max={y_train.max():.2f}")
print(f"  测试集 ramping: mean={y_test.mean():.2f}  std={y_test.std():.2f}  "
      f"min={y_test.min():.2f}  max={y_test.max():.2f}")

# ---------- 训练集内 5 折交叉验证选超参 ----------
# BEST_PARAMS 先置为默认值，网格搜索后更新，供模型工厂使用
BEST_PARAMS = dict(alpha_linear=100.0, alpha_kernel=1.0, gamma=0.05, degree=2)

cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
param_grid = {
    "alpha_linear": [10.0, 100.0, 300.0],
    "alpha_kernel": [0.01, 0.03, 0.1, 0.3],
    "gamma": [0.05, 0.1, 0.2],
}

print("\n--- 训练集 5 折交叉验证超参搜索 (MAE 最小) ---")
gs = GridSearchCV(
    SemiparametricMixedModel(),
    param_grid,
    scoring="neg_mean_absolute_error",
    cv=cv,
    n_jobs=-1,
    refit=True,
)
gs.fit(X_train, y_train)
BEST_PARAMS = gs.best_params_
print(f"  ★ 最优配置: {BEST_PARAMS}")
print(f"  CV MAE = {-gs.best_score_:.4f}")

# ---------- 最终模型：在训练集上重训 ----------
model = make_spmm().fit(X_train, y_train)
pred_test = model.predict(X_test)
pred_train_oof = cross_val_predict(make_spmm(), X_train, y_train, cv=cv, n_jobs=-1)

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

# 参数项 / 非参数项的贡献占比
Z_test = model.scaler_.transform(X_test)
param_part = model.linear_.predict(model.poly_.transform(Z_test))
nonparam_part = model.kernel_.predict(Z_test)
print(f"\n  测试集参数项贡献: std={param_part.std():.3f}")
print(f"  测试集非参数残差修正: std={nonparam_part.std():.3f}")

# ---------- 拟合曲线可视化 ----------
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# 图1: 测试集拟合曲线（按真实值排序 + 滑动平均趋势线）
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
out_png = "spmm_fit_curves.png"
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"\n✓ 拟合曲线已保存: {out_png}")

# ---------- Lab6 评估器 ----------
LAB6 = Path(r"D:\HuaweiMoveData\Users\11659\Desktop\机器学习项目\Lab6")
if not (LAB6 / "surrogate_evaluator.py").exists():
    raise FileNotFoundError(f"找不到评估器: {LAB6 / 'surrogate_evaluator.py'}")
sys.path.insert(0, str(LAB6))

from surrogate_evaluator import SurrogateEvaluator  # noqa: E402

model_label = (
    f"SPMM(半参数混合, alpha_l={BEST_PARAMS['alpha_linear']}, "
    f"alpha_k={BEST_PARAMS['alpha_kernel']}, gamma={BEST_PARAMS['gamma']})"
)

print("\n" + "=" * 70)
print("  Lab6 SurrogateEvaluator —— 测试集（8:2 holdout）评估")
print("=" * 70)
ev_test = SurrogateEvaluator(X_test, y_test, feature_names=FEATURES)
report_test = ev_test.evaluate(
    model,
    model_name=model_label,
    oof_predictions=pred_test,   # 模型未见过测试集，预测即为样本外结果
    run_subset=True,
    model_factory=make_spmm,
    subset_folds=3,
    subset_topk=10,
)
print(report_test.summary())

# 测试集仅 410 条，最优 5% 局部样本约 21 条，指标波动较大；
# 额外在全量数据上做 5 折 OOF 评估，作为更稳定的参考。
print("\n" + "=" * 70)
print("  Lab6 SurrogateEvaluator —— 全量数据 5 折 OOF 评估（稳定参考）")
print("=" * 70)
cv_full = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof_full = cross_val_predict(make_spmm(), X, y, cv=cv_full, n_jobs=-1)
model_full = make_spmm().fit(X, y)

ev_full = SurrogateEvaluator(X, y, feature_names=FEATURES)
report_full = ev_full.evaluate(
    model_full,
    model_name=model_label + " [全量OOF]",
    oof_predictions=oof_full,
    run_subset=True,
    model_factory=make_spmm,
    subset_folds=3,
    subset_topk=10,
)
print(report_full.summary())

print("\n--- 多模型对比表 ---")
print(ev_full.compare_models([report_test, report_full]).to_string(index=False))
print(f"\n复现参数：SEED={SEED}, test_size=0.20, strata=qcut(y, q=10), "
      f"best_params={BEST_PARAMS}")
