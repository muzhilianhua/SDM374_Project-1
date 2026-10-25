"""Bayesian GAM 预测 ramping (R)：8:2 分层划分 + 拟合曲线 + Lab6 评估器

Bayesian GAM = Bayesian P-spline 广义可加模型（Eilers & Marx 1996, Lang & Brezger 2004）：
    对每个特征 x_j 构造三次 B-样条基 B_j(x_j)，模型为
        y = α + Σ_j B_j(x_j) β_j + ε,   ε ~ N(0, σ²)
    先验（Bayesian P-spline）：
        β_j ~ N(0, τ_j^{-1} (D₂'D₂ + κI)^{-1})   二阶随机游走光滑先验
        τ_j ~ Gamma(a_τ, b_τ)                      每个光滑项的精度超先验
        σ^{-2} ~ Gamma(a_σ, b_σ)                   噪声精度超先验
    通过 Gibbs 采样对 (β, τ_1..τ_J, σ²) 联合后验推断，预测取后验样本均值。

仅依赖 numpy / scipy / scikit-learn，无需额外安装 pyGAM 或 PyMC。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.interpolate import BSpline
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

# 固定随机种子，保证数据划分与 MCMC 采样均可复现
SEED = 2026
np.random.seed(SEED)

METRICS = ("cost", "carbon", "peak", "discomfort", "degradation", "ramping", "resilience")
FEATURES = [f"b{b}_{m}" for b in range(1, 4) for m in METRICS]


# ──────────────────────────────────────────────
# B-样条基构造
# ──────────────────────────────────────────────
def _bspline_basis(x, n_knots=8, degree=3):
    """构造 clamped 三次 B-样条设计矩阵。

    Parameters
    ----------
    x : array (n,)
    n_knots : 内部节点数
    degree : 样条阶数

    Returns
    -------
    B : array (n, K)   K = n_knots + degree + 1
    knots : array, 完整节点序列（含重复边界节点）
    """
    x = np.asarray(x, dtype=float)
    xmin, xmax = float(x.min()), float(x.max())
    if xmax - xmin < 1e-12:
        # 退化为常数特征，返回单列常数
        return np.ones((len(x), 1)), np.array([xmin] * (degree + 1) * 2)
    # 内部节点取分位数
    interior = np.quantile(x, np.linspace(0.0, 1.0, n_knots + 2))[1:-1]
    knots = np.concatenate(
        [np.full(degree + 1, xmin), interior, np.full(degree + 1, xmax)]
    )
    n_basis = len(knots) - degree - 1
    B = np.zeros((len(x), n_basis))
    for i in range(n_basis):
        c = np.zeros(n_basis)
        c[i] = 1.0
        sp = BSpline(knots, c, degree, extrapolate=False)
        B[:, i] = np.nan_to_num(sp(x), nan=0.0)
    return B, knots


def _second_diff_matrix(K):
    """K×K 二阶差分惩罚矩阵 K_j = D₂' D₂（rank K-2），加微小脊使其正定。"""
    if K < 3:
        return np.eye(K)
    D = np.zeros((K - 2, K))
    for i in range(K - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D.T @ D + 1e-6 * np.eye(K)


# ──────────────────────────────────────────────
# Bayesian P-spline GAM
# ──────────────────────────────────────────────
class BayesianGAM(BaseEstimator, RegressorMixin):
    """贝叶斯广义可加模型（B-样条 + Gibbs 采样）。"""

    def __init__(
        self,
        n_knots=8,
        degree=3,
        n_burn=1000,
        n_samples=2000,
        thin=2,
        a_tau=1.0,
        b_tau=0.1,
        a_sigma=1e-3,
        b_sigma=1e-3,
        seed=SEED,
    ):
        self.n_knots = n_knots
        self.degree = degree
        self.n_burn = n_burn
        self.n_samples = n_samples
        self.thin = thin
        self.a_tau = a_tau
        self.b_tau = b_tau
        self.a_sigma = a_sigma
        self.b_sigma = b_sigma
        self.seed = seed

    # -- 构造设计矩阵 --
    def _build_design(self, X, fit=False):
        X = np.asarray(X, dtype=float)
        blocks = []
        if fit:
            self._meta = []  # (knots, col_mean) per feature
        for j in range(X.shape[1]):
            B, knots = _bspline_basis(X[:, j], self.n_knots, self.degree)
            if fit:
                col_mean = B.mean(axis=0)
                self._meta.append((knots, col_mean))
            else:
                knots, col_mean = self._meta[j]
                # 用训练集节点重新评估，保证 train/test 基一致
                n_basis = len(knots) - self.degree - 1
                B = np.zeros((len(X), n_basis))
                for i in range(n_basis):
                    c = np.zeros(n_basis)
                    c[i] = 1.0
                    sp = BSpline(knots, c, self.degree, extrapolate=False)
                    B[:, i] = np.nan_to_num(sp(X[:, j]), nan=0.0)
            B = B - col_mean  # 列中心化，截距吸收均值
            blocks.append(B)
        return np.hstack([np.ones((X.shape[0], 1)), *blocks])

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)

        Z = self._build_design(X, fit=True)
        p = Z.shape[1]
        J = X.shape[1]

        # 每个特征块在 Z 中的列切片（含截距列）
        self._slices = []
        col = 1
        for j in range(J):
            Kj = Z.shape[1] if False else None
            # 从已构造的 blocks 推块宽
            # 直接用 meta 里 knots 算 K
            knots, _ = self._meta[j]
            Kj = len(knots) - self.degree - 1
            self._slices.append(slice(col, col + Kj))
            col += Kj
        assert col == p

        # 预计算固定量
        ZtZ = Z.T @ Z
        Zty = Z.T @ y
        yty = float(y @ y)

        # 各块惩罚矩阵 K_j
        Kpens = []
        for j in range(J):
            s = self._slices[j]
            Kj_width = s.stop - s.start
            Kpens.append(_second_diff_matrix(Kj_width))

        # 初始化
        sigma2 = float(np.var(y))
        tau = np.ones(J)
        # β 初始化为 OLS（带小脊）
        beta = np.linalg.solve(ZtZ + 1e-3 * np.eye(p), Zty)

        n_total = self.n_burn + self.n_samples * self.thin
        keep_every = self.thin
        beta_keep = np.zeros((self.n_samples, p))
        sigma2_keep = np.zeros(self.n_samples)
        tau_keep = np.zeros((self.n_samples, J))
        kept = 0

        # 截距不惩罚
        def make_P(tau_vec):
            P = np.zeros((p, p))
            for j in range(J):
                s = self._slices[j]
                P[s, s] = tau_vec[j] * Kpens[j]
            return P

        for it in range(n_total):
            # 1) 更新 β | rest
            P = make_P(tau)
            prec = ZtZ / sigma2 + P
            L = np.linalg.cholesky(prec)
            m = np.linalg.solve(L.T, np.linalg.solve(L, Zty / sigma2))
            beta = m + np.linalg.solve(L.T, rng.standard_normal(p))

            # 2) 更新 τ_j | β_j
            for j in range(J):
                s = self._slices[j]
                bj = beta[s]
                quad = float(bj @ Kpens[j] @ bj)
                Kj_width = s.stop - s.start
                shape = self.a_tau + 0.5 * Kj_width
                rate = self.b_tau + 0.5 * quad
                tau[j] = rng.gamma(shape, 1.0 / rate)

            # 3) 更新 σ^{-2} | y, β
            resid = y - Z @ beta
            rss = float(resid @ resid)
            shape = self.a_sigma + 0.5 * n
            rate = self.b_sigma + 0.5 * rss
            inv_sigma2 = rng.gamma(shape, 1.0 / rate)
            sigma2 = 1.0 / inv_sigma2

            # 保留样本
            if it >= self.n_burn and (it - self.n_burn) % keep_every == 0:
                beta_keep[kept] = beta
                sigma2_keep[kept] = sigma2
                tau_keep[kept] = tau
                kept += 1

        self.beta_samples_ = beta_keep
        self.sigma2_samples_ = sigma2_keep
        self.tau_samples_ = tau_keep
        self.beta_mean_ = beta_keep.mean(axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        Z = self._build_design(X, fit=False)
        return Z @ self.beta_mean_

    def predict_samples(self, X, include_noise=False, rng=None):
        """返回后验预测样本，形状 (n_samples, n_rows)。"""
        Z = self._build_design(X, fit=False)
        # beta_samples_: (n_samples, p)  ->  mean: (n_rows, n_samples)
        mean = Z @ self.beta_samples_.T
        if include_noise:
            rng = rng or np.random.default_rng(self.seed + 1)
            noise = rng.normal(0.0, 1.0, size=mean.shape) * np.sqrt(self.sigma2_samples_[None, :])
            mean = mean + noise
        # 转置为 (n_samples, n_rows)，与 BART 接口保持一致
        return mean.T


def make_gam(seed=SEED, **overrides):
    params = dict(
        n_knots=8, degree=3, n_burn=1000, n_samples=2000, thin=2,
        a_tau=1.0, b_tau=0.1, a_sigma=1e-3, b_sigma=1e-3, seed=seed,
    )
    params.update(overrides)
    return BayesianGAM(**params)


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
print("  Bayesian GAM (P-spline + Gibbs) —— ramping (R) 预测 (8:2 分层划分)")
print("=" * 70)
print(f"\n划分结果 (random_state={SEED}, 分层标签=ramping 10 分位):")
print(f"  训练集: {len(y_train)} ({len(y_train)/len(y):.1%})")
print(f"  测试集: {len(y_test)} ({len(y_test)/len(y):.1%})")
print(f"  训练集 ramping: mean={y_train.mean():.2f}  std={y_train.std():.2f}  "
      f"min={y_train.min():.2f}  max={y_train.max():.2f}")
print(f"  测试集 ramping: mean={y_test.mean():.2f}  std={y_test.std():.2f}  "
      f"min={y_test.min():.2f}  max={y_test.max():.2f}")

GAM_PARAMS = dict(n_knots=8, degree=3, n_burn=1000, n_samples=2000, thin=2,
                  a_tau=1.0, b_tau=0.1, a_sigma=1e-3, b_sigma=1e-3)
print("\n--- Bayesian GAM 配置 ---")
print(f"  n_knots={GAM_PARAMS['n_knots']}  degree={GAM_PARAMS['degree']}  "
      f"n_burn={GAM_PARAMS['n_burn']}  n_samples={GAM_PARAMS['n_samples']}  "
      f"thin={GAM_PARAMS['thin']}")
print(f"  先验: τ~Gamma({GAM_PARAMS['a_tau']},{GAM_PARAMS['b_tau']}),  "
      f"σ⁻²~Gamma({GAM_PARAMS['a_sigma']},{GAM_PARAMS['b_sigma']})")

# ──────────────────────────────────────────────
# 最终模型：训练集上拟合
# ──────────────────────────────────────────────
final_model = make_gam(SEED).fit(X_train, y_train)
pred_test = final_model.predict(X_test)
# 完整后验预测分布（含观测噪声 σ²），用于 95% 预测区间与覆盖率
pred_test_samples = final_model.predict_samples(X_test, include_noise=True)  # (n_samples, n_test)
pred_std = pred_test_samples.std(axis=0)

# 训练集 5 折 OOF（每折单独固定种子）
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
pred_train_oof = np.zeros_like(y_train)
for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train), start=1):
    fold_model = make_gam(SEED + fold).fit(X_train[tr_idx], y_train[tr_idx])
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
print(f"  后验预测标准差均值 = {pred_std.mean():.3f} (真实值 std={y_test.std():.3f})")
print(f"  95% 区间覆盖率 = {np.mean(np.abs(y_test - pred_test) <= 1.96 * pred_std):.1%}")

# 光滑项有效自由度（近似）：tr[(Z'Z + σ² P)^{-1} Z'Z]
print("\n--- 后验超参均值 ---")
print(f"  噪声 σ² = {final_model.sigma2_samples_.mean():.3f}")
print(f"  各特征光滑精度 τ 均值范围: "
      f"[{final_model.tau_samples_.mean(0).min():.2f}, "
      f"{final_model.tau_samples_.mean(0).max():.2f}]")

# ──────────────────────────────────────────────
# 拟合曲线可视化
# ──────────────────────────────────────────────
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# 图1: 测试集拟合曲线（按真实值排序 + 后验 95% 区间 + 滑动平均趋势）
ax = axes[0]
order = np.argsort(y_test)
xs, ys = y_test[order], pred_test[order]
lo = (pred_test - 1.96 * pred_std)[order]
hi = (pred_test + 1.96 * pred_std)[order]
ax.plot(xs, xs, "k--", lw=1.5, label="理想拟合 y=x")
ax.fill_between(xs, lo, hi, color="steelblue", alpha=0.15, label="95% 后验区间")
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
out_png = "gam_fit_curves_strat82.png"
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

model_label = (
    f"BayesianGAM(P-spline, knots={GAM_PARAMS['n_knots']}, "
    f"burn={GAM_PARAMS['n_burn']}, samples={GAM_PARAMS['n_samples']})"
)

print("\n" + "=" * 70)
print("  Lab6 SurrogateEvaluator —— 测试集（8:2 holdout）评估")
print("=" * 70)
ev = SurrogateEvaluator(X_test, y_test, feature_names=FEATURES)
report = ev.evaluate(
    final_model,
    model_name=model_label,
    oof_predictions=pred_test,
    run_subset=True,
    model_factory=lambda: make_gam(SEED + 7),
    subset_folds=3,
    subset_topk=10,
)
print(report.summary())

print("\n--- 多模型对比表 ---")
print(ev.compare_models([report]).to_string(index=False))
print(f"\n复现参数：SEED={SEED}, test_size=0.20, strata=qcut(y, q=10), GAM={GAM_PARAMS}")
