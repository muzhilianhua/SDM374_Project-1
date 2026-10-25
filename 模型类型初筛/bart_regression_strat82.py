"""BART 预测 ramping (R)：8:2 分层划分 + 拟合曲线 + Lab6 评估器

BART（Bayesian Additive Regression Trees）用 50 棵浅树做贝叶斯加性回归，
通过 MCMC 对树结构、叶节点参数和噪声方差联合采样，预测取后验样本均值。

环境：bartpy2 0.0.2（pip install bartpy2）
"""
import contextlib
import io
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bartpy2.sklearnmodel import SklearnModel
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

# 固定随机种子，保证数据划分与 MCMC 采样均可复现
SEED = 2026
np.random.seed(SEED)
random.seed(SEED)

METRICS = ("cost", "carbon", "peak", "discomfort", "degradation", "ramping", "resilience")
FEATURES = [f"b{b}_{m}" for b in range(1, 4) for m in METRICS]

# ---------- BART 配置 ----------
# 50 棵树、alpha=0.95/beta=2（Chipman 默认先验，偏向浅树）、1 条链（保证可复现）
BART_PARAMS = dict(
    n_trees=50,
    alpha=0.95,
    beta=2.0,
    n_burn=500,
    n_samples=1000,
    thin=0.05,          # 每 20 次迭代保留 1 个后验样本 → 50 个森林
    n_chains=1,
    n_jobs=1,           # 单进程，np.random 全局种子即可复现
    store_in_sample_predictions=False,
)


class SeededBART:
    """给 bartpy2 套一层固定种子的 sklearn 风格接口（fit / predict）。"""

    def __init__(self, seed=SEED, **params):
        self.seed = seed
        self.params = params
        self.model = None

    def fit(self, X, y):
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.model = SklearnModel(**self.params)
        # bartpy2 会打印 tqdm 进度条，这里静默以减少控制台噪声
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
        return self

    def predict(self, X):
        return self.model.predict(np.asarray(X, dtype=float))

    def predict_samples(self, X):
        """返回每个后验森林对 X 的预测，形状 (n_samples, n_rows)，已还原到原始量纲。"""
        X = np.asarray(X, dtype=float)
        data = self.model.data
        return np.array([data.unnormalize_y(m.predict(X)) for m in self.model.model_samples])


def make_bart(seed=SEED):
    return SeededBART(seed=seed, **BART_PARAMS)


# ---------- 数据加载 ----------
df = pd.read_csv("dataset_same8/train.csv", float_precision="round_trip")
X = df[FEATURES].to_numpy(dtype=float)
y = df["ramping"].to_numpy(dtype=float)

# ---------- 8 : 2 分层划分 ----------
# ramping 是连续值，先按 10 分位切层，再分层抽样，保证两集目标分布一致
strata = pd.qcut(y, q=10, labels=False, duplicates="drop")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=strata, random_state=SEED, shuffle=True
)

print("=" * 68)
print("  BART 回归 —— ramping (R) 预测 (8:2 分层划分)")
print("=" * 68)
print(f"\n划分结果 (random_state={SEED}, 分层标签=ramping 10 分位):")
print(f"  训练集: {len(y_train)} ({len(y_train)/len(y):.1%})")
print(f"  测试集: {len(y_test)} ({len(y_test)/len(y):.1%})")
print(f"  训练集 ramping: mean={y_train.mean():.2f}  std={y_train.std():.2f}  "
      f"min={y_train.min():.2f}  max={y_train.max():.2f}")
print(f"  测试集 ramping: mean={y_test.mean():.2f}  std={y_test.std():.2f}  "
      f"min={y_test.min():.2f}  max={y_test.max():.2f}")

print("\n--- BART 配置 ---")
print(f"  n_trees={BART_PARAMS['n_trees']}  alpha={BART_PARAMS['alpha']}  "
      f"beta={BART_PARAMS['beta']}  n_burn={BART_PARAMS['n_burn']}  "
      f"n_samples={BART_PARAMS['n_samples']}  thin={BART_PARAMS['thin']}")

# ---------- 最终模型：训练集上拟合 ----------
final_model = make_bart(SEED).fit(X_train, y_train)
pred_test = final_model.predict(X_test)
pred_test_samples = final_model.predict_samples(X_test)
pred_std = pred_test_samples.std(axis=0)

# ---------- 训练集 5 折 OOF（每折单独固定种子） ----------
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
pred_train_oof = np.zeros_like(y_train)
for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train), start=1):
    fold_model = make_bart(SEED + fold).fit(X_train[tr_idx], y_train[tr_idx])
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

# ---------- 拟合曲线可视化 ----------
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
out_png = "bart_fit_curves_strat82.png"
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
    final_model,
    model_name=f"BART(8:2分层, {BART_PARAMS['n_trees']}树, "
               f"alpha={BART_PARAMS['alpha']}, beta={BART_PARAMS['beta']})",
    oof_predictions=pred_test,   # 模型未见过测试集，预测即为样本外结果
    run_subset=True,
    model_factory=lambda: make_bart(SEED + 7),
    subset_folds=3,
    subset_topk=10,
)
print(report.summary())

print("\n--- 多模型对比表 ---")
print(ev.compare_models([report]).to_string(index=False))
print(f"\n复现参数：SEED={SEED}, test_size=0.20, strata=qcut(y, q=10), BART={BART_PARAMS}")
