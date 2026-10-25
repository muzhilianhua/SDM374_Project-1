"""XGBoost 预测 ramping (R)：最优 5% 样本加权（8 倍）+ 8:2 分层划分 + Lab6 评估器

- 权重策略：训练集中 y <= 5% 分位数的"最优 5%"样本权重 = 8（其余为 1），
  差样本不删除、全部参与训练；阈值只用训练集计算，测试集不参与，避免泄漏。
- 划分：8:2 分层（ramping 10 分位标签），random_state=2026 可复现。
- 公平性：加权 / 不加权（1 倍）两条流水线使用完全相同的超参、划分与种子，
  唯一差别是样本权重；另附 1/5/8/10 倍权重的训练集 5 折 OOF 敏感度对照。
- 评估：Lab6/surrogate_evaluator.py（最优 5% 局部精度 + Spearman + 子集模拟寻优）。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from xgboost import XGBRegressor

# 固定随机种子，保证划分、交叉验证与模型训练均可复现
SEED = 2026
np.random.seed(SEED)

TOP_Q = 5.0          # “最优 5%”分位
TOP5_WEIGHT = 8.0    # 最优 5% 样本的权重倍数（用户建议区间 5~10，取中值 8）

METRICS = ("cost", "carbon", "peak", "discomfort", "degradation", "ramping", "resilience")
FEATURES = [f"b{b}_{m}" for b in range(1, 4) for m in METRICS]

# 沿用项目此前验证集调出的 XGBoost 最优配置（xgb_surrogate_eval.py），
# 加权与不加权共用同一套超参，保证对照公平
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


class WeightedXGB(BaseEstimator, RegressorMixin):
    """在 fit 内部按"y 的最优 5%"自动加权的 XGBoost 包装器。

    Lab6 评估器的 subset_optimization 只调用 model.fit(X, y)（不传 sample_weight），
    因此把加权逻辑封装进 fit：每个训练折各自用自己的 5% 分位计算阈值，
    主实验与子集模拟的加权方式一致，且不存在阈值泄漏。
    """

    def __init__(self, weight=TOP5_WEIGHT, top_q=TOP_Q, xgb_params=None):
        self.weight = weight
        self.top_q = top_q
        self.xgb_params = xgb_params

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        thr = float(np.percentile(y, self.top_q))
        w = np.where(y <= thr, self.weight, 1.0)
        self.threshold_ = thr
        self.n_good_ = int((y <= thr).sum())
        self.model_ = XGBRegressor(**(self.xgb_params or {}))
        self.model_.fit(X, y, sample_weight=w)
        return self

    def predict(self, X):
        return self.model_.predict(X)


# ---------- 数据加载 ----------
df = pd.read_csv("dataset_same8/train.csv", float_precision="round_trip")
X = df[FEATURES].to_numpy(dtype=float)
y = df["ramping"].to_numpy(dtype=float)

# ---------- 8 : 2 分层划分 ----------
strata = pd.qcut(y, q=10, labels=False, duplicates="drop")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=strata, random_state=SEED, shuffle=True
)

thr_train = float(np.percentile(y_train, TOP_Q))   # 加权阈值（仅训练集）
n_good_train = int((y_train <= thr_train).sum())

print("=" * 70)
print(f"  XGBoost 回归 —— ramping (R) 预测 | 最优 5% 加权 x{TOP5_WEIGHT:.0f} | 8:2 分层划分")
print("=" * 70)
print(f"\n划分结果 (random_state={SEED}, 分层标签=ramping 10 分位):")
print(f"  训练集: {len(y_train)} ({len(y_train)/len(y):.1%})   测试集: {len(y_test)} ({len(y_test)/len(y):.1%})")
print(f"  训练集 ramping: mean={y_train.mean():.2f}  std={y_train.std():.2f}  "
      f"min={y_train.min():.2f}  max={y_train.max():.2f}")
print(f"  测试集 ramping: mean={y_test.mean():.2f}  std={y_test.std():.2f}  "
      f"min={y_test.min():.2f}  max={y_test.max():.2f}")
print(f"\n加权策略: y <= {thr_train:.2f} (训练集 {TOP_Q:.0f}% 分位) 的 {n_good_train} 条样本 "
      f"权重={TOP5_WEIGHT:.0f}，其余权重=1，差样本不删除")

# ---------- 权重倍数敏感度：训练集 5 折 OOF 对照 ----------
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

print("\n--- 权重倍数敏感度（训练集 5 折 OOF，超参固定）---")
print(f"  {'权重':>4s}  {'OOF MAE(全)':>12s}  {'OOF MAE(最优5%)':>15s}  {'OOF R²':>8s}")
sweep_rows = {}
for w_mult in (1.0, 5.0, 8.0, 10.0):
    oof_w = cross_val_predict(
        WeightedXGB(weight=w_mult, xgb_params=BEST), X_train, y_train, cv=kf
    )
    mae_all = mean_absolute_error(y_train, oof_w)
    good = y_train <= thr_train
    mae_good = mean_absolute_error(y_train[good], oof_w[good])
    r2_all = r2_score(y_train, oof_w)
    sweep_rows[w_mult] = (oof_w, mae_all, mae_good, r2_all)
    tag = "  <-- 主实验" if w_mult == TOP5_WEIGHT else ""
    print(f"  {w_mult:>4.0f}x  {mae_all:>12.4f}  {mae_good:>15.4f}  {r2_all:>8.4f}{tag}")

# ---------- 最终模型：加权 vs 不加权（同超参/同划分/同种子，公平对照）----------
model_w = WeightedXGB(weight=TOP5_WEIGHT, xgb_params=BEST).fit(X_train, y_train)
model_b = WeightedXGB(weight=1.0, xgb_params=BEST).fit(X_train, y_train)

pred_test_w = model_w.predict(X_test)
pred_test_b = model_b.predict(X_test)
oof_train_w = sweep_rows[TOP5_WEIGHT][0]
oof_train_b = sweep_rows[1.0][0]


def metric_row(y_true, y_pred, thr):
    good = y_true <= thr
    return dict(
        mae=mean_absolute_error(y_true, y_pred),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2=r2_score(y_true, y_pred),
        rho=spearmanr(y_true, y_pred).statistic,
        mae_good=mean_absolute_error(y_true[good], y_pred[good]),
        bias_good=float(np.mean(y_pred[good] - y_true[good])),
    )


print("\n--- 测试集表现（阈值用训练集 5% 分位，公平一致）---")
print(f"  {'模型':<22s}  {'MAE':>7s}  {'RMSE':>7s}  {'R²':>7s}  {'ρ':>7s}  "
      f"{'5%MAE':>7s}  {'5%偏差':>7s}")
for name, pred in (("加权 x8 (主实验)", pred_test_w), ("不加权 x1 (对照)", pred_test_b)):
    m = metric_row(y_test, pred, thr_train)
    print(f"  {name:<22s}  {m['mae']:>7.3f}  {m['rmse']:>7.3f}  {m['r2']:>7.4f}  "
          f"{m['rho']:>7.4f}  {m['mae_good']:>7.3f}  {m['bias_good']:>+7.3f}")

# 特征重要性（加权模型）
imp = model_w.model_.feature_importances_
print("\n--- 特征重要性 Top-8（加权模型）---")
for rank, idx in enumerate(np.argsort(imp)[::-1][:8]):
    print(f"  #{rank + 1}: {FEATURES[idx]:18s} = {imp[idx]:.4f}")

# ---------- 拟合曲线可视化 ----------
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

m_w = metric_row(y_test, pred_test_w, thr_train)
good_test = y_test <= thr_train

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# 图1: 测试集拟合曲线（按真实值排序 + 滑动平均趋势，高亮最优 5% 区）
ax = axes[0]
order = np.argsort(y_test)
xs, ys = y_test[order], pred_test_w[order]
ax.axvspan(xs.min(), thr_train, color="gold", alpha=0.15, label=f"最优 5% 区 (y≤{thr_train:.1f})")
ax.plot(xs, xs, "k--", lw=1.5, label="理想拟合 y=x")
ax.scatter(xs[~good_test[order]], ys[~good_test[order]], c="steelblue", s=30,
           edgecolors="white", linewidths=0.4, zorder=3, label="测试集预测")
ax.scatter(xs[good_test[order]], ys[good_test[order]], c="crimson", s=45,
           edgecolors="white", linewidths=0.5, zorder=4, label="最优 5% 样本")
trend = pd.Series(ys).rolling(window=41, center=True, min_periods=5).mean().to_numpy()
ax.plot(xs, trend, c="darkorange", lw=2.2, label="拟合趋势（滑动平均）")
ax.set_xlabel("真实 ramping")
ax.set_ylabel("预测 ramping")
ax.set_title(f"测试集拟合曲线（加权 x{TOP5_WEIGHT:.0f}）\n"
             f"R²={m_w['r2']:.4f}  MAE={m_w['mae']:.2f}  最优5%MAE={m_w['mae_good']:.2f}")
ax.legend(loc="upper left", fontsize=8.5)
ax.grid(alpha=0.3)

# 图2: 训练集 OOF vs 测试集 预测散点（加权模型）
ax = axes[1]
ax.scatter(y_train, oof_train_w, c="gray", s=10, alpha=0.4,
           label=f"训练集 OOF (R²={r2_score(y_train, oof_train_w):.4f})")
ax.scatter(y_test[~good_test], pred_test_w[~good_test], c="steelblue", s=30,
           edgecolors="white", linewidths=0.4, zorder=3, label=f"测试集 (R²={m_w['r2']:.4f})")
ax.scatter(y_test[good_test], pred_test_w[good_test], c="crimson", s=45,
           edgecolors="white", linewidths=0.5, zorder=4, label="测试集最优 5%")
lims = [y.min() - 5, y.max() + 5]
ax.plot(lims, lims, "k--", lw=1.5, label="y=x")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("真实 ramping")
ax.set_ylabel("预测 ramping")
ax.set_title("训练集 OOF vs 测试集 预测对比")
ax.legend(loc="upper left", fontsize=8.5)
ax.grid(alpha=0.3)

# 图3: 测试集残差（加权 vs 不加权在最优 5% 区的对比）
ax = axes[2]
ax.scatter(pred_test_w[~good_test], y_test[~good_test] - pred_test_w[~good_test],
           c="steelblue", s=30, edgecolors="white", linewidths=0.4, label="普通样本 (加权)")
ax.scatter(pred_test_w[good_test], y_test[good_test] - pred_test_w[good_test],
           c="crimson", s=45, edgecolors="white", linewidths=0.5, label="最优 5% (加权)")
ax.scatter(pred_test_b[good_test], y_test[good_test] - pred_test_b[good_test],
           facecolors="none", edgecolors="darkgreen", s=55, linewidths=1.2,
           label="最优 5% (不加权对照)")
ax.axhline(0, color="k", ls="--", lw=1.5)
res_std = (y_test - pred_test_w).std()
ax.set_xlabel("预测 ramping")
ax.set_ylabel("残差 (真实 - 预测)")
ax.set_title(f"测试集残差分布 (加权 std={res_std:.2f})")
ax.legend(loc="upper right", fontsize=8.5)
ax.grid(alpha=0.3)

plt.tight_layout()
out_png = "xgb_fit_curves_top5w.png"
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"\n✓ 拟合曲线已保存: {out_png}")

# ---------- Lab6 评估器 ----------
LAB6 = Path(r"D:\HuaweiMoveData\Users\11659\Desktop\机器学习项目\Lab6")
if not (LAB6 / "surrogate_evaluator.py").exists():
    raise FileNotFoundError(f"找不到评估器: {LAB6 / 'surrogate_evaluator.py'}")
sys.path.insert(0, str(LAB6))

from surrogate_evaluator import SurrogateEvaluator  # noqa: E402

print("\n" + "=" * 70)
print("  Lab6 SurrogateEvaluator —— 测试集（8:2 holdout）评估")
print("=" * 70)

ev = SurrogateEvaluator(X_test, y_test, feature_names=FEATURES)

# 加权模型：预测为样本外结果；子集模拟用同款加权工厂，加权逻辑在 fit 内自闭合
report_w = ev.evaluate(
    model_w,
    model_name=f"XGBoost 最优5%加权x{TOP5_WEIGHT:.0f} (depth=5, lr=0.05, n=600)",
    oof_predictions=pred_test_w,
    run_subset=True,
    model_factory=lambda: WeightedXGB(weight=TOP5_WEIGHT, xgb_params=BEST),
    subset_folds=3,
    subset_topk=10,
)
print(report_w.summary())

# 不加权对照：同一评估器、同一 holdout、同一超参
report_b = ev.evaluate(
    model_b,
    model_name="XGBoost 不加权 x1 (对照)",
    oof_predictions=pred_test_b,
    run_subset=True,
    model_factory=lambda: WeightedXGB(weight=1.0, xgb_params=BEST),
    subset_folds=3,
    subset_topk=10,
)
print("\n" + report_b.summary())

print("\n--- 加权 vs 不加权 对比表 ---")
print(ev.compare_models([report_w, report_b]).to_string(index=False))

print(f"\n复现参数：SEED={SEED}, test_size=0.20, strata=qcut(y, q=10), "
      f"TOP_Q={TOP_Q}%, TOP5_WEIGHT={TOP5_WEIGHT}, XGB 超参={BEST}")
