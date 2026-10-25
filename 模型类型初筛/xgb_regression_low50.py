"""XGBoost 预测 ramping（仅 ramping 值较小的前 50% 样本）：

- 数据：dataset_same8/train.csv 中 ramping 升序排序后的前 1024 条
- 划分：训练/验证/测试 = 7 : 1.5 : 1.5（random_state=2026，可复现）
- 流程：验证集选超参 → 训练+验证重训 → 测试集评估 → 拟合曲线
- 评估：Lab6/surrogate_evaluator.py（5 折 OOF + 子集模拟寻优）
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict, train_test_split
from xgboost import XGBRegressor

LAB6 = Path(r"D:\HuaweiMoveData\Users\11659\Desktop\机器学习项目\Lab6")
sys.path.insert(0, str(LAB6))
from surrogate_evaluator import SurrogateEvaluator  # noqa: E402

# 固定随机种子，保证可复现
SEED = 2026
np.random.seed(SEED)

METRICS = ("cost", "carbon", "peak", "discomfort", "degradation", "ramping", "resilience")
FEATURES = [f"b{b}_{m}" for b in range(1, 4) for m in METRICS]

# ---------- 数据加载：只保留 ramping 较小的前 50% ----------
df = pd.read_csv("dataset_same8/train.csv", float_precision="round_trip")
n_half = len(df) // 2
df = df.nsmallest(n_half, "ramping").reset_index(drop=True)

X = df[FEATURES].values.astype(float)
y = df["ramping"].values.astype(float)

print("=" * 60)
print("  XGBoost 回归 —— 低 50% ramping 样本 (7:1.5:1.5 划分)")
print("=" * 60)
print(f"\n数据筛选: ramping 升序取前 {n_half} 条 "
      f"(y 范围 [{y.min():.2f}, {y.max():.2f}], 中位数={np.median(y):.2f})")

# ---------- 7 : 1.5 : 1.5 划分 ----------
# 先切出 70% 训练集，剩余 30% 再对半分为验证/测试（各 15%）
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED, shuffle=True
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, shuffle=True
)

print(f"\n划分结果 (random_state={SEED}):")
print(f"  训练集: {len(y_train)} ({len(y_train)/len(y):.1%})")
print(f"  验证集: {len(y_val)} ({len(y_val)/len(y):.1%})")
print(f"  测试集: {len(y_test)} ({len(y_test)/len(y):.1%})")


def make_xgb(n_est, depth, lr):
    return XGBRegressor(
        n_estimators=n_est,
        max_depth=depth,
        learning_rate=lr,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
    )


# ---------- 用验证集做简单超参选择 ----------
print("\n--- 验证集超参选择 (max_depth × learning_rate × n_estimators) ---")
configs = [
    (d, lr, n)
    for d in (3, 5, 7)
    for lr in (0.05, 0.1)
    for n in (300, 600)
]
best = None
for depth, lr, n_est in configs:
    xgb = make_xgb(n_est, depth, lr)
    xgb.fit(X_train, y_train)
    val_mae = mean_absolute_error(y_val, xgb.predict(X_val))
    print(f"  depth={depth}  lr={lr:<5}  n_est={n_est}  val_MAE={val_mae:.4f}")
    if best is None or val_mae < best[0]:
        best = (val_mae, depth, lr, n_est)

best_mae, best_d, best_lr, best_n = best
print(f"\n  ★ 最优配置: max_depth={best_d}, learning_rate={best_lr}, "
      f"n_estimators={best_n}, val_MAE={best_mae:.4f}")


def make_final_xgb():
    return make_xgb(best_n, best_d, best_lr)


# ---------- 用 训练集+验证集 重训最终模型，在测试集评估 ----------
X_trval = np.vstack([X_train, X_val])
y_trval = np.concatenate([y_train, y_val])

final_xgb = make_final_xgb()
final_xgb.fit(X_trval, y_trval)

pred_test = final_xgb.predict(X_test)
pred_trval = final_xgb.predict(X_trval)

test_mae = mean_absolute_error(y_test, pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, pred_test))
test_r2 = r2_score(y_test, pred_test)
train_mae = mean_absolute_error(y_trval, pred_trval)
train_r2 = r2_score(y_trval, pred_trval)

print("\n--- 最终模型评估 ---")
print(f"  训练+验证集: MAE={train_mae:.4f}  R²={train_r2:.4f}")
print(f"  测试集:      MAE={test_mae:.4f}  RMSE={test_rmse:.4f}  R²={test_r2:.4f}")

# 特征重要性
imp = final_xgb.feature_importances_
print("\n--- 特征重要性 Top-8 ---")
for rank, idx in enumerate(np.argsort(imp)[::-1][:8]):
    print(f"  #{rank+1}: {FEATURES[idx]:18s} = {imp[idx]:.4f}")

# ---------- 可视化 ----------
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# 图1: 测试集拟合曲线（真实值 vs 预测值，按真实值排序连线）
ax = axes[0]
order = np.argsort(y_test)
ax.plot(y_test[order], y_test[order], "k--", lw=1.5, label="理想拟合 y=x")
ax.scatter(y_test[order], pred_test[order], c="steelblue", s=28,
           edgecolors="white", linewidths=0.4, zorder=3, label="测试集预测")
ax.plot(y_test[order], pred_test[order], c="tomato", lw=1.2, alpha=0.7, label="预测趋势")
ax.set_xlabel("真实 ramping")
ax.set_ylabel("预测 ramping")
ax.set_title(f"测试集拟合曲线  (R²={test_r2:.4f}, MAE={test_mae:.2f})")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)

# 图2: 训练集 vs 测试集 预测散点对比
ax = axes[1]
ax.scatter(y_trval, pred_trval, c="gray", s=10, alpha=0.4, label=f"训练+验证 (R²={train_r2:.3f})")
ax.scatter(y_test, pred_test, c="steelblue", s=28, edgecolors="white",
           linewidths=0.4, zorder=3, label=f"测试集 (R²={test_r2:.3f})")
lims = [y.min() - 5, y.max() + 5]
ax.plot(lims, lims, "k--", lw=1.5, label="y=x")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("真实 ramping")
ax.set_ylabel("预测 ramping")
ax.set_title("训练集 vs 测试集 预测对比")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)

# 图3: 残差图
ax = axes[2]
residuals = y_test - pred_test
ax.scatter(pred_test, residuals, c="steelblue", s=28, edgecolors="white", linewidths=0.4)
ax.axhline(0, color="k", ls="--", lw=1.5)
ax.set_xlabel("预测 ramping")
ax.set_ylabel("残差 (真实 - 预测)")
ax.set_title(f"测试集残差分布  (std={residuals.std():.2f})")
ax.grid(alpha=0.3)

plt.tight_layout()
out_png = "xgb_fit_curves_low50.png"
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"\n✓ 拟合曲线已保存: {out_png}")

# ---------- Lab6 SurrogateEvaluator 评估 ----------
print("\n" + "=" * 60)
print("  Lab6 SurrogateEvaluator 评估（低 50% ramping 子集）")
print("=" * 60)

# 5 折 OOF 预测，避免用训练集自预测造成指标虚高
oof = cross_val_predict(make_final_xgb(), X, y, cv=5, method="predict")

model = make_final_xgb().fit(X, y)

ev = SurrogateEvaluator(X, y, feature_names=FEATURES)
report = ev.evaluate(
    model,
    model_name=f"XGBoost(depth={best_d}, lr={best_lr}, n={best_n}) [低50%数据]",
    oof_predictions=oof,
    run_subset=True,
    model_factory=make_final_xgb,
    subset_folds=3,
    subset_topk=10,
)
print(report.summary())

print("\n--- OOF 全样本指标（低 50% 子集） ---")
print(f"  MAE={mean_absolute_error(y, oof):.3f}  "
      f"RMSE={np.sqrt(mean_squared_error(y, oof)):.3f}  R²={r2_score(y, oof):.4f}")

print("\n--- 多模型对比表 ---")
print(ev.compare_models([report]).to_string(index=False))
