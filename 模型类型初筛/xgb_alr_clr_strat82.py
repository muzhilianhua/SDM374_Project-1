"""XGBoost 预测 ramping (R)：成分数据 ALR / CLR 变换对比 (8:2 分层划分 + Lab6 评估器)

背景：21 个输入是 3 栋楼 x 7 项的成分数据（每楼权重非负、和为 1）。
直接用原始权重做欧氏特征不适配单形几何，本脚本对比三种特征表示：
  1. RAW: 原始 21 维权重（基线，与 xgb_regression_strat82.py 同协议）
  2. ALR: 每栋楼 7 维 -> 6 维，log(x_i / x_resilience)，共 18 维（加法对数比）
  3. CLR: 每栋楼 7 维 -> 7 维，log(x_i / g(x))，g 为楼内几何均值，共 21 维（中心对数比）

数据已验证：无零值（最小 1.45e-17，对数安全）、每楼严格和为 1，无需零值替换。

三种表示共用：同一 8:2 分层划分索引（SEED=2026, qcut(y,10)）、同一超参网格、
同一 Lab6 SurrogateEvaluator 协议（测试集评估 + 全量 5 折 OOF 评估 + 3 折子集寻优）。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_predict,
    train_test_split,
)
from scipy.stats import spearmanr
from xgboost import XGBRegressor

# 固定随机种子，保证划分、交叉验证与模型训练均可复现
SEED = 2026
np.random.seed(SEED)

METRICS = ("cost", "carbon", "peak", "discomfort", "degradation", "ramping", "resilience")
FEATURES = [f"b{b}_{m}" for b in range(1, 4) for m in METRICS]
REF_METRIC = "resilience"  # ALR 参照成分：每楼第 7 项

# ---------- 数据加载 ----------
df = pd.read_csv("dataset_same8/train.csv", float_precision="round_trip")
X_raw = df[FEATURES].to_numpy(dtype=float)
y = df["ramping"].to_numpy(dtype=float)


# ---------- 成分数据变换 ----------
def alr_transform(X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """逐楼做 ALR：log(x_i / x_ref)，参照为 resilience，每楼 7->6 维。"""
    cols, names = [], []
    for b in range(3):
        block = X[:, b * 7:(b + 1) * 7]
        ref = block[:, 6:7]  # resilience 列
        for j, m in enumerate(METRICS):
            if m == REF_METRIC:
                continue
            cols.append(np.log(block[:, j:j + 1] / ref))
            names.append(f"b{b + 1}_alr_{m}")
    return np.hstack(cols), names


def clr_transform(X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """逐楼做 CLR：log(x_i / g(x))，g 为楼内几何均值，维度不变（21 维，楼内和为 0）。"""
    cols, names = [], []
    for b in range(3):
        block = X[:, b * 7:(b + 1) * 7]
        log_b = np.log(block)
        clr = log_b - log_b.mean(axis=1, keepdims=True)
        for j, m in enumerate(METRICS):
            cols.append(clr[:, j:j + 1])
            names.append(f"b{b + 1}_clr_{m}")
    return np.hstack(cols), names


X_alr, ALR_FEATURES = alr_transform(X_raw)
X_clr, CLR_FEATURES = clr_transform(X_raw)

REPRESENTATIONS = {
    "RAW_原始权重": (X_raw, FEATURES),
    "ALR_对数比18维": (X_alr, ALR_FEATURES),
    "CLR_中心对数比21维": (X_clr, CLR_FEATURES),
}

# ---------- 8 : 2 分层划分（三种表示共用同一索引） ----------
strata = pd.qcut(y, q=10, labels=False, duplicates="drop")
idx_train, idx_test = train_test_split(
    np.arange(len(y)), test_size=0.20, stratify=strata, random_state=SEED, shuffle=True
)
y_train, y_test = y[idx_train], y[idx_test]

print("=" * 72)
print("  XGBoost —— 成分数据 ALR / CLR 特征变换对比 (8:2 分层划分)")
print("=" * 72)
print(f"\n划分结果 (random_state={SEED}, 分层标签=ramping 10 分位):")
print(f"  训练集: {len(y_train)} ({len(y_train)/len(y):.1%})   测试集: {len(y_test)} ({len(y_test)/len(y):.1%})")
print(f"  训练集 ramping: mean={y_train.mean():.2f}  std={y_train.std():.2f}  "
      f"min={y_train.min():.2f}  max={y_train.max():.2f}")
print(f"  测试集 ramping: mean={y_test.mean():.2f}  std={y_test.std():.2f}  "
      f"min={y_test.min():.2f}  max={y_test.max():.2f}")

cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
param_grid = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [300, 600],
}

# ---------- Lab6 评估器 ----------
LAB6 = Path(r"D:\HuaweiMoveData\Users\11659\Desktop\机器学习项目\Lab6")
if not (LAB6 / "surrogate_evaluator.py").exists():
    raise FileNotFoundError(f"找不到评估器: {LAB6 / 'surrogate_evaluator.py'}")
sys.path.insert(0, str(LAB6))
from surrogate_evaluator import SurrogateEvaluator  # noqa: E402


def run_one(rep_name: str, X: np.ndarray, feat_names: list[str]) -> dict:
    """对一种特征表示执行：超参搜索 -> 测试集指标 -> Lab6 双层面评估。"""
    X_train, X_test = X[idx_train], X[idx_test]

    print("\n" + "#" * 72)
    print(f"#  特征表示: {rep_name}  (维度 {X.shape[1]})")
    print("#" * 72)

    print("\n--- 训练集 5 折交叉验证超参搜索 (MAE 最小) ---")
    gs = GridSearchCV(
        XGBRegressor(
            subsample=0.9, colsample_bytree=0.9, tree_method="hist",
            random_state=SEED, n_jobs=-1,
        ),
        param_grid, scoring="neg_mean_absolute_error", cv=cv, n_jobs=1, refit=True,
    )
    gs.fit(X_train, y_train)
    best_params = gs.best_params_
    print(f"  ★ 最优配置: {best_params}")
    print(f"  CV MAE = {-gs.best_score_:.4f}")

    def make_xgb():
        return XGBRegressor(
            **best_params, subsample=0.9, colsample_bytree=0.9,
            tree_method="hist", random_state=SEED, n_jobs=-1,
        )

    # 最终模型 + 测试集指标
    model = make_xgb().fit(X_train, y_train)
    pred_test = model.predict(X_test)
    pred_train_oof = cross_val_predict(make_xgb(), X_train, y_train, cv=cv, n_jobs=-1)

    train_mae = mean_absolute_error(y_train, pred_train_oof)
    train_r2 = r2_score(y_train, pred_train_oof)
    test_mae = mean_absolute_error(y_test, pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, pred_test))
    test_r2 = r2_score(y_test, pred_test)
    test_rho = spearmanr(y_test, pred_test).statistic

    print("\n--- 最终模型评估 ---")
    print(f"  训练集 (5 折 OOF): MAE={train_mae:.4f}  R²={train_r2:.4f}")
    print(f"  测试集:            MAE={test_mae:.4f}  RMSE={test_rmse:.4f}  R²={test_r2:.4f}")
    print(f"  测试集 Spearman ρ = {test_rho:.4f}")

    imp = model.feature_importances_
    print("\n--- 特征重要性 Top-5 ---")
    for rank, idx in enumerate(np.argsort(imp)[::-1][:5]):
        print(f"  #{rank + 1}: {feat_names[idx]:22s} = {imp[idx]:.4f}")

    # Lab6 层面一：8:2 holdout 测试集评估
    print("\n--- Lab6 评估 (一)：测试集 holdout ---")
    ev_test = SurrogateEvaluator(X_test, y_test, feature_names=feat_names)
    rep_test = ev_test.evaluate(
        model, model_name=f"XGB+{rep_name}(holdout)",
        oof_predictions=pred_test,
        run_subset=True, model_factory=make_xgb, subset_folds=3, subset_topk=10,
    )
    print(rep_test.summary())

    # Lab6 层面二：全量 5 折 OOF 评估（更稳定，与历史结论对齐）
    print("\n--- Lab6 评估 (二)：全量 5 折 OOF ---")
    oof_full = cross_val_predict(make_xgb(), X, y, cv=cv, n_jobs=-1)
    ev_full = SurrogateEvaluator(X, y, feature_names=feat_names)
    rep_full = ev_full.evaluate(
        model, model_name=f"XGB+{rep_name}(OOF)",
        oof_predictions=oof_full,
        run_subset=True, model_factory=make_xgb, subset_folds=3, subset_topk=10,
    )
    print(rep_full.summary())

    return {
        "rep": rep_name,
        "best_params": best_params,
        "test_mae": test_mae, "test_r2": test_r2, "test_rho": test_rho,
        "holdout_local_mae": rep_test.local.mae,
        "holdout_bias": rep_test.local.bias,
        "holdout_rho": rep_test.ranking.spearman_r,
        "holdout_subset": rep_test.subset_pass_rate,
        "oof_mae_overall": rep_full.local.mae_overall,
        "oof_local_mae": rep_full.local.mae,
        "oof_bias": rep_full.local.bias,
        "oof_rho": rep_full.ranking.spearman_r,
        "oof_subset": rep_full.subset_pass_rate,
        "oof_verdict": rep_full.verdict,
        "y_test_pred": pred_test,
        "oof_full": oof_full,
    }


results = [run_one(name, X, feats) for name, (X, feats) in REPRESENTATIONS.items()]

# ---------- 三方对比汇总 ----------
print("\n" + "=" * 72)
print("  三方对比汇总（全量 5 折 OOF 层面，评估器原始指标）")
print("=" * 72)
header = f"{'表示':<20}{'测试R²':>8}{'测试ρ':>8}{'OOF整体MAE':>12}{'OOF最优5%MAE':>14}{'偏差':>9}{'OOF ρ':>8}{'子集通过':>9}"
print(header)
print("-" * len(header.expandtabs()))
for r in results:
    print(f"{r['rep']:<20}{r['test_r2']:>8.3f}{r['test_rho']:>8.3f}"
          f"{r['oof_mae_overall']:>12.2f}{r['oof_local_mae']:>14.2f}"
          f"{r['oof_bias']:>+9.2f}{r['oof_rho']:>8.3f}{r['oof_subset']*100:>8.0f}%")
print("\n全量 OOF 综合判定：")
for r in results:
    print(f"  {r['rep']}: {r['oof_verdict']}")

# ---------- 拟合曲线对比图（3 表示 x 3 图） ----------
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
lims = [y.min() - 5, y.max() + 5]

for row, r in enumerate(results):
    pred_t, oof = r["y_test_pred"], r["oof_full"]

    # 图1: 测试集拟合曲线
    ax = axes[row, 0]
    order = np.argsort(y_test)
    xs, ys = y_test[order], pred_t[order]
    ax.plot(xs, xs, "k--", lw=1.5, label="理想拟合 y=x")
    ax.scatter(xs, ys, c="steelblue", s=25, edgecolors="white", linewidths=0.4, zorder=3)
    trend = pd.Series(ys).rolling(window=41, center=True, min_periods=5).mean().to_numpy()
    ax.plot(xs, trend, c="tomato", lw=2.2, label="拟合趋势（滑动平均）")
    ax.set_title(f"{r['rep']} 测试集 (R²={r['test_r2']:.3f}, MAE={r['test_mae']:.2f})")
    ax.set_xlabel("真实 ramping")
    ax.set_ylabel("预测 ramping")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # 图2: 全量 OOF 散点
    ax = axes[row, 1]
    ax.scatter(y, oof, c="gray", s=10, alpha=0.4)
    ax.plot(lims, lims, "k--", lw=1.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title(f"{r['rep']} 全量OOF (ρ={r['oof_rho']:.3f}, 最优5%偏差={r['oof_bias']:+.2f})")
    ax.set_xlabel("真实 ramping")
    ax.set_ylabel("OOF 预测")
    ax.grid(alpha=0.3)

    # 图3: 测试集残差
    ax = axes[row, 2]
    res = y_test - pred_t
    ax.scatter(pred_t, res, c="steelblue", s=25, edgecolors="white", linewidths=0.4)
    ax.axhline(0, color="k", ls="--", lw=1.5)
    ax.set_title(f"{r['rep']} 测试集残差 (std={res.std():.2f})")
    ax.set_xlabel("预测 ramping")
    ax.set_ylabel("残差 (真实-预测)")
    ax.grid(alpha=0.3)

plt.tight_layout()
out_png = "xgb_alr_clr_fit_curves.png"
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"\n✓ 对比拟合曲线已保存: {out_png}")
print(f"\n复现参数：SEED={SEED}, test_size=0.20, strata=qcut(y,q=10), ALR参照={REF_METRIC}")
