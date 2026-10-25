"""XGBoost + 距离分层 Mondrian Conformal：给寻优候选加“候选相关”的悲观修正

背景（实测结论）：
  RAW+XGB 的 5 折 OOF 在真实最优 5% 区域 bias=mean(pred-true)=+6.39，
  即模型把好解预测得偏高，属于“保守”而非“乐观”；`pred ~ true` 斜率仅 0.743，
  是树模型向均值收缩。真正的风险不在已观测候选的排序，而在连续寻优时
  模型对稀疏区域的外推。

  实测：候选到训练集的 Aitchison kNN 距离与 |残差| 强相关
  （近 25% MAE≈2.8，远 25% MAE≈7.1，|res| 的 p90 从 6.25 升到 16.38）。

做法（Mondrian conformal）：
  1. 8:2 分层划分，训练集内 5 折选超参
  2. 训练集内再做 5 折 OOF，得到校准残差，同时记录 OOF 点到“该折训练集”的距离
  3. 用 CLR 空间的平均 kNN 距离把样本分成若干距离层，每层单独取 |残差| 的
     90% 分位数 q̂（而不是全局一个 q̂）
  4. 候选打分 = pred + β·q̂_bin(d)，按上界排序
     —— 全局常数 q̂ 只会给所有候选加同一个数，排序不变；
        距离分层让远端候选的区间更宽，排序才会真正改变

运行方式: python xgb_conformal_strat82.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
from xgboost import XGBRegressor

SEED = 2026
np.random.seed(SEED)

METRICS = ("cost", "carbon", "peak", "discomfort", "degradation", "ramping", "resilience")
FEATURES = [f"b{b}_{m}" for b in range(1, 4) for m in METRICS]

K_NN = 10          # 距离用的近邻数
N_BINS = 4         # 距离分层层数
COVERAGE = 0.90    # conformal 目标覆盖率
BETA = 0.5        # 悲观系数：1.0 = 直接用 90% 上界；0.5 = 只加半宽，排序损失更小


# ---------- 数据加载 ----------
df = pd.read_csv("dataset_same8/train.csv", float_precision="round_trip")
X = df[FEATURES].to_numpy(dtype=float)
y = df["ramping"].to_numpy(dtype=float)

strata = pd.qcut(y, q=10, labels=False, duplicates="drop")
idx_train, idx_test = train_test_split(
    np.arange(len(y)), test_size=0.20, stratify=strata, random_state=SEED, shuffle=True
)
X_tr, y_tr = X[idx_train], y[idx_train]
X_te, y_te = X[idx_test], y[idx_test]


# ---------- CLR / Aitchison 距离 ----------
def clr_transform(X_raw: np.ndarray) -> np.ndarray:
    """逐楼 CLR：log(x_i / g(x))。Aitchison 距离 = CLR 空间的欧氏距离。"""
    log_x = np.log(X_raw)
    blocks = []
    for b in range(3):
        block = log_x[:, b * 7:(b + 1) * 7]
        blocks.append(block - block.mean(axis=1, keepdims=True))
    return np.hstack(blocks)


def knn_mean_distance(ref_clr: np.ndarray, query_clr: np.ndarray, k: int = K_NN) -> np.ndarray:
    """query 到 ref 的平均 kNN Aitchison 距离。"""
    k = min(k, len(ref_clr))
    nn = NearestNeighbors(n_neighbors=k).fit(ref_clr)
    return nn.kneighbors(query_clr)[0].mean(axis=1)


# ---------- 训练集内选超参 ----------
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
param_grid = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [300, 600],
}

print("=" * 72)
print("  XGBoost + 距离分层 Mondrian Conformal (8:2 分层划分)")
print("=" * 72)

gs = GridSearchCV(
    XGBRegressor(subsample=0.9, colsample_bytree=0.9, tree_method="hist",
                 random_state=SEED, n_jobs=-1),
    param_grid, scoring="neg_mean_absolute_error", cv=cv, n_jobs=1, refit=True,
)
gs.fit(X_tr, y_tr)
best_params = gs.best_params_
print(f"\n  ★ 最优配置: {best_params}")
print(f"  CV MAE = {-gs.best_score_:.4f}")


def make_xgb():
    return XGBRegressor(
        **best_params, subsample=0.9, colsample_bytree=0.9,
        tree_method="hist", random_state=SEED, n_jobs=-1,
    )


# ---------- 距离分层 conformal ----------
class DistanceStratifiedConformal:
    """Mondrian conformal：按候选到训练集的距离分层，分别校准区间半宽。

    fit(X, y) 内部用 5 折 OOF 得到校准残差；predict(X_new) 返回
    (普通预测, 悲观上界)。上界 = pred + beta * q̂_bin(d)。
    """

    def __init__(self, model_factory, coverage=COVERAGE, k=K_NN,
                 n_bins=N_BINS, beta=BETA, random_state=SEED):
        self.model_factory = model_factory
        self.coverage = coverage
        self.k = k
        self.n_bins = n_bins
        self.beta = beta
        self.random_state = random_state

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        clr = clr_transform(X_train)
        oof = np.zeros(len(y_train))
        dist_oof = np.zeros(len(y_train))

        # 每个 OOF 点到“该折训练集”的距离，与模型实际可见的样本保持一致
        for tr, va in KFold(5, shuffle=True, random_state=self.random_state).split(X_train):
            m = self.model_factory().fit(X_train[tr], y_train[tr])
            oof[va] = m.predict(X_train[va])
            dist_oof[va] = knn_mean_distance(clr[tr], clr[va], self.k)

        # 距离分层边界取校准集的分位数；q̂ 只在层内估计
        self.edges = np.percentile(
            dist_oof, np.linspace(0, 100, self.n_bins + 1)[1:-1]
        )
        bins = np.digitize(dist_oof, self.edges)
        resid = np.abs(y_train - oof)
        self.global_q = float(np.quantile(resid, self.coverage))
        self.qhat = np.array([
            float(np.quantile(resid[bins == b], self.coverage))
            if (bins == b).sum() >= 20 else self.global_q
            for b in range(self.n_bins)
        ])
        self.bin_counts = np.array([int((bins == b).sum()) for b in range(self.n_bins)])
        self.oof = oof

        self.model = self.model_factory().fit(X_train, y_train)
        self.ref_clr = clr
        return self

    def predict(self, X_new: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_new = np.asarray(X_new, dtype=float)
        pred = self.model.predict(X_new)
        dist_new = knn_mean_distance(self.ref_clr, clr_transform(X_new), self.k)
        bins = np.clip(np.digitize(dist_new, self.edges), 0, self.n_bins - 1)
        upper = pred + self.beta * self.qhat[bins]
        return pred, upper, dist_new, bins


conf = DistanceStratifiedConformal(make_xgb).fit(X_tr, y_tr)
pred_te, upper_te, dist_te, bin_te = conf.predict(X_te)
upper_te_10 = pred_te + 1.0 * conf.qhat[bin_te]   # β=1.0 对照

print("\n--- 距离分层校准结果（校准集 = 训练集 OOF） ---")
print(f"  距离分层边界: {np.round(conf.edges, 2)}")
print(f"  各层样本数:   {conf.bin_counts}")
print(f"  各层 q̂(90%):  {np.round(conf.qhat, 2)}   （不分层的全局 OOF q̂={conf.global_q:.2f}）")

# ---------- 覆盖率与误差随距离的变化 ----------
print("\n--- 测试集：分层覆盖率 / 区间宽度 / 预测误差 ---")
print(f"  {'距离层':<8}{'n':>6}{'覆盖率':>9}{'半宽q̂':>9}{'MAE':>8}{'距离均值':>10}")
covered = np.abs(y_te - pred_te) <= conf.qhat[bin_te]
for b in range(N_BINS):
    m = bin_te == b
    if m.sum() == 0:
        continue
    print(f"  {b:<8}{m.sum():>6}{covered[m].mean():>9.3f}"
          f"{conf.qhat[b]:>9.2f}{mean_absolute_error(y_te[m], pred_te[m]):>8.2f}"
          f"{dist_te[m].mean():>10.2f}")
print(f"  {'合计':<8}{len(y_te):>6}{covered.mean():>9.3f}")


# ---------- 选点对比 ----------
def selection_stats(score: np.ndarray, y_pool: np.ndarray, k: int = 10) -> dict:
    idx = np.argsort(score)[:k]
    sel = y_pool[idx]
    srt = np.sort(y_pool)
    pct = np.array([np.searchsorted(srt, v) / len(srt) for v in sel])
    return {
        "mean_pct": float(pct.mean() * 100),
        "best": float(sel.min()),
        "gap": float(sel.min() - y_pool.min()),
        "dist": float(dist_te[idx].mean()),
    }


raw = selection_stats(pred_te, y_te)
pess_05 = selection_stats(upper_te, y_te)
pess_10 = selection_stats(upper_te_10, y_te)


def two_stage_stats(upper_scores: np.ndarray) -> dict:
    """先按普通预测取 Top-50，再在 Top-50 内按上界重排。"""
    top50 = np.argsort(pred_te)[:50]
    order50 = top50[np.argsort(upper_scores[top50])]
    sel = y_te[order50[:10]]
    srt = np.sort(y_te)
    return {
        "mean_pct": float(np.mean([np.searchsorted(srt, v) / len(srt) for v in sel]) * 100),
        "best": float(sel.min()),
        "gap": float(sel.min() - y_te.min()),
        "dist": float(dist_te[order50[:10]].mean()),
    }


print("\n--- 测试集 Top-10 选点对比（分位数越小越好，gap=与池内真实最优的差） ---")
print(f"  {'策略':<24}{'平均真实分位':>14}{'最优真实值':>12}{'gap':>8}{'选中点距离':>12}")
for name, r in [
    ("普通预测排序", raw),
    ("上界排序(β=0.5)", pess_05),
    ("上界排序(β=1.0)", pess_10),
    ("Top-50后上界(β=0.5)", two_stage_stats(upper_te)),
    ("Top-50后上界(β=1.0)", two_stage_stats(upper_te_10)),
]:
    print(f"  {name:<24}{r['mean_pct']:>13.2f}%{r['best']:>12.2f}{r['gap']:>8.2f}{r['dist']:>12.2f}")

print("\n--- 参考：测试集整体指标 ---")
print(f"  MAE={mean_absolute_error(y_te, pred_te):.3f}  "
      f"Spearman ρ={spearmanr(y_te, pred_te).statistic:.4f}")


# ---------- Lab6 评估器（口径已修正：正 bias = 保守） ----------
LAB6 = Path(r"D:\HuaweiMoveData\Users\11659\Desktop\机器学习项目\Lab6")
if not (LAB6 / "surrogate_evaluator.py").exists():
    raise FileNotFoundError(f"找不到评估器: {LAB6 / 'surrogate_evaluator.py'}")
sys.path.insert(0, str(LAB6))
from surrogate_evaluator import SurrogateEvaluator  # noqa: E402

print("\n" + "=" * 72)
print("  Lab6 SurrogateEvaluator（测试集，bias 口径已修正）")
print("=" * 72)
ev = SurrogateEvaluator(X_te, y_te, feature_names=FEATURES)
report = ev.evaluate(
    conf.model,
    model_name=f"XGB+DistConformal(depth={best_params['max_depth']}, "
               f"lr={best_params['learning_rate']}, n={best_params['n_estimators']})",
    oof_predictions=pred_te,
    run_subset=True,
    model_factory=make_xgb,
    subset_folds=3,
    subset_topk=10,
)
print(report.summary())

print(f"\n复现参数：SEED={SEED}, test_size=0.20, K_NN={K_NN}, N_BINS={N_BINS}, "
      f"COVERAGE={COVERAGE}, BETA={BETA}")
