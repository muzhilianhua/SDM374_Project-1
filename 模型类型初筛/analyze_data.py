"""数据分析 + 多模型对比实验"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

np.random.seed(2026)

METRICS = ("cost", "carbon", "peak", "discomfort", "degradation", "ramping", "resilience")
FEATURES = [f"b{b}_{m}" for b in range(1, 4) for m in METRICS]

df = pd.read_csv("dataset_same8/train.csv")
X = df[FEATURES].values.astype(float)
y = df["ramping"].values.astype(float)

print("=" * 60)
print("【1. 数据基本统计】")
print("=" * 60)
print(f"样本数: {len(y)}")
print(f"ramping 范围: [{y.min():.4f}, {y.max():.4f}]")
print(f"ramping 均值: {y.mean():.4f}, 标准差: {y.std():.4f}")
print(f"ramping 中位数: {np.median(y):.4f}")
best_idx = np.argmin(y)
print(f"训练集最优: candidate={df.loc[best_idx, 'candidate_id']}, ramping={y[best_idx]:.4f}")

# Top 10 最优
top10 = np.argsort(y)[:10]
print(f"\nTop-10 训练集真实最优 ramping:")
for rank, idx in enumerate(top10):
    print(f"  #{rank+1}: {df.loc[idx, 'candidate_id']} -> {y[idx]:.4f}")

print("\n" + "=" * 60)
print("【2. 各楼权重分布】")
print("=" * 60)
X37 = X.reshape(-1, 3, 7)
for b in range(3):
    means = X37[:, b, :].mean(axis=0)
    stds = X37[:, b, :].std(axis=0)
    print(f"\n楼 {b+1}:")
    for m, mu, sd in zip(METRICS, means, stds):
        print(f"  {m:12s}: mean={mu:.4f}  std={sd:.4f}  min={X37[:,b,:].min(axis=0)[list(METRICS).index(m)]:.4f}  max={X37[:,b,:].max(axis=0)[list(METRICS).index(m)]:.4f}")

print("\n" + "=" * 60)
print("【3. Top-10 最优候选的权重模式】")
print("=" * 60)
for rank, idx in enumerate(top10):
    w = X37[idx]
    print(f"\n#{rank+1} (ramping={y[idx]:.4f}):")
    for b in range(3):
        parts = [f"{m}={w[b,i]:.3f}" for i, m in enumerate(METRICS) if w[b,i] > 0.01]
        print(f"  楼{b+1}: {' '.join(parts)}")

print("\n" + "=" * 60)
print("【4. 多模型 5 折交叉验证对比】")
print("=" * 60)

# 准备交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=2026)

def cv_score(model_factory, X, y, name):
    maes, r2s = [], []
    for train_idx, val_idx in kf.split(X):
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[val_idx])
        maes.append(mean_absolute_error(y[val_idx], pred))
        r2s.append(r2_score(y[val_idx], pred))
    print(f"{name:30s}: MAE={np.mean(maes):.4f} ± {np.std(maes):.4f}  R2={np.mean(r2s):.4f}")
    return np.mean(maes)

# 模型列表
models = {}

# 线性模型族
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

models["线性回归"] = lambda: LinearRegression()
models["Ridge(alpha=1)"] = lambda: Ridge(alpha=1)
models["Ridge(alpha=100)"] = lambda: Ridge(alpha=100)
models["Lasso(alpha=0.01)"] = lambda: Lasso(alpha=0.01, max_iter=10000)
models["二次+Ridge(1)"] = lambda: Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scale", StandardScaler()),
    ("ridge", Ridge(alpha=1)),
])
models["二次+Ridge(100)"] = lambda: Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scale", StandardScaler()),
    ("ridge", Ridge(alpha=100)),
])
models["二次+Ridge(1000)"] = lambda: Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scale", StandardScaler()),
    ("ridge", Ridge(alpha=1000)),
])

# 树模型
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

models["随机森林(500)"] = lambda: RandomForestRegressor(n_estimators=500, max_depth=10, random_state=2026, n_jobs=-1)
models["GBDT(200)"] = lambda: GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=2026)

# SVR
from sklearn.svm import SVR
models["SVR(RBF)"] = lambda: Pipeline([
    ("scale", StandardScaler()),
    ("svr", SVR(kernel="rbf", C=10, gamma="scale")),
])

# XGBoost (如果可用)
try:
    from xgboost import XGBRegressor
    models["XGBoost(200)"] = lambda: XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=2026, n_jobs=-1)
    models["XGBoost(500)"] = lambda: XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03, random_state=2026, n_jobs=-1)
except ImportError:
    print("XGBoost 未安装，跳过")

# LightGBM (如果可用)
try:
    from lightgbm import LGBMRegressor
    models["LightGBM(200)"] = lambda: LGBMRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=2026, n_jobs=-1)
except ImportError:
    print("LightGBM 未安装，跳过")

# 运行所有模型
results = {}
for name, factory in models.items():
    try:
        mae = cv_score(factory, X, y, name)
        results[name] = mae
    except Exception as e:
        print(f"{name:30s}: 失败 - {e}")

print("\n" + "=" * 60)
print("【5. 模型排名 (按 MAE 升序)】")
print("=" * 60)
for i, (name, mae) in enumerate(sorted(results.items(), key=lambda x: x[1])):
    print(f"  #{i+1}: {name:30s} MAE={mae:.4f}")

# 简单的特征重要性分析
print("\n" + "=" * 60)
print("【6. 特征重要性 (随机森林)】")
print("=" * 60)
rf = RandomForestRegressor(n_estimators=500, max_depth=10, random_state=2026, n_jobs=-1)
rf.fit(X, y)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
for rank, idx in enumerate(indices[:15]):
    print(f"  #{rank+1}: {FEATURES[idx]:20s} importance={importances[idx]:.4f}")
