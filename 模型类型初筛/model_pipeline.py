"""
完整数模建模框架：Ramping 最小化优化

三段式流程：
  Step 1: 预测模型 —— GBDT + 二次Ridge 集成，5折CV验证
  Step 2: 决策模型 —— scipy SLSQP 约束寻优（单纯形约束）
  Step 3: 回滚优化 —— 贝叶斯优化框架（高斯过程代理），用于有限预算黑盒优化

运行方式: python model_pipeline.py
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from scipy.optimize import minimize
from scipy.stats import dirichlet

warnings.filterwarnings("ignore")

np.random.seed(2026)

# ========== 常量 ==========
METRICS = ("cost", "carbon", "peak", "discomfort", "degradation", "ramping", "resilience")
FEATURES = [f"b{b}_{m}" for b in range(1, 4) for m in METRICS]
N_BUILDINGS = 3
N_METRICS = 7
N_TOTAL = N_BUILDINGS * N_METRICS  # 21

TRAIN_CSV = Path("dataset_same8/train.csv")


# ======================================================================
#  Step 1: 数据加载与预测模型
# ======================================================================

def load_data(path=TRAIN_CSV):
    df = pd.read_csv(path, float_precision="round_trip", dtype={"candidate_id": str})
    X = df[FEATURES].values.astype(float)
    y = df["ramping"].values.astype(float)
    ids = df["candidate_id"].tolist()
    return df, X, y, ids


def train_gbdt(X, y):
    """GBDT —— 当前最优单一模型"""
    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=2026,
    )
    model.fit(X, y)
    return model


def train_quadratic_ridge(X, y, alpha=100.0):
    """二次展开 + Ridge —— baseline 方案"""
    model = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha, solver="cholesky")),
    ])
    model.fit(X, y)
    return model


def train_random_forest(X, y):
    """随机森林 —— 特征重要性分析用"""
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        random_state=2026,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def cv_validate(X, y, factory, name, n_splits=5):
    """5折交叉验证"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=2026)
    maes, r2s = [], []
    for tr, va in kf.split(X):
        m = factory()
        m.fit(X[tr], y[tr])
        pred = m.predict(X[va])
        maes.append(mean_absolute_error(y[va], pred))
        r2s.append(r2_score(y[va], pred))
    return np.mean(maes), np.std(maes), np.mean(r2s)


def ensemble_predict(gbdt, ridge, X, weights=(0.6, 0.4)):
    """GBDT + Ridge 加权集成"""
    pred_gbdt = gbdt.predict(X)
    pred_ridge = ridge.predict(X)
    return weights[0] * pred_gbdt + weights[1] * pred_ridge


# ======================================================================
#  Step 2: 约束优化 —— scipy SLSQP
# ======================================================================

def project_simplex(X):
    """投影到合法权重：每栋楼7项非负且和为1"""
    X = np.asarray(X, dtype=float)
    rows = X.reshape(-1, N_METRICS)
    ordered = np.sort(rows, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1) - 1
    counts = np.arange(1, N_METRICS + 1)
    last_positive = np.sum(ordered - cumulative / counts > 0, axis=1) - 1
    theta = cumulative[np.arange(len(rows)), last_positive] / (last_positive + 1)
    return np.maximum(rows - theta[:, None], 0).reshape(X.shape)


def make_objective(gbdt, ridge, ensemble_w=(0.6, 0.4)):
    """构建可微目标函数（用二次Ridge的解析梯度）"""
    # 提取 Ridge 的二次型：f(w) = c + a^T w + 0.5 w^T H w
    poly = ridge.named_steps["poly"]
    scale = ridge.named_steps["scale"]
    ridge_step = ridge.named_steps["ridge"]

    coefs = ridge_step.coef_ / scale.scale_
    c = float(ridge_step.intercept_ - coefs @ scale.mean_)
    a = np.zeros(N_TOTAL)
    H = np.zeros((N_TOTAL, N_TOTAL))
    for powers, coef in zip(poly.powers_, coefs):
        positions = np.flatnonzero(powers)
        if powers.sum() == 1:
            a[positions[0]] += coef
        elif len(positions) == 1:
            H[positions[0], positions[0]] += 2 * coef
        else:
            i, j = positions
            H[i, j] += coef
            H[j, i] += coef

    def quad_fn(w):
        return c + w @ a + 0.5 * w @ H @ w

    def quad_grad(w):
        return a + H @ w

    def objective(w):
        # 先用投影保证合法（SLSQP的约束可能不严格满足所有边界）
        w_proj = project_simplex(w)
        # GBDT 预测
        gbdt_score = gbdt.predict(w_proj.reshape(1, -1))[0]
        # Ridge 预测
        ridge_score = quad_fn(w_proj)
        return ensemble_w[0] * gbdt_score + ensemble_w[1] * ridge_score

    def gradient(w):
        w_proj = project_simplex(w)
        # Ridge 的解析梯度
        grad_ridge = quad_grad(w_proj)
        # GBDT 用有限差分近似（沿投影后的方向）
        eps = 1e-6
        grad_gbdt = np.zeros(N_TOTAL)
        base = gbdt.predict(w_proj.reshape(1, -1))[0]
        for i in range(N_TOTAL):
            wp = w_proj.copy()
            wp[i] += eps
            wp = project_simplex(wp)
            grad_gbdt[i] = (gbdt.predict(wp.reshape(1, -1))[0] - base) / (wp[i] - w_proj[i] + 1e-12)
        return ensemble_w[0] * grad_gbdt + ensemble_w[1] * grad_ridge

    return objective, gradient


def optimize_slsqp(objective, gradient, starts, max_iter=1000):
    """多起点 SLSQP 约束优化"""
    bounds = [(0.0, 1.0)] * N_TOTAL
    constraints = []
    for b in range(N_BUILDINGS):
        idx = list(range(b * N_METRICS, (b + 1) * N_METRICS))
        def sum_cons(w, idx=idx):
            return np.sum(w[idx]) - 1.0
        constraints.append({"type": "eq", "fun": sum_cons})

    results = []
    for start in starts:
        start_proj = project_simplex(start)
        res = minimize(
            objective,
            start_proj,
            method="SLSQP",
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": max_iter, "ftol": 1e-9},
        )
        w_opt = project_simplex(res.x)
        score = objective(w_opt)
        results.append({
            "weights": w_opt.reshape(N_BUILDINGS, N_METRICS),
            "score": float(score),
            "success": res.success,
            "nit": res.nit,
            "message": str(res.message),
        })

    results.sort(key=lambda r: r["score"])
    return results


def generate_starts(X_train, y_train, n_starts=20):
    """生成多个优化起点"""
    rng = np.random.default_rng(2026)
    starts = []

    # 1. 全均匀
    starts.append(np.full(N_TOTAL, 1.0 / N_METRICS))

    # 2. 训练集 Top-N 最优（真实数据）
    n_best = min(10, np.argsort(y_train).size)
    best_idx = np.argsort(y_train)[:n_best]
    for idx in best_idx:
        starts.append(X_train[idx].copy())

    # 3. 随机 Dirichlet 采样
    remaining = n_starts - len(starts)
    if remaining > 0:
        for _ in range(remaining):
            starts.append(rng.dirichlet(np.ones(N_METRICS), size=N_BUILDINGS).flatten())

    return np.array(starts)


# ======================================================================
#  Step 3: 贝叶斯优化框架 (用于回滚/在线阶段)
# ======================================================================

class BayesianOptimizer:
    """
    简易贝叶斯优化：用高斯过程回归做代理模型，EI/acquisition采样。
    适用于有限预算黑盒优化（离线2次 + 在线20×50次场景）。
    """

    def __init__(self, gp_factory=None, seed=2026):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

        self.kernel = ConstantKernel(1.0) * RBF(length_scale=0.3) + WhiteKernel(noise_level=1e-5)
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel, alpha=1e-6, n_restarts_optimizer=5, random_state=seed
        )
        self.X_observed = []
        self.y_observed = []

    def observe(self, w, y):
        """记录一次真实观测"""
        self.X_observed.append(np.asarray(w).flatten())
        self.y_observed.append(float(y))
        if len(self.X_observed) >= 2:
            self.gp.fit(np.array(self.X_observed), np.array(self.y_observed))

    def predict_mean_std(self, X_candidates):
        """GP 预测均值和标准差"""
        if len(self.X_observed) == 0:
            return np.zeros(len(X_candidates)), np.ones(len(X_candidates)) * 10.0
        elif len(self.X_observed) == 1:
            # 单观测点，做简单常数预测 + 大不确定性
            base = self.y_observed[0]
            return np.full(len(X_candidates), base), np.ones(len(X_candidates)) * 10.0
        return self.gp.predict(X_candidates, return_std=True)

    def ei_acquisition(self, X_candidates, xi=0.01):
        """期望改进 (Expected Improvement) —— 最小化问题"""
        from scipy.stats import norm

        means, stds = self.predict_mean_std(X_candidates)
        y_best = np.min(self.y_observed) if self.y_observed else 0.0

        with np.errstate(divide="warn"):
            imp = y_best - means - xi
            Z = np.where(stds > 1e-10, imp / stds, 0.0)
            ei = imp * norm.cdf(Z) + stds * norm.pdf(Z)
            ei[stds < 1e-10] = 0.0

        return ei

    def propose(self, n_candidates=5000):
        """在单纯形约束下采样候选，选择 EI 最高的点"""
        rng = np.random.default_rng(2026 + len(self.X_observed))
        candidates = rng.dirichlet(np.ones(N_METRICS), size=(n_candidates, N_BUILDINGS)).reshape(n_candidates, N_TOTAL)

        # 也加入训练集已知点做参考
        if len(self.X_observed) > 0:
            near = []
            for xo in self.X_observed:
                for _ in range(10):
                    perturb = rng.normal(0, 0.05, size=N_TOTAL)
                    near.append(project_simplex(xo + perturb))
            candidates = np.vstack([candidates, np.array(near)])

        scores = self.ei_acquisition(candidates)
        best_idx = int(np.argmax(scores))
        return candidates[best_idx]


def simulate_response(w, gbdt, ridge, ensemble_w=(0.6, 0.4), noise_scale=2.0):
    """模拟真实响应器（仅开发测试用）——实际应用时替换为真实提交"""
    base = ensemble_predict(gbdt, ridge, w.reshape(1, -1), ensemble_w)[0]
    noise = np.random.normal(0, noise_scale)
    return base + noise


# ======================================================================
#  主流程
# ======================================================================

def run_pipeline():
    print("=" * 70)
    print("  RAMPIING 最小化优化 —— 完整数模建模框架")
    print("=" * 70)

    # ---- 数据加载 ----
    print("\n[Step 0] 加载数据")
    df, X, y, ids = load_data()
    print(f"  样本数: {len(y)}, ramping ∈ [{y.min():.2f}, {y.max():.2f}]")
    print(f"  训练集真实最优: {y.min():.2f} (candidate={ids[np.argmin(y)]})")

    # ---- Step 1: 预测模型训练与验证 ----
    print("\n" + "=" * 70)
    print("  [Step 1] 预测模型训练")
    print("=" * 70)

    print("\n  --- 5 折交叉验证 ---")
    cv_results = {}

    for name, factory in [
        ("GBDT", lambda: GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.03,
            min_samples_leaf=5, subsample=0.8, random_state=2026)),
        ("二次+Ridge(100)", lambda: Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=100, solver="cholesky")),
        ])),
        ("随机森林", lambda: RandomForestRegressor(
            n_estimators=500, max_depth=10, random_state=2026, n_jobs=-1)),
    ]:
        mae, sd, r2 = cv_validate(X, y, factory, name)
        cv_results[name] = mae
        print(f"  {name:20s}: MAE={mae:.4f} ± {sd:.4f}  R²={r2:.4f}")

    # 训练最终模型
    print("\n  --- 训练最终模型 ---")
    t0 = time.time()
    gbdt = train_gbdt(X, y)
    ridge = train_quadratic_ridge(X, y, alpha=100.0)
    rf = train_random_forest(X, y)
    print(f"  训练耗时: {time.time() - t0:.2f}s")

    # 集成验证
    pred_ensemble = ensemble_predict(gbdt, ridge, X)
    print(f"\n  集成模型 (GBDT×0.6 + Ridge×0.4) 全量 MAE: {mean_absolute_error(y, pred_ensemble):.4f}")
    print(f"  GBDT 单独全量 MAE: {mean_absolute_error(y, gbdt.predict(X)):.4f}")
    print(f"  Ridge 单独全量 MAE: {mean_absolute_error(y, ridge.predict(X)):.4f}")

    # 特征重要性
    print("\n  --- 特征重要性 Top-10 (随机森林) ---")
    imp = rf.feature_importances_
    for rank, idx in enumerate(np.argsort(imp)[::-1][:10]):
        print(f"    #{rank+1}: {FEATURES[idx]:20s} = {imp[idx]:.4f}")

    # ---- Step 2: 约束寻优 ----
    print("\n" + "=" * 70)
    print("  [Step 2] 约束寻优 (SLSQP + 多起点)")
    print("=" * 70)

    starts = generate_starts(X, y, n_starts=25)
    print(f"\n  生成 {len(starts)} 个起点")

    objective, gradient = make_objective(gbdt, ridge, ensemble_w=(0.6, 0.4))

    t0 = time.time()
    results = optimize_slsqp(objective, gradient, starts, max_iter=2000)
    print(f"  寻优耗时: {time.time() - t0:.2f}s")

    print(f"\n  --- Top-5 寻优结果 (代理模型预测) ---")
    for rank, r in enumerate(results[:5]):
        print(f"  #{rank+1}: score={r['score']:.4f}  nit={r['nit']}  ok={r['success']}")

    best = results[0]
    print(f"\n  ★ 最优代理预测值: {best['score']:.2f}")
    print(f"  ★ 比训练集真实最优({y.min():.2f})预测低: {y.min() - best['score']:.2f}")

    # ---- Step 3: 贝叶斯优化 (模拟回滚/在线阶段) ----
    print("\n" + "=" * 70)
    print("  [Step 3] 贝叶斯优化框架 (回滚阶段)")
    print("=" * 70)

    bo = BayesianOptimizer(seed=2026)

    # 阶段1：离线探索 (2次)
    print("\n  --- 离线探索 (2次) ---")
    off_line_budgets = [best["weights"].flatten()]
    rng = np.random.default_rng(999)
    for _ in range(1):
        off_line_budgets.append(rng.dirichlet(np.ones(N_METRICS), size=N_BUILDINGS).flatten())

    for i, w in enumerate(off_line_budgets):
        # 用代理模型"模拟"真实响应器 (实际部署时这里替换为提交到真实API)
        y_true = simulate_response(w, gbdt, ridge, noise_scale=3.0)
        bo.observe(w, y_true)
        print(f"    离线 #{i+1}: predicted={ensemble_predict(gbdt, ridge, w.reshape(1, -1))[0]:.2f} → simulated_true={y_true:.2f}")

    # 阶段2：在线迭代优化 (模拟 20次, 每次50轮内部采样)
    print("\n  --- 在线迭代优化 (模拟 20轮 × 50次采样) ---")
    online_budget = 20
    samples_per_round = 50
    best_online = float("inf")
    best_online_w = None

    for round_idx in range(online_budget):
        best_ei_w = None
        best_ei_val = -np.inf

        for _ in range(samples_per_round):
            w_candidate = bo.propose(n_candidates=3000)
            ei_val = bo.ei_acquisition(w_candidate.reshape(1, -1))[0]
            if ei_val > best_ei_val:
                best_ei_val = ei_val
                best_ei_w = w_candidate

        y_true = simulate_response(best_ei_w, gbdt, ridge, noise_scale=3.0)
        bo.observe(best_ei_w, y_true)

        if y_true < best_online:
            best_online = y_true
            best_online_w = best_ei_w.copy()

        if (round_idx + 1) % 5 == 0:
            pred = ensemble_predict(gbdt, ridge, best_ei_w.reshape(1, -1))[0]
            print(f"    Round {round_idx+1:2d}: new_obs={y_true:.2f}  best_so_far={best_online:.2f}  pred={pred:.2f}")

    print(f"\n  ★ 贝叶斯优化最优: simulated={best_online:.2f}")

    # ---- 结果汇总 ----
    print("\n" + "=" * 70)
    print("  结果汇总")
    print("=" * 70)

    # 方案A：训练集最优
    train_best_idx = np.argmin(y)
    train_best_w = X[train_best_idx].reshape(N_BUILDINGS, N_METRICS)
    train_best_ramping = y[train_best_idx]

    # 方案B：SLSQP 约束寻优最优
    slsqp_best_w = best["weights"]
    slsqp_best_pred = best["score"]

    # 方案C：贝叶斯优化最优
    bo_best_w = best_online_w.reshape(N_BUILDINGS, N_METRICS)
    bo_best_sim = best_online

    # 方案D：保守集成（训练集Top-5平均）
    top5_idx = np.argsort(y)[:5]
    conservative_w = X[top5_idx].mean(axis=0).reshape(N_BUILDINGS, N_METRICS)

    proposals = {
        "A_训练集真实最优": {"weights": train_best_w.tolist(), "source_ramping": float(train_best_ramping)},
        "B_SLSQP约束寻优": {"weights": slsqp_best_w.tolist(), "proxy_prediction": float(slsqp_best_pred)},
        "C_贝叶斯优化最优": {"weights": bo_best_w.tolist(), "bo_simulated": float(bo_best_sim)},
        "D_保守集成Top5均值": {"weights": conservative_w.tolist(), "n_sources": 5},
    }

    output = {
        "summary": {
            "train_best_ramping": float(train_best_ramping),
            "train_mean_ramping": float(y.mean()),
            "train_std_ramping": float(y.std()),
            "slsqp_proxy_prediction": float(slsqp_best_pred),
            "bo_simulated_best": float(bo_best_sim),
            "ensemble_cv_mae": float(min(cv_results.values())),
        },
        "proposals": proposals,
        "feature_importance_top": [
            {"feature": FEATURES[idx], "importance": float(imp[idx])}
            for idx in np.argsort(imp)[::-1][:10]
        ],
    }

    output_path = Path("optimization_result.json")
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  完整结果已保存至: {output_path.absolute()}")

    # 打印每个方案的权重详情
    for name, prop in proposals.items():
        print(f"\n  [{name}]")
        w = np.array(prop["weights"])
        for b in range(N_BUILDINGS):
            parts = [f"{m}={w[b,i]:.3f}" for i, m in enumerate(METRICS) if w[b,i] > 0.02]
            print(f"    楼{b+1}: {'  '.join(parts)}")

    return output


if __name__ == "__main__":
    run_pipeline()
