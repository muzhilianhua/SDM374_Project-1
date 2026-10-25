"""修正版 Ramping 优化 —— 快速版，加入可信度评估"""
import json, warnings, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")
np.random.seed(2026)

METRICS = ("cost","carbon","peak","discomfort","degradation","ramping","resilience")
FEATURES = [f"b{b}_{m}" for b in range(1,4) for m in METRICS]
NB, NM, NT = 3, 7, 21

def project_simplex(X):
    X = np.asarray(X, float).reshape(-1, NM)
    ordered = np.sort(X,1)[:,::-1]
    cum = np.cumsum(ordered,1)-1
    theta = cum[np.arange(len(X)), np.sum(ordered - cum/np.arange(1,NM+1) > 0, 1)-1] / np.sum(ordered - cum/np.arange(1,NM+1) > 0, 1)
    return np.maximum(X - theta[:,None], 0).reshape(-1, NT)

def trust_check(w, X_train, gbdt, ridge):
    """返回 (gbdt_pred, ridge_pred, gap, nn_dist)"""
    w = np.asarray(w, float).flatten()
    gp = gbdt.predict(w.reshape(1,-1))[0]
    rp = ridge.predict(w.reshape(1,-1))[0]
    gap = abs(gp - rp)
    nn = np.min(np.linalg.norm(X_train - w, axis=1))
    return gp, rp, gap, nn

def local_refine(center, gbdt, ridge, X_train, n_iter=200):
    """在 center 附近做简单的局部精修"""
    rng = np.random.default_rng(abs(hash(center.tobytes())) % (2**31))
    candidates = [center.copy()]
    # Dirichlet 小扰动 + Ridge 二次下降
    for _ in range(n_iter):
        # 加随机扰动
        perturb = rng.normal(0, 0.05, NT)
        c = project_simplex(center + perturb)
        candidates.append(c)
    return candidates

def main():
    t0 = time.time()
    print("=" * 60)
    print("  RAMPIING 优化框架 —— 完整建模分析")
    print("=" * 60)

    # 加载
    df = pd.read_csv("dataset_same8/train.csv")
    X = df[FEATURES].values.astype(float)
    y = df["ramping"].values.astype(float)
    ids = df["candidate_id"].tolist()
    print(f"\n数据: {len(y)} 组, ramping ∈ [{y.min():.2f}, {y.max():.2f}]")
    print(f"训练集真实最优: {y.min():.2f} (candidate={ids[np.argmin(y)]})")

    # 训练模型
    print("\n训练 GBDT + 二次Ridge...")
    gbdt = GradientBoostingRegressor(n_estimators=300,max_depth=5,learning_rate=0.03,min_samples_leaf=5,subsample=0.8,random_state=2026).fit(X,y)
    ridge = Pipeline([("poly",PolynomialFeatures(degree=2,include_bias=False)),("scale",StandardScaler()),("ridge",Ridge(alpha=100,solver="cholesky"))]).fit(X,y)

    # 全训练集代理预测
    pred_gbdt = gbdt.predict(X)
    pred_ridge = ridge.predict(X)
    pred_ens = 0.7*pred_gbdt + 0.3*pred_ridge
    from sklearn.metrics import mean_absolute_error, r2_score
    print(f"训练集内 MAE: GBDT={mean_absolute_error(y,pred_gbdt):.2f}, Ridge={mean_absolute_error(y,pred_ridge):.2f}")

    # ====== 关键分析: 训练集内外推可信度 ======
    print("\n" + "=" * 60)
    print("  可信度分析")
    print("=" * 60)

    # 训练集点的模型差异分布
    gaps_in = np.abs(pred_gbdt - pred_ridge)
    print(f"\n训练集内 GBDT vs Ridge 差异:")
    print(f"  均值={gaps_in.mean():.2f}, 95分位={np.percentile(gaps_in,95):.2f}, 最大={gaps_in.max():.2f}")

    # Top-10 真实最优点的模型表现
    top10_idx = np.argsort(y)[:10]
    print(f"\nTop-10 真实最优点 ({y.min():.1f} ~ {y[top10_idx[-1]]:.1f}):")
    for rank, idx in enumerate(top10_idx):
        gp, rp, gap, _ = trust_check(X[idx], X, gbdt, ridge)
        print(f"  #{rank+1}: id={ids[idx]:7s} true={y[idx]:7.2f} gbdt={gp:7.2f} ridge={rp:7.2f} gap={gap:5.2f} ✓" if gap < 10 else f"  #{rank+1}: id={ids[idx]:7s} true={y[idx]:7.2f} gbdt={gp:7.2f} ridge={rp:7.2f} gap={gap:5.2f} ⚠")

    # ====== 信任域局部优化 ======
    print("\n" + "=" * 60)
    print("  信任域局部优化")
    print("=" * 60)

    # 在 Top-20 训练点周围做 Dirichlet 扰动采样
    rng = np.random.default_rng(2026)
    top20_idx = np.argsort(y)[:20]
    all_cand = []

    for idx in top20_idx:
        center = X[idx]
        # 每个点采样 200 个 Dirichlet 扰动
        for _ in range(200):
            perturb = rng.normal(0, 0.06, NT)
            c = project_simplex(center + perturb)
            gp, rp, gap, nn = trust_check(c, X, gbdt, ridge)
            ens = 0.7*gp + 0.3*rp
            all_cand.append((ens, gap, nn, gp, rp, c))

    # 筛选可信候选
    trusted = [c for c in all_cand if c[1] < 10 and c[2] < 0.5]
    trusted.sort(key=lambda x: x[0])
    print(f"\n采样 {len(all_cand)} 个候选, 可信 (gap<10 & nn<0.5): {len(trusted)}")

    # 也看看 GBDT 单独最优的可信候选
    gbdt_trusted = [c for c in all_cand if c[1] < 10 and c[2] < 0.5]
    gbdt_trusted.sort(key=lambda x: x[3])  # 按 GBDT 预测排

    print(f"\n可信候选 Top-8 (按集成代理预测):")
    for rank, (ens, gap, nn, gp, rp, w) in enumerate(trusted[:8]):
        print(f"  #{rank+1}: ens={ens:.2f} gbdt={gp:.2f} ridge={rp:.2f} gap={gap:.1f} nn={nn:.3f}")

    # ====== 4 套提交方案 ======
    print("\n" + "=" * 60)
    print("  推荐提交方案")
    print("=" * 60)

    plans = {}

    # 方案 A: 可信候选最优
    if trusted:
        w_a = trusted[0][5].reshape(NB, NM).tolist()
        plans["A_可信代理最优"] = {
            "weights": w_a,
            "proxy_ensemble": round(trusted[0][0], 2),
            "gbdt_pred": round(trusted[0][3], 2),
            "ridge_pred": round(trusted[0][4], 2),
            "model_gap": round(trusted[0][1], 2),
            "nn_distance": round(trusted[0][2], 4),
            "confidence": "high",
        }

    # 方案 B: 训练集真实最优
    best_idx = np.argmin(y)
    w_b = X[best_idx].reshape(NB, NM).tolist()
    gp_b, rp_b, gap_b, nn_b = trust_check(X[best_idx], X, gbdt, ridge)
    plans["B_训练集真实最优"] = {
        "weights": w_b,
        "known_ramping": float(y[best_idx]),
        "gbdt_pred": round(gp_b, 2),
        "ridge_pred": round(rp_b, 2),
        "model_gap": round(gap_b, 2),
        "confidence": "certain",
    }

    # 方案 C: 训练集 Top-5 均值
    top5_idx = np.argsort(y)[:5]
    w_c = X[top5_idx].mean(axis=0).reshape(NB, NM).tolist()
    gp_c, rp_c, gap_c, nn_c = trust_check(np.array(w_c).flatten(), X, gbdt, ridge)
    plans["C_Top5_训练均值"] = {
        "weights": w_c,
        "origin_mean": round(y[top5_idx].mean(), 2),
        "gbdt_pred": round(gp_c, 2),
        "ridge_pred": round(rp_c, 2),
        "model_gap": round(gap_c, 2),
        "confidence": "high",
    }

    # 方案 D: 可信候选集成 (取 Top-3 的加权平均)
    if len(trusted) >= 3:
        top3_w = np.array([c[5] for c in trusted[:3]])
        weights = rng.dirichlet(np.ones(3)+2, size=500)
        best_w, best_score = None, float("inf")
        for w in weights:
            avg = top3_w.T @ w
            gp, rp, gap, nn = trust_check(avg, X, gbdt, ridge)
            score = 0.7*gp + 0.3*rp
            if score < best_score and gap < 12:
                best_score = score
                best_w = avg
        if best_w is not None:
            gp_d, rp_d, gap_d, nn_d = trust_check(best_w, X, gbdt, ridge)
            plans["D_可信候选集成"] = {
                "weights": best_w.reshape(NB, NM).tolist(),
                "proxy_ensemble": round(best_score, 2),
                "gbdt_pred": round(gp_d, 2),
                "ridge_pred": round(rp_d, 2),
                "model_gap": round(gap_d, 2),
                "confidence": "high" if gap_d < 10 else "medium",
            }

    # 打印方案
    for name, plan in plans.items():
        conf = plan.pop("confidence", "N/A")
        print(f"\n  [{name}] (可信级别: {conf})")
        for k, v in plan.items():
            if k != "weights":
                print(f"    {k}: {v}")
        w = np.array(plan["weights"])
        for b in range(NB):
            parts = [f"{m}={w[b,i]:.3f}" for i, m in enumerate(METRICS) if w[b,i] > 0.02]
            print(f"    楼{b+1}: {'  '.join(parts)}")

    # 保存
    out = {
        "summary": {
            "train_best_ramping": float(y.min()),
            "train_top10_mean": float(y[top10_idx].mean()),
            "train_mean": float(y.mean()),
            "train_std": float(y.std()),
            "trust_analysis": {
                "in_sample_model_gap_mean": round(float(gaps_in.mean()), 2),
                "in_sample_model_gap_p95": round(float(np.percentile(gaps_in, 95)), 2),
            },
        },
        "plans": plans,
    }
    out_path = Path("optimization_result_v2.json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ 结果已保存至: {out_path.absolute()}")
    print(f"\n总耗时: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
