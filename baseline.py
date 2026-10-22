"""二次岭回归 baseline：先学会预测 ramping，再用梯度下降寻找控制权重。

阅读顺序：train_model（训练）→ quadratic_form（二次函数）→ gradient_search（寻优）。
训练改变回归系数 beta；寻优固定 beta，只改变三栋楼的 21 个控制权重 w。
"""
from pathlib import Path
import argparse
import json
import os

# 固定数值库线程数，避免多进程时每个进程都占用大量线程。
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

METRICS = ("cost", "carbon", "peak", "discomfort", "degradation", "ramping", "resilience")
FEATURES = [f"b{building}_{metric}" for building in range(1, 4) for metric in METRICS]
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)


def load_table(path):
    """读取均值表：X 是 N×21 权重，y 是 N 个跨周平均 ramping。"""
    table = pd.read_csv(path, float_precision="round_trip", dtype={"candidate_id": str})
    if set(table.columns) != {"candidate_id", *FEATURES, "ramping"}:
        raise ValueError("需要均值表：candidate_id、21个权重列、ramping")
    if len(table) < 2 or table.candidate_id.isna().any() or table.candidate_id.duplicated().any():
        raise ValueError("至少需要两组数据，且 candidate_id 不能缺失或重复")
    table = table.sort_values("candidate_id", kind="stable").reset_index(drop=True)
    X = table[FEATURES].to_numpy(dtype=float)  # 编号不作为输入特征。
    y = table.ramping.to_numpy(dtype=float)
    sums = X.reshape(-1, 3, 7).sum(axis=2)
    if not np.isfinite(X).all() or (X < 0).any() or not np.allclose(sums, 1, atol=1e-12, rtol=0):
        raise ValueError("每栋楼的7个权重必须有限、非负且和为1")
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("ramping 必须是有限的非负数")
    return table.candidate_id.tolist(), X, y


def make_model(alpha):
    """二次展开、标准化、岭回归组成同一个流程，避免交叉验证泄漏。"""
    return Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha, solver="cholesky")),
    ])


def train_model(X, y):
    """只用训练表做5折交叉验证选alpha，然后用全部训练行重新拟合。"""
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score
    if len(y) < 5:
        raise ValueError("5折交叉验证至少需要5条训练数据")
    folds = list(KFold(n_splits=5, shuffle=True, random_state=2026).split(X))
    trials = []
    best_error, best_alpha, best_oof = float("inf"), None, None
    for alpha in ALPHAS:
        predicted = np.empty(len(y))
        for fitting, held_out in folds:
            # 每一折的均值、标准差和回归系数都只用该折训练部分计算。
            model = make_model(alpha).fit(X[fitting], y[fitting])
            predicted[held_out] = model.predict(X[held_out])
        error = float(mean_absolute_error(y, predicted))
        trials.append({"alpha": alpha, "cv_mae": error,
                       "cv_r2": float(r2_score(y, predicted))})
        if error < best_error:
            best_error, best_alpha, best_oof = error, alpha, predicted.copy()
    model = make_model(best_alpha).fit(X, y)
    # 这是选参用的内部交叉验证分数，不是独立测试集成绩。
    return model, {"alpha": best_alpha, "trials": trials,
                   "cv_mae": best_error, "cv_r2": float(r2_score(y, best_oof)),
                   "oof_predictions": best_oof.tolist()}


def search_starts(X, y, starts=16):
    if starts < 2:
        raise ValueError("至少使用两个寻优起点")
    rng = np.random.default_rng(2026)
    return np.concatenate([
        np.full((1, 21), 1 / 7), X[[int(np.argmin(y))]],
        rng.dirichlet(np.ones(7), size=(starts - 2, 3)).reshape(-1, 21),
    ])


def save_model(model, initial_weights, path):
    """保存预测模型及固定寻优起点，不保存训练数据或仿真环境。"""
    c, a, H = quadratic_form(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, offset=c, linear=a, hessian=H, search_starts=initial_weights,
        polynomial_powers=model.named_steps["poly"].powers_,
        feature_mean=model.named_steps["scale"].mean_,
        feature_scale=model.named_steps["scale"].scale_,
        ridge_coefficients=model.named_steps["ridge"].coef_,
        ridge_intercept=model.named_steps["ridge"].intercept_,
        alpha=model.named_steps["ridge"].alpha,
    )


def quadratic_form(model):
    """将已训练模型等价写成 f(w) = c + aᵀw + 0.5 wᵀHw，便于求导。"""
    poly, scale, ridge = (model.named_steps[k] for k in ("poly", "scale", "ridge"))
    # 将标准化 z=(特征-均值)/标准差 合并到回归系数中。
    coefficients = ridge.coef_ / scale.scale_
    c = float(ridge.intercept_ - coefficients @ scale.mean_)
    a, H = np.zeros(21), np.zeros((21, 21))
    for powers, coefficient in zip(poly.powers_, coefficients):
        positions = np.flatnonzero(powers)  # 这一项涉及哪些输入变量。
        if powers.sum() == 1:              # 一次项 coefficient * w[i]
            a[positions[0]] += coefficient
        elif len(positions) == 1:         # 平方项 coefficient * w[i]**2
            H[positions[0], positions[0]] += 2 * coefficient
        else:                             # 交叉项 coefficient * w[i] * w[j]
            i, j = positions
            H[i, j] += coefficient
            H[j, i] += coefficient
    # 平方项乘2、交叉项填两侧，都用于抵消公式中的0.5。
    return c, a, H


def predict_quadratic(X, c, a, H):
    """批量计算二次函数；einsum 对每行权重计算 wᵀHw。"""
    X = np.asarray(X, dtype=float)
    return c + X @ a + 0.5 * np.einsum("...i,ij,...j->...", X, H, X)


def project_weights(X):
    """找到最近的合法权重：每栋楼7项非负、总和为1。"""
    X = np.asarray(X, dtype=float)
    if X.shape[-1] != 21 or not np.isfinite(X).all():
        raise ValueError("需要有限的21维权重向量")
    rows = X.reshape(-1, 7)  # 每栋楼独立处理。
    ordered = np.sort(rows, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1) - 1
    # 找到阈值 theta，使 sum(max(w - theta, 0)) = 1。
    counts = np.arange(1, 8)
    last_positive = np.sum(ordered - cumulative / counts > 0, axis=1) - 1
    theta = cumulative[np.arange(len(rows)), last_positive] / (last_positive + 1)
    return np.maximum(rows - theta[:, None], 0).reshape(X.shape)


def gradient_search(c, a, H, starts, *, max_iter=5000, tolerance=1e-7):
    """多起点投影梯度下降。全程使用预测模型，不调用真实控制器。"""
    if max_iter < 1 or tolerance <= 0:
        raise ValueError("迭代次数和容差必须为正")
    X = project_weights(starts)  # 每行是一个起点；所有起点一起更新。
    # H 描述曲率；曲率越大，步长越小，避免一步走得太远。
    step = 1.0 / max(float(np.max(np.abs(np.linalg.eigvalsh(H)))), 1.0)
    for iteration in range(1, max_iter + 1):
        gradient = X @ H + a    # 单个向量的梯度是 H @ w + a。
        updated = project_weights(X - step * gradient)
        residuals = np.max(np.abs(updated - X), axis=1) / step
        X = updated
        if residuals.max() <= tolerance:
            break              # 所有起点都基本停止变化。
    scores = predict_quadratic(X, c, a, H)
    best = int(np.argmin(scores))
    remaining = np.max(np.abs(project_weights(X - step * (X @ H + a)) - X), axis=1) / step
    return X[best].reshape(3, 7), {
        "iterations": iteration, "converged_starts": int(np.count_nonzero(remaining <= tolerance))
    }



MODEL = Path(__file__).resolve().with_name("model.npz")


def train(train_csv, model_path=MODEL):
    """从单张训练表训练，保存模型；不调用模拟器。"""
    _, X, y = load_table(train_csv)
    model, report = train_model(X, y)
    save_model(model, search_starts(X, y), model_path)
    print(f"交叉验证选出的 alpha：{report['alpha']:g}")
    return Path(model_path)


def optimize(model_path=MODEL):
    """固定模型，对21个输入权重做梯度寻优。"""
    with np.load(model_path, allow_pickle=False) as model:
        weights, _ = gradient_search(float(model["offset"]), model["linear"],
                                     model["hessian"], model["search_starts"])
    return weights


def predict(weights, model_path=MODEL):
    """预测一组或多组权重，返回的不是控制器实测分数。"""
    x = np.asarray(weights, dtype=float)
    single = x.shape in ((21,), (3, 7))
    x = x.reshape(1, 21) if single else x
    if x.ndim != 2 or x.shape[1] != 21 or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("每组输入需要21个有限非负数")
    if not np.allclose(x.reshape(-1, 3, 7).sum(axis=2), 1, atol=1e-12, rtol=0):
        raise ValueError("每栋楼的7个权重之和需要为1")
    with np.load(model_path, allow_pickle=False) as model:
        result = predict_quadratic(x, float(model["offset"]), model["linear"], model["hessian"])
    return float(result[0]) if single else result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train", help="只用训练CSV训练模型")
    fit.add_argument("--train", type=Path, required=True)
    fit.add_argument("--model", type=Path, default=MODEL)
    search = commands.add_parser("optimize", help="对模型做梯度寻优")
    search.add_argument("--model", type=Path, default=MODEL)
    search.add_argument("--output", type=Path)
    score = commands.add_parser("predict", help="预测权重的ramping")
    score.add_argument("--model", type=Path, default=MODEL)
    score.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        print(f"模型已保存：{train(args.train, args.model)}")
    elif args.command == "optimize":
        text = json.dumps(optimize(args.model).tolist(), ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
        print(text)
    else:
        weights = json.loads(args.weights.read_text(encoding="utf-8-sig"))
        print(json.dumps({"predicted_ramping": np.asarray(predict(weights, args.model)).tolist()}))


if __name__ == "__main__":
    main()
