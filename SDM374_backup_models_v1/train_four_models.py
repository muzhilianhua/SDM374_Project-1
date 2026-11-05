# -*- coding: utf-8 -*-
"""四个代理模型的训练（拟合）源代码 —— 可重现 gp-ard / tail5opt / hp_wlearn / blend_xgb_w20x3。

本文件是 `外推方案/模型库/build_library.py` 与 `外推方案/gp提分/common.py` 的精简可移植版本，
只保留这四个配置需要的代码：分块 ARD Matérn-5/2 核、全区域交互项、可学习交互权重、
加权对数边际似然（低尾加权）、与树模型的固定权重混合。产物为版本无关的纯数组载荷
model.npz + params.json（不含 pickle 的 numpy 数组，numpy 1.24.x / 2.x 均可直接加载）。

入口命令（在包根目录执行；需要 numpy/scipy/scikit-learn/pandas/joblib，混合模型另需 xgboost）：
    python train_four_models.py --list
    python train_four_models.py --model gp-ard
    python train_four_models.py --model tail5opt
    python train_four_models.py --model hp_wlearn
    python train_four_models.py --model blend_xgb_w20x3
    python train_four_models.py --model all
    python train_four_models.py --probe-seed 2027        # 只打印实际生效的多起点种子

默认读取 仓库根/dataset_same8/train.csv；产物写到 models/<模型名>/model.npz（可用 --out 改）。
--seed（默认 2026）会一路传给多起点优化器与 GaussianProcessRegressor.random_state，
并写入 fit_report.json 的 seed_used（实际生效值）。
注意：四个模型都在**全部 2048 行**上 refit（交付用），这与 OOF 的"逐折重拟合"是两回事；
OOF 与 v3.2 评分见 run_oof_and_score.py。
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Hyperparameter, Kernel, Matern, WhiteKernel

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import modellib as M  # noqa: E402  仅用于特征契约与 schema 常量

TARGET = "ramping"
SEED = 2026

# ─────────────────────────── 四个配置（与历史实验逐字一致） ───────────────────────────
BASE = dict(transform="raw", bounds=(0.03, 3.0), n_starts=1, maxiter=18,
            noise=0.03, noise_bounds=(0.001, 0.3))
TAIL5OPT = dict(transform="raw", bounds=(0.03, 6.0), weighted_opt=True, weight_k=5.0,
                n_starts=3, maxiter=80, noise=0.03, noise_bounds=(0.001, 0.3))
HP_WLEARN = dict(transform="raw", bounds=(0.03, 6.0), n_starts=4, maxiter=100,
                 noise=0.03, noise_bounds=(0.001, 0.3), learn_weight=True)
HP_MS = dict(transform="raw", bounds=(0.03, 6.0), n_starts=4, maxiter=100,
             noise=0.03, noise_bounds=(0.001, 0.3))
BLEND_ALPHA = 0.15
BLEND_TREE = "xgb_w20x3_full"

CONFIGS = {"gp-ard": BASE, "tail5opt": TAIL5OPT, "hp_wlearn": HP_WLEARN,
           "blend_xgb_w20x3": HP_MS}


# ─────────────────────────── 核：分块 ARD + 全区域交互 ───────────────────────────
class BlockARDKernel(Kernel):
    """逐块 ARD Matérn-5/2 的加法核 + 全区域交互项。

    K = (1-w)/3 * Σ_b Matern5/2(X_b, Y_b; ls) + w * Matern5/2(X, Y; [ls,ls,ls])
    learn_weight=True 时 w 作为核超参（对数空间），否则固定为 global_weight。
    """

    def __init__(self, blocks=None, full=None, length_scale=(0.5,) * 7,
                 global_weight=0.1, ls_bounds=(0.03, 3.0), learn_weight=False, nu=2.5):
        self.blocks = blocks if blocks is not None else [np.arange(7 * b, 7 * b + 7) for b in range(3)]
        self.full = full if full is not None else np.arange(21)
        self.length_scale = np.asarray(length_scale, float)
        self.global_weight = float(global_weight)
        self.ls_bounds = tuple(ls_bounds)
        self.learn_weight = bool(learn_weight)
        self.nu = nu
        if self.learn_weight:
            self.hyperparameter_global_weight = Hyperparameter("global_weight", "numeric", (1e-3, 0.3))

    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter("length_scale", "numeric", self.ls_bounds, len(self.length_scale))

    def __call__(self, X, Y=None, eval_gradient=False):
        ls = np.asarray(self.length_scale, float)
        w = float(self.global_weight)
        parts = [Matern(ls, length_scale_bounds=self.ls_bounds, nu=self.nu)(
            X[:, blk], None if Y is None else Y[:, blk], eval_gradient=eval_gradient) for blk in self.blocks]
        full_kernel = Matern(np.tile(ls, len(self.blocks)), length_scale_bounds=self.ls_bounds, nu=self.nu)
        full = full_kernel(X[:, self.full], None if Y is None else Y[:, self.full], eval_gradient=eval_gradient)

        if not eval_gradient:
            return sum(parts) * (1 - w) / len(parts) + w * full

        matrix = sum(p[0] for p in parts) * (1 - w) / len(parts) + w * full[0]
        grad_ls = sum(p[1] for p in parts) * (1 - w) / len(parts)
        grad_ls += w * full[1].reshape(len(X), len(X), len(self.blocks), -1).sum(axis=2)
        if self.learn_weight:
            names = [h.name for h in self.hyperparameters]
            assert names == ["global_weight", "length_scale"], names
            grad_w = (w * (full[0] - sum(p[0] for p in parts) / len(parts)))[:, :, None]
            grad = np.concatenate([grad_w, grad_ls], axis=2)
        else:
            grad = grad_ls
        return matrix, grad

    def diag(self, X):
        return np.ones(len(X))

    def is_stationary(self):
        return True

    def __repr__(self):
        return f"BlockARD(w={self.global_weight}, ls_bounds={self.ls_bounds}, learn_w={self.learn_weight})"


def build_kernel(config, n_features=21, n_blocks=3):
    per = n_features // n_blocks
    blocks = [np.arange(per * b, per * b + per) for b in range(n_blocks)]
    full = np.arange(n_features)
    ls = config.get("length_scale_init", (0.5,) * per)
    kernel = BlockARDKernel(blocks=blocks, full=full, length_scale=ls,
                            global_weight=config.get("global_weight", 0.1),
                            ls_bounds=config.get("bounds", (0.03, 3.0)),
                            learn_weight=config.get("learn_weight", False))
    white = WhiteKernel(config.get("noise", 0.03), noise_level_bounds=config.get("noise_bounds", (0.001, 0.3)))
    return kernel + white, per


# ─────────────────────────── 优化器（对数空间多起点） ───────────────────────────
def gaussian_log_prior(indices, mean, sd):
    idx = np.asarray(indices, int)
    mean, sd = float(mean), float(sd)

    def prior(theta):
        val = float(-0.5 * np.sum(((theta[idx] - mean) / sd) ** 2))
        grad = np.zeros_like(theta)
        grad[idx] = -(theta[idx] - mean) / (sd ** 2)
        return val, grad
    return prior


def make_optimizer(maxiter=100, n_starts=4, seed=SEED, prior=None):
    def optimizer(obj_func, initial_theta, bounds):
        bounds = np.asarray(bounds, float)
        lo, hi = bounds[:, 0], bounds[:, 1]
        ok = np.isfinite(lo) & np.isfinite(hi)
        rng = np.random.default_rng(seed)
        optimizer.seed = int(seed)
        starts = [np.asarray(initial_theta, float)]
        for _ in range(max(n_starts - 1, 0)):
            s = lo.copy()
            span = hi - lo
            s[ok] = rng.uniform(lo[ok] + 0.02 * span[ok], hi[ok] - 0.02 * span[ok])
            starts.append(s)
        optimizer.starts = [np.asarray(s, float).copy() for s in starts]

        def obj(theta):
            f, g = obj_func(theta)
            if prior is not None:
                pv, pg = prior(theta)
                f, g = f - pv, g - pg
            return f, g

        trials, best = [], None
        for start in starts:
            res = minimize(obj, start, method="L-BFGS-B", jac=True, bounds=bounds,
                           options={"maxiter": maxiter, "ftol": 1e-8, "gtol": 1e-6, "maxls": 20})
            rec = {"theta_log": np.asarray(res.x, float), "fun": float(res.fun),
                   "nit": int(res.nit), "status": int(res.status), "message": str(res.message)}
            trials.append(rec)
            if best is None or rec["fun"] < best["fun"]:
                best = rec
        optimizer.record.extend(trials)
        return best["theta_log"], best["fun"]

    optimizer.record = []
    optimizer.starts = []
    optimizer.seed = int(seed)
    return optimizer


def fit_gp(X, y, kernel, optimizer=None, seed=SEED):
    loc, scale = float(np.mean(y)), float(np.std(y))
    gp = GaussianProcessRegressor(kernel=kernel, optimizer=optimizer, normalize_y=False,
                                  alpha=1e-8, random_state=seed)
    gp.fit(X, (np.asarray(y, float) - loc) / scale)
    return gp, loc, scale


def fit_gp_weighted_opt(X, y, config, sample_weight, alpha=1e-8, maxiter=80,
                        n_starts=3, seed=SEED, noise_bounds=(0.001, 0.3)):
    """加权对数边际似然：似然方差 = (noise+alpha)/w_i，等价于按 w_i 加权最小二乘。

    theta = [log(length_scale) × 7, log(noise)]；只对 block 核优化，噪声作为独立参数。
    """
    from scipy.linalg import cho_solve, cholesky

    X = np.asarray(X, float)
    y = np.asarray(y, float)
    w = np.asarray(sample_weight, float)
    w = w / np.mean(w)

    kernel, per = build_kernel(config, n_features=X.shape[1])
    block = kernel.k1
    loc = float(np.average(y, weights=w))
    scale = float(np.sqrt(np.average((y - loc) ** 2, weights=w)))
    z = (y - loc) / scale
    lo_ls, hi_ls = config.get("bounds", (0.03, 6.0))
    theta0 = np.r_[np.log(np.full(per, 0.5)), np.log(config.get("noise", 0.03))]
    lb = np.r_[np.log(np.full(per, lo_ls)), np.log(noise_bounds[0])]
    ub = np.r_[np.log(np.full(per, hi_ls)), np.log(noise_bounds[1])]

    def nll(theta):
        block.length_scale = np.exp(theta[:-1])
        Ksig, dKsig = block(X, eval_gradient=True)
        noise = float(np.exp(theta[-1]))
        K = np.array(Ksig, float)
        K[np.diag_indices_from(K)] += (noise + alpha) / w
        try:
            L = cholesky(K, lower=True)
            a = cho_solve((L, True), z)
        except np.linalg.LinAlgError:
            return 1e25, np.zeros_like(theta)
        val = 0.5 * float(z @ a) + float(np.sum(np.log(np.diag(L)))) + 0.5 * len(z) * np.log(2 * np.pi)
        if not np.isfinite(val):
            return 1e25, np.zeros_like(theta)
        A = cho_solve((L, True), np.eye(len(z))) - np.outer(a, a)
        grad = np.empty_like(theta)
        grad[:-1] = 0.5 * np.einsum("ij,ijd->d", A, dKsig)
        grad[-1] = 0.5 * float(np.sum(np.diag(A) * (noise / w)))
        return val, grad

    rng = np.random.default_rng(seed)
    starts = [np.asarray(theta0, float)]
    for _ in range(max(n_starts - 1, 0)):
        starts.append(rng.uniform(lb + 0.02 * (ub - lb), ub - 0.02 * (ub - lb)))
    trials, best = [], None
    for start in starts:
        res = minimize(nll, start, jac=True, method="L-BFGS-B", bounds=list(zip(lb, ub)),
                       options={"maxiter": maxiter, "ftol": 1e-8, "gtol": 1e-6, "maxls": 20})
        rec = {"theta_log": np.asarray(res.x, float), "fun": float(res.fun),
               "nit": int(res.nit), "status": int(res.status), "message": str(res.message)}
        trials.append(rec)
        if best is None or rec["fun"] < best["fun"]:
            best = rec

    theta = np.asarray(best["theta_log"], float)
    block.length_scale = np.exp(theta[:-1])
    noise = float(np.exp(theta[-1]))
    K = np.array(block(X), float)
    K[np.diag_indices_from(K)] += (noise + alpha) / w
    L = cholesky(K, lower=True)
    a = cho_solve((L, True), z)
    return dict(block=block, X=X, L=L, a=a, loc=loc, scale=scale, noise=noise, w=w,
                theta=theta, nll=float(best["fun"]), trials=trials, seed=int(seed),
                start_seeds=[np.asarray(s, float).tolist() for s in starts])


def tail_weights(y, cfg):
    """低值加权：ramping <= 132.80 的样本权重为 k（tail5opt 用 k=5）。"""
    cutoff = float(cfg.get("weight_cutoff", 132.80))
    k = float(cfg.get("weight_k", 1.0))
    return np.where(np.asarray(y, float) <= cutoff, k, 1.0)


# ─────────────────────────── 载荷（纯 numpy 字典） ───────────────────────────
def payload_gp(gp, loc, scale, info_theta):
    k = gp.kernel_
    return dict(schema=M.SCHEMA, kind="gp", target="raw",
                feature_order=list(M.FEATURES), n_features=M.N_FEATURES,
                X_train=np.asarray(gp.X_train_, float), alpha=np.asarray(gp.alpha_, float),
                length_scale=np.asarray(k.k1.length_scale, float),
                global_weight=float(k.k1.global_weight), noise=float(k.k2.noise_level), jitter=1e-8,
                weights=np.ones(len(gp.X_train_)), loc=float(loc), scale=float(scale),
                train_rows=int(len(gp.X_train_)))


def payload_weighted(model):
    return dict(schema=M.SCHEMA, kind="gp", target="raw",
                feature_order=list(M.FEATURES), n_features=M.N_FEATURES,
                X_train=np.asarray(model["X"], float), alpha=np.asarray(model["a"], float),
                length_scale=np.asarray(model["block"].length_scale, float),
                global_weight=float(model["block"].global_weight),
                noise=float(model["noise"]), jitter=1e-8,
                weights=np.asarray(model["w"], float), loc=float(model["loc"]),
                scale=float(model["scale"]), train_rows=int(len(model["X"])))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


ARRAY_FIELDS = ("X_train", "alpha", "length_scale", "weights")


def save_payload(payload, out_dir):
    """版本无关落盘：数组 → model.npz，其余（超参/元数据/树路径）→ params.json。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if payload["kind"] == "blend":
        gp = payload["gp"]
        meta = {k: v for k, v in payload.items() if k != "gp"}
        meta["gp"] = {k: v for k, v in gp.items() if k not in ARRAY_FIELDS}
        arrays = {f"gp_{k}": np.asarray(gp[k], float) for k in ARRAY_FIELDS}
    else:
        meta = {k: v for k, v in payload.items() if k not in ARRAY_FIELDS}
        arrays = {k: np.asarray(payload[k], float) for k in ARRAY_FIELDS}
    np.savez_compressed(out_dir / "model.npz", **arrays)
    (out_dir / "params.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def probe_seed(seed, n_starts=4):
    """用 hp_ms 配置真实触发一次多起点生成（参数空间与训练完全一致）。

    起点直接取自 `build_kernel(...)` 返回核的 `theta` / `bounds`（sklearn 的 log 空间），
    不再手写 raw 边界；同时输出 log 向量与 exp 还原后的原始值（长度尺度、噪声），
    并自检每个还原值都落在真实边界内。
    """
    kernel, _ = build_kernel(CONFIGS["blend_xgb_w20x3"], n_features=M.N_FEATURES)
    theta = np.asarray(kernel.theta, float)          # log(length_scale ×7, noise)
    bounds = np.asarray(kernel.bounds, float)        # 同维度的 log 边界
    opt = make_optimizer(maxiter=1, n_starts=n_starts, seed=seed)
    opt(lambda t: (0.0, np.zeros_like(t)), theta, bounds)
    starts_log = np.asarray(opt.starts, float)
    starts_raw = np.exp(starts_log)
    lo, hi = np.exp(bounds[:, 0]), np.exp(bounds[:, 1])
    within = np.all((starts_raw >= lo) & (starts_raw <= hi), axis=1)
    return {"seed_argument": int(seed), "seed_effective": int(opt.seed),
            "n_starts": int(n_starts),
            "parameter_space": "log(theta) = [log(length_scale) ×7, log(noise)]",
            "bounds_log": bounds.tolist(),
            "bounds_raw": {"length_scale": [float(lo[0]), float(hi[0])],
                           "noise": [float(lo[-1]), float(hi[-1])]},
            "starts_log": starts_log.tolist(),
            "starts_raw": [{"length_scale": np.round(s[:-1], 6).tolist(),
                            "noise": float(round(s[-1], 6))} for s in starts_raw],
            "all_starts_within_bounds": bool(within.all())}


# ─────────────────────────── 训练入口 ───────────────────────────
def train(name, X, y, tree_path, seed=SEED):
    t0 = time.perf_counter()
    cfg = CONFIGS[name]
    if name in ("gp-ard", "hp_wlearn"):
        kernel, _ = build_kernel(cfg, n_features=X.shape[1])
        opt = make_optimizer(maxiter=cfg.get("maxiter", 100), n_starts=cfg.get("n_starts", 4), seed=seed)
        gp, loc, scale = fit_gp(X, y, kernel, optimizer=opt, seed=seed)
        payload = payload_gp(gp, loc, scale, None)
        info = {"method": "sklearn GPR + L-BFGS-B 多起点",
                "seed_used": int(opt.seed), "n_starts": int(cfg.get("n_starts", 4)),
                "log_marginal_likelihood": float(gp.log_marginal_likelihood_value_),
                "theta_log": np.asarray(gp.kernel_.theta, float).tolist(),
                "trials": [{"theta_log": np.asarray(t["theta_log"], float).tolist(), "fun": float(t["fun"]),
                            "nit": int(t["nit"]), "status": int(t["status"]), "message": str(t["message"])}
                           for t in opt.record]}
    elif name == "tail5opt":
        sw = tail_weights(y, cfg)
        model = fit_gp_weighted_opt(X, y, cfg, sw, maxiter=cfg.get("maxiter", 80),
                                    n_starts=cfg.get("n_starts", 3), seed=seed,
                                    noise_bounds=cfg.get("noise_bounds", (0.001, 0.3)))
        payload = payload_weighted(model)
        info = {"method": "加权对数边际似然重优化",
                "seed_used": int(model["seed"]), "n_starts": int(cfg.get("n_starts", 3)),
                "weighted_neg_log_marginal_likelihood": float(model["nll"]),
                "theta_log": np.asarray(model["theta"], float).tolist(),
                "n_weighted": int(np.sum(sw > 1)), "weight_sum": float(np.sum(sw)),
                "trials": [{"theta_log": np.asarray(t["theta_log"], float).tolist(), "fun": float(t["fun"]),
                            "nit": int(t["nit"]), "status": int(t["status"]), "message": str(t["message"])}
                           for t in model["trials"]]}
    elif name == "blend_xgb_w20x3":
        kernel, _ = build_kernel(cfg, n_features=X.shape[1])
        opt = make_optimizer(maxiter=cfg.get("maxiter", 100), n_starts=cfg.get("n_starts", 4), seed=seed)
        gp, loc, scale = fit_gp(X, y, kernel, optimizer=opt, seed=seed)
        base = payload_gp(gp, loc, scale, None)
        if not Path(tree_path).exists():
            raise FileNotFoundError(f"混合模型需要树模型文件: {tree_path}")
        payload = dict(schema=M.SCHEMA, kind="blend", alpha_mix=float(BLEND_ALPHA), gp=base,
                       tree={"name": BLEND_TREE, "path": f"models/blend_xgb_w20x3/{BLEND_TREE}.joblib",
                             "sha256": sha256(Path(tree_path))},
                       feature_order=list(M.FEATURES), n_features=M.N_FEATURES, train_rows=int(len(y)))
        info = {"method": "固定权重混合 p=(1-α)·gp(hp_ms)+α·tree", "alpha": float(BLEND_ALPHA),
                "seed_used": int(opt.seed), "n_starts": int(cfg.get("n_starts", 4)),
                "tree": BLEND_TREE, "theta_log": np.asarray(gp.kernel_.theta, float).tolist(),
                "gp_hyperparameters": cfg}
    else:
        raise ValueError(name)
    info["seconds"] = float(time.perf_counter() - t0)
    return payload, info


def main():
    ap = argparse.ArgumentParser(description="四个代理模型的拟合源代码")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--model", choices=list(CONFIGS) + ["all"])
    ap.add_argument("--data", default=str(HERE.parent / "dataset_same8" / "train.csv"))
    ap.add_argument("--out", default=str(HERE / "models"))
    ap.add_argument("--tree", default=str(HERE / "models" / "blend_xgb_w20x3" / f"{BLEND_TREE}.joblib"))
    ap.add_argument("--seed", type=int, default=SEED, help="多起点与 GPR 的随机种子（默认 2026）")
    ap.add_argument("--probe-seed", type=int, default=None,
                    help="只演示：打印该种子实际生成的多起点向量，不做训练")
    a = ap.parse_args()

    if a.list:
        print(json.dumps({k: v for k, v in CONFIGS.items()}, ensure_ascii=False, indent=2, default=str))
        return
    if a.probe_seed is not None:
        print(json.dumps(probe_seed(a.probe_seed), ensure_ascii=False, indent=2, default=str))
        return
    if not a.model:
        ap.error("需要 --model 或 --list")

    df = pd.read_csv(a.data, encoding="utf-8-sig").sort_values("candidate_id").reset_index(drop=True)
    X = df[list(M.FEATURES)].to_numpy(float)
    y = df[TARGET].to_numpy(float)
    print(f"train rows={len(y)} features={X.shape[1]} target={TARGET}")

    names = list(CONFIGS) if a.model == "all" else [a.model]
    for name in names:
        payload, info = train(name, X, y, a.tree, seed=a.seed)
        out = Path(a.out) / name
        save_payload(payload, out)
        bundle = M.load_model(out)                     # 回读确认可加载
        mean = bundle.predict(X, return_std=False)
        rmse = float(np.sqrt(np.mean((mean - y) ** 2)))
        (out / "fit_report.json").write_text(json.dumps(
            {"model": name, "config": {k: v for k, v in CONFIGS[name].items()},
             **info, "in_sample_rmse_not_generalization": rmse}, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")
        print(f"[{name}] OK  seed={info.get('seed_used')}  in-sample RMSE={rmse:.4f}"
              f"（仅插值，不是泛化）  {info['seconds']:.0f}s -> {out}")


if __name__ == "__main__":
    main()
