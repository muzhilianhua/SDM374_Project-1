# -*- coding: utf-8 -*-
"""SDM374 备用代理模型包推理内核（只依赖 numpy + scipy；混合模型额外依赖 joblib）。

为什么要自己实现核函数：训练时用的是 scikit-learn 1.6.1 的自定义 Kernel 子类，
如果把 sklearn 估计器直接 pickle，加载端必须能 import 训练时的包路径与 sklearn 版本，
跨环境很容易失败。这里把预测所需的全部状态（训练集、alpha 向量、长度尺度、噪声、权重）
压成一个"纯 numpy 数组字典"再 pickle，加载端不 import sklearn，因此稳定。

本文件与 `外推方案/模型库/modellib.py` 的数学内核完全相同，仅把路径常量改为**包内相对路径**：
    PKG_ROOT = 本文件所在目录
    models/<模型名>/model.pkl
    models/blend_xgb_w20x3/xgb_w20x3_full.joblib
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

# ─────────────────────────────── 硬契约：特征顺序 ───────────────────────────────
METRICS = ("cost", "carbon", "peak", "discomfort", "degradation", "ramping", "resilience")
FEATURES = tuple(f"b{b}_{m}" for b in (1, 2, 3) for m in METRICS)
N_FEATURES = 21
N_BUILDINGS = 3
N_METRICS = 7

# 中文列名 → 规范指标名（只用于校验，真正排序仍按规范顺序）
CN_METRIC = {"电费": "cost", "碳排": "carbon", "峰值": "peak", "舒适": "discomfort",
             "损耗": "degradation", "波动": "ramping", "韧性": "resilience"}
CN_BUILDING = {"第 1 栋": 1, "第1栋": 1, "第 2 栋": 2, "第2栋": 2, "第 3 栋": 3, "第3栋": 3}

SCHEMA = "sdm374.model_library.v1"
PKG_ROOT = Path(__file__).resolve().parent          # 交付包根目录
PROJECT = PKG_ROOT.parent                           # 仓库根（仅作数据/树模型的只读回退）


class ContractError(ValueError):
    """输入权重不满足 3×7 / 21 维硬契约。"""


def _from_dataframe(df):
    """把带表头/索引的 DataFrame 按**列名/行名**映射到规范顺序，而不是按位置。"""
    cols = [str(c).strip() for c in df.columns]
    idx = [str(i).strip() for i in df.index]
    if all(c in CN_METRIC for c in cols):
        metric_order = [CN_METRIC[c] for c in cols]
        if tuple(metric_order) != METRICS:
            raise ContractError(f"列顺序/列名不符：得到 {cols}，要求 cost→…→resilience")
        if len(idx) == 3 and all(i in CN_BUILDING for i in idx):
            df = df.loc[[i for i in idx if CN_BUILDING[i] == 1]
                       + [i for i in idx if CN_BUILDING[i] == 2]
                       + [i for i in idx if CN_BUILDING[i] == 3]]
        elif len(idx) != 3:
            raise ContractError(f"需要 3 行建筑，得到 {len(idx)} 行")
        arr = df.to_numpy(float)
    else:
        arr = df.to_numpy(float)
    return arr


def pack_weights(obj):
    """把 3×7 权重（或多组）或 21 维向量整理成 (n, 21) 的规范顺序矩阵。

    规范顺序：建筑1→建筑3，每栋 cost→carbon→peak→discomfort→degradation→ramping→resilience。
    接受：pandas.DataFrame（按列名映射）、嵌套 list、np.ndarray (3,7)/(21,)/(1,21)/(n,21)。
    其他形状一律抛 ContractError —— 宁可报错，也不静默算错。
    """
    import pandas as pd

    if isinstance(obj, pd.DataFrame):
        arr = _from_dataframe(obj)
    else:
        arr = np.asarray(obj, float)

    if arr.ndim == 1:
        if arr.size != N_FEATURES:
            raise ContractError(f"21 维向量长度应为 21，得到 {arr.size}")
        arr = arr.reshape(1, N_FEATURES)
    elif arr.ndim == 2:
        if arr.shape == (N_BUILDINGS, N_METRICS):
            arr = arr.reshape(1, N_FEATURES)
        elif arr.shape[1] == N_FEATURES:
            pass
        elif arr.shape == (N_METRICS, N_BUILDINGS):
            raise ContractError("输入看起来是 7×3（指标×建筑），本库只接受 3×7（建筑×指标）")
        else:
            raise ContractError(f"二维输入应为 (3,7) 或 (n,21)，得到 {arr.shape}")
    else:
        raise ContractError(f"不支持的维度 {arr.ndim}")

    if not np.all(np.isfinite(arr)):
        raise ContractError("权重中存在 NaN / inf")
    return np.ascontiguousarray(arr, dtype=np.float64)


# ─────────────────────────────── Matérn-5/2（sklearn 口径） ───────────────────────────────
def _scale_sqdist(X, Y, ls):
    A = X / ls
    B = Y / ls
    d2 = np.sum(A * A, axis=1)[:, None] + np.sum(B * B, axis=1)[None, :] - 2.0 * (A @ B.T)
    np.maximum(d2, 0.0, out=d2)
    return d2


def matern52(X, Y, ls):
    """与 sklearn.gaussian_process.kernels.Matern(nu=2.5, ARD) 相同的协方差。"""
    d2 = _scale_sqdist(X, Y, np.asarray(ls, float))
    r = np.sqrt(5.0 * d2)
    return (1.0 + r + (5.0 / 3.0) * d2) * np.exp(-r)


def block_ard_cov(X, Y, length_scale, global_weight):
    """分块 ARD Matérn-5/2 + 全区域交互，与训练时的 BlockARDKernel 等价。"""
    ls = np.asarray(length_scale, float)
    w = float(global_weight)
    parts = None
    for b in range(N_BUILDINGS):
        k = matern52(X[:, 7 * b:7 * b + 7], Y[:, 7 * b:7 * b + 7], ls)
        parts = k if parts is None else parts + k
    K = parts * ((1.0 - w) / N_BUILDINGS)
    K += w * matern52(X, Y, np.tile(ls, N_BUILDINGS))
    return K


class ModelBundle:
    """加载后的模型：只依赖 numpy（混合模型额外依赖 joblib 读取包内树模型）。"""

    def __init__(self, payload, path=None):
        self.payload = payload
        self.path = Path(path) if path else None
        self._L = None
        self._tree = None
        self._validate()

    def _validate(self):
        p = self.payload
        if p.get("schema") != SCHEMA:
            raise ContractError(f"未知模型 schema: {p.get('schema')}")
        if tuple(p["feature_order"]) != FEATURES:
            raise ContractError("模型的 feature_order 与加载端硬契约不一致，拒绝加载")
        if p["kind"] not in ("gp", "blend"):
            raise ContractError(f"未知 kind: {p['kind']}")
        gps = [p["gp"]] if p["kind"] == "blend" else [p]
        for g in gps:
            if g["X_train"].shape[1] != N_FEATURES:
                raise ContractError("X_train 特征维度不是 21")
            if len(g["alpha"]) != len(g["X_train"]):
                raise ContractError("alpha 与训练集行数不一致")
            if len(g["length_scale"]) != 7:
                raise ContractError("length_scale 不是 7 维")
            if len(g["weights"]) != len(g["X_train"]):
                raise ContractError("样本权重与训练集行数不一致")

    # ── 高斯过程部分 ──
    def _gp_mean(self, g, Xq):
        Ks = block_ard_cov(g["X_train"], Xq,
                           g["length_scale"], g["global_weight"])   # (n, m)
        mz = g["loc"] + g["scale"] * (Ks.T @ g["alpha"])
        if g["target"] == "log":
            mean = np.exp(np.clip(mz, -20.0, 20.0))
            return mean, Ks, mz
        return mz, Ks, mz

    def _gp_chol(self, g):
        if self._L is None:
            K = block_ard_cov(g["X_train"], g["X_train"],
                              g["length_scale"], g["global_weight"])
            K[np.diag_indices_from(K)] += (g["noise"] + g["jitter"]) / g["weights"]
            self._L = np.linalg.cholesky(K)
        return self._L

    def _gp_std(self, g, Xq, Ks, mz):
        from scipy.linalg import solve_triangular
        L = self._gp_chol(g)
        Z = solve_triangular(L, Ks, lower=True)
        v = 1.0 - np.sum(Z * Z, axis=0)
        sd_z = g["scale"] * np.sqrt(np.maximum(v + g["noise"], 1e-7))
        if g["target"] == "log":
            return np.exp(np.clip(mz, -20.0, 20.0)) * sd_z
        return sd_z

    # ── 混合部分 ──
    def _tree_model(self):
        if self._tree is None:
            import joblib
            info = self.payload["tree"]
            raw = Path(info["path"])
            name = self.path.parent.name if self.path is not None else "blend_xgb_w20x3"
            candidates = [
                PKG_ROOT / raw,                                    # 包内显式相对路径
                PKG_ROOT / "models" / name / raw.name,             # 包内同目录
                PROJECT / raw,                                     # 仓库根回退（只读）
            ]
            for path in candidates:
                if path.exists():
                    self._tree = (joblib.load(path), info)
                    break
            else:
                raise FileNotFoundError(
                    "找不到混合模型所需的树模型，已尝试: " + "; ".join(str(p) for p in candidates))
        return self._tree

    def predict(self, Xq, return_std=True):
        Xq = np.asarray(Xq, float)
        if Xq.ndim == 1:
            Xq = Xq.reshape(1, -1)
        if Xq.shape[1] != N_FEATURES:
            raise ContractError(f"查询特征应为 21 维，得到 {Xq.shape}")

        p = self.payload
        if p["kind"] == "blend":
            g = p["gp"]
            mean_gp, Ks, mz = self._gp_mean(g, Xq)
            est, info = self._tree_model()
            mean_tree = np.asarray(est.predict(Xq), float)
            a = float(p["alpha_mix"])
            mean = (1.0 - a) * mean_gp + a * mean_tree
            if not return_std:
                return mean
            sd = (1.0 - a) * self._gp_std(g, Xq, Ks, mz)   # 近似，非校准区间；见说明文档风险节
            return mean, sd

        g = p
        mean, Ks, mz = self._gp_mean(g, Xq)
        if not return_std:
            return mean
        return mean, self._gp_std(g, Xq, Ks, mz)


def load_model(model_dir):
    """加载 `<模型库>/<模型名>/model.pkl`。"""
    model_dir = Path(model_dir)
    pkl = model_dir / "model.pkl"
    if not pkl.exists():
        raise FileNotFoundError(pkl)
    with open(pkl, "rb") as f:
        payload = pickle.load(f)
    return ModelBundle(payload, path=pkl)


def load_registry(lib_root=None):
    lib_root = Path(lib_root) if lib_root else PKG_ROOT / "models"
    import pandas as pd
    return pd.read_csv(lib_root / "registry.csv", encoding="utf-8-sig")


def describe():
    return {"feature_order": list(FEATURES), "n_features": N_FEATURES, "schema": SCHEMA,
            "package_root": str(PKG_ROOT)}
