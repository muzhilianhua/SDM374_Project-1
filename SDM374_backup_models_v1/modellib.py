# -*- coding: utf-8 -*-
"""SDM374 备用代理模型包推理内核（只依赖 numpy + scipy；混合模型额外依赖 joblib + xgboost）。

为什么要自己实现核函数：训练时用的是 scikit-learn 1.6.1 的自定义 Kernel 子类，
如果把 sklearn 估计器直接 pickle，加载端必须能 import 训练时的包路径与 sklearn 版本，
跨环境很容易失败。这里把预测所需的全部状态（训练集、alpha 向量、长度尺度、噪声、权重）
拆成"纯 numpy 数组 + JSON 元数据"：数组存 `model.npz`（`np.savez_compressed`，
`allow_pickle=False`），其余存 `params.json`。**不使用 pickle**，因此 numpy 1.24.x
（`numpy.core`）与 numpy 2.x（`numpy._core`）都能直接加载，加载端也不 import sklearn。

输入契约有两种，区别只在"有没有标签"：
  - 21 维向量 / (n,21) 数组：**只能按位置**，顺序写死为 FEATURES。
  - DataFrame / CSV（3×7 或 21 列宽表）：**一律按标签重排**，行列物理顺序无关；
    未知/重复/缺失标签或行数≠3 一律抛 ContractError，绝不静默按位置解释。

路径常量（全部包内相对路径）：
    PKG_ROOT = 本文件所在目录
    models/<模型名>/model.npz + params.json
    models/blend_xgb_w20x3/xgb_w20x3_full.joblib
"""
from __future__ import annotations

import json
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
CN_BUILDING.update({f"楼{b}": b for b in (1, 2, 3)})
CN_BUILDING.update({f"楼 {b}": b for b in (1, 2, 3)})
CN_BUILDING.update({f"楼栋{b}": b for b in (1, 2, 3)})
CN_BUILDING.update({f"建筑{b}": b for b in (1, 2, 3)})

# 英文列名 → 规范指标名（大小写/空白/连字符已归一）
EN_METRIC = {m: m for m in METRICS}
EN_METRIC.update({
    "electricity": "cost", "electricity_cost": "cost", "energy_cost": "cost",
    "carbon_emission": "carbon", "co2": "carbon", "carbon_emissions": "carbon",
    "peak_load": "peak", "peak_demand": "peak",
    "thermal_discomfort": "discomfort", "comfort": "discomfort",
    "degradation_cost": "degradation", "battery_degradation": "degradation",
    "ramping_cost": "ramping", "ramp": "ramping",
    "resilience_penalty": "resilience", "resiliency": "resilience",
})
# 英文楼栋标识 → 1/2/3（覆盖 building_1 / building1 / bldg1 / b1 等常见写法）
EN_BUILDING = {}
for _b in (1, 2, 3):
    for _name in (f"building_{_b}", f"building{_b}", f"building_{_b:02d}", f"building{_b:02d}",
                  f"bldg_{_b}", f"bldg{_b}", f"b{_b}", f"b_{_b}", f"bat_{_b}", f"battery_{_b}",
                  str(_b)):
        EN_BUILDING[_name] = _b

SCHEMA = "sdm374.model_library.v1"
PKG_ROOT = Path(__file__).resolve().parent          # 交付包根目录
PROJECT = PKG_ROOT.parent                           # 仓库根（仅作数据/树模型的只读回退）


class ContractError(ValueError):
    """输入权重不满足 3×7 / 21 维硬契约。"""


def _norm_label(x) -> str:
    """标签归一：去空白、转小写、空格/连字符/点统一成下划线，并压缩连续下划线。"""
    s = str(x).strip().lower()
    for ch in (" ", "-", "."):
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def _metric_label(x):
    key = _norm_label(x)
    for k, v in CN_METRIC.items():
        if _norm_label(k) == key:
            return v
    return EN_METRIC.get(key)


def _building_label(x):
    key = _norm_label(x)
    for k, v in CN_BUILDING.items():
        if _norm_label(k) == key:
            return v
    return EN_BUILDING.get(key)


def _from_dataframe(df):
    """把带表头/索引的 DataFrame 按**标签**映射到规范顺序，绝不按位置猜。

    规则（有标签就按标签，无标签才按位置 —— 位置路径只存在于 21 维向量/数组）：
      - 7 列 × 3 行：列必须是可识别的指标标签，行必须是可识别的建筑标签；
      - 21 列：列名必须是规范 FEATURES（顺序任意，按名重排）；
      - 其他情况、未知标签、重复标签、缺失标签、行数≠3 一律抛 ContractError。
    """
    cols = [str(c).strip() for c in df.columns]
    idx = [str(i).strip() for i in df.index]

    # 扁平 21 列（规范特征名，允许任意顺序）
    if len(cols) == N_FEATURES:
        norm = [_norm_label(c) for c in cols]
        if set(norm) == {_norm_label(f) for f in FEATURES}:
            order = [norm.index(_norm_label(f)) for f in FEATURES]
            return df.iloc[:, order].to_numpy(float)
        raise ContractError(
            f"21 列表格无法按名对齐：得到 {cols}，要求恰好是 {list(FEATURES)} 这 21 个列名")

    if len(cols) != N_METRICS:
        raise ContractError(
            f"二维表格应为 3×7（建筑×指标）或 n×21（规范特征），得到 {len(idx)}×{len(cols)}")

    metrics = [_metric_label(c) for c in cols]
    unknown = [c for c, m in zip(cols, metrics) if m is None]
    if unknown:
        raise ContractError(
            f"未知列标签 {unknown}；要求 7 个指标标签（中文 电费/碳排/峰值/舒适/损耗/波动/韧性，"
            f"或英文 {'/'.join(METRICS)} 及常见同义词）")
    dup = sorted({m for m in metrics if metrics.count(m) > 1})
    if dup:
        raise ContractError(f"指标标签重复 {dup}；每个指标只能出现一次")
    if set(metrics) != set(METRICS):
        raise ContractError(f"指标标签缺失：得到 {sorted(set(metrics))}，要求 {sorted(METRICS)}")

    if len(idx) != N_BUILDINGS:
        raise ContractError(f"需要 3 行建筑，得到 {len(idx)} 行")
    buildings = [_building_label(i) for i in idx]
    unknown_b = [i for i, b in zip(idx, buildings) if b is None]
    if unknown_b:
        raise ContractError(
            f"未知行标签 {unknown_b}；要求 3 个建筑标签（中文 第 1 栋/第1栋，"
            "或英文 building_1/b1/bldg1 等）")
    dup_b = sorted({b for b in buildings if buildings.count(b) > 1})
    if dup_b:
        raise ContractError(f"建筑标签重复（规范编号 {dup_b}）；每栋只能出现一次")
    if set(buildings) != {1, 2, 3}:
        raise ContractError(f"建筑标签缺失：得到 {buildings}，要求 1/2/3 各一行")

    row_order = [buildings.index(b) for b in (1, 2, 3)]
    col_order = [metrics.index(m) for m in METRICS]
    arr = df.iloc[row_order, col_order].to_numpy(float)
    return arr.reshape(1, N_FEATURES)


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
            name = self.path.name if self.path is not None else "blend_xgb_w20x3"
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


GP_ARRAY_KEYS = ("X_train", "alpha", "length_scale", "weights")
MODEL_ARRAYS = "model.npz"
MODEL_PARAMS = "params.json"


def _load_payload(model_dir: Path):
    """读取版本无关载荷：数组在 model.npz（np.load，allow_pickle=False），元数据在 params.json。

    不使用 pickle：numpy 1.24.x 与 2.x 的 pickle 模块路径不同（`numpy._core` vs `numpy.core`），
    存数组的 pickle 会跨版本失败。npz 是纯二进制格式，两代 numpy 都能读。
    """
    npz_path = model_dir / MODEL_ARRAYS
    params_path = model_dir / MODEL_PARAMS
    if not npz_path.exists() or not params_path.exists():
        raise FileNotFoundError(
            f"缺少 {npz_path.name} 或 {params_path.name}（在 {model_dir}）；"
            "本包不再使用 pickle 载荷，请从 train_four_models.py 重新生成")
    params = json.loads(params_path.read_text(encoding="utf-8-sig"))
    with np.load(npz_path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}

    if params.get("kind") == "blend":
        gp_meta = params.pop("gp")
        gp = dict(gp_meta)
        for k in GP_ARRAY_KEYS:
            gp[k] = arrays[f"gp_{k}"]
        params["gp"] = gp
        return params

    payload = dict(params)
    for k in GP_ARRAY_KEYS:
        payload[k] = arrays[k]
    return payload


def load_model(model_dir):
    """加载 `<模型库>/<模型名>/`（model.npz + params.json + 可选的树模型）。"""
    model_dir = Path(model_dir)
    payload = _load_payload(model_dir)
    return ModelBundle(payload, path=model_dir)


def load_registry(lib_root=None):
    lib_root = Path(lib_root) if lib_root else PKG_ROOT / "models"
    import pandas as pd
    return pd.read_csv(lib_root / "registry.csv", encoding="utf-8-sig")


def describe():
    return {"feature_order": list(FEATURES), "n_features": N_FEATURES, "schema": SCHEMA,
            "package_root": str(PKG_ROOT)}
