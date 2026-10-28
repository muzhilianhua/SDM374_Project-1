"""Streamlit helpers shared by the pages; anything not tied to Streamlit belongs in sdm.catalog."""
from __future__ import annotations

from pathlib import Path
import shlex

import numpy as np
import pandas as pd
import streamlit as st

from sdm import catalog
from sdm.data import Dataset, load_dataset
from sdm.paths import DEFAULT_RUNS_DIR, REPO_ROOT
from sdm.simplex import N_BUILDINGS

DATASET_DIR = REPO_ROOT / "dataset_same8"

# Display names for the 7 objectives, in sdm.simplex.METRICS order (course slide wording).
METRIC_LABELS = ("电费", "碳排", "峰值", "舒适", "损耗", "波动", "韧性")

# Chinese headers for table columns; metric acronyms (MAE, R², ...) stay as they are.
COLUMN_LABELS = {
    "run": "运行 ID", "kind": "类型", "name": "配置", "created": "创建时间", "model": "模型",
    "optimizer": "寻优器", "model_run": "模型来源运行", "commit": "提交", "dirty": "有未提交改动",
    "complete": "已完成", "protocol": "评估协议", "v32_score": "v3.2 总分", "v32_score_std": "v3.2 折间标准差",
    "v32_top1_regret": "第一推荐差距", "v32_local_mae": "低值区 MAE", "mae": "MAE", "rmse": "RMSE", "r2": "R²", "spearman": "Spearman",
    "mean_predicted_std": "平均预测标准差", "predicted_mean": "预测波动量", "predicted_std": "预测标准差",
    "nearest_train_distance": "最近训练点距离", "distance_ratio": "距离比", "extrapolating": "外推",
    "candidate_id": "候选编号", "ramping": "波动量 R", "actual": "实测波动量", "type": "类型",
    "default": "默认值",
}


def runs_root() -> Path:
    return Path(st.session_state.get("runs_dir") or DEFAULT_RUNS_DIR).expanduser()


def records() -> pd.DataFrame:
    # Not cached: runs appear while the app is open, and reading their JSON files is cheap.
    return catalog.run_records(runs_root())


@st.cache_data(show_spinner=False, max_entries=4)
def _load_dataset(path: str, mtime_ns: int) -> tuple[Dataset | None, str | None]:
    try:
        return load_dataset(path), None
    except (ValueError, FileNotFoundError) as error:
        return None, str(error)


def dataset() -> tuple[Dataset | None, str | None]:
    """(dataset, None) when the checksum and constraints hold, else (None, reason)."""
    csv = DATASET_DIR / "train.csv"
    if not csv.exists():
        return None, f"找不到 {csv}"
    return _load_dataset(str(DATASET_DIR), csv.stat().st_mtime_ns)


def weights_frame(matrix) -> pd.DataFrame:
    return pd.DataFrame(np.asarray(matrix), columns=METRIC_LABELS,
                        index=[f"第 {i} 栋" for i in range(1, N_BUILDINGS + 1)])


def column_config(table: pd.DataFrame) -> dict:
    """Chinese headers for known columns, plus 4-decimal formatting for float columns."""
    config = {}
    for column in table.columns:
        label = COLUMN_LABELS.get(column)
        if column.startswith("top") and column.endswith("_recall"):
            label = f"Top-{column[3:-7]} 召回率"
        if table[column].dtype.kind == "f":
            config[column] = st.column_config.NumberColumn(label, format="%.4f")
        elif label:
            config[column] = st.column_config.Column(label)
    return config


def cli_command(*args: str) -> str:
    """The `sdm` invocation for args, including --runs-dir when the sidebar points elsewhere."""
    prefix = ["uv", "run", "sdm"]
    if runs_root().resolve() != DEFAULT_RUNS_DIR.resolve():
        prefix += ["--runs-dir", str(runs_root())]
    return shlex.join([*prefix, *args])


JOB_STATUS = {  # status -> (label, badge color, icon)
    "running": ("运行中", "blue", ":material/progress_activity:"),
    "succeeded": ("成功", "green", ":material/check_circle:"),
    "failed": ("失败", "red", ":material/error:"),
    "cancelled": ("已取消", "gray", ":material/cancel:"),
}


def experiment_configs() -> list[str]:
    """Config paths relative to the repo root, as the CLI expects them; `_*.toml` templates are skipped."""
    return sorted(f"experiments/{path.name}" for path in (REPO_ROOT / "experiments").glob("*.toml")
                  if not path.name.startswith("_"))


def parse_param(text: str, type_name: str):
    """Turn a form field back into the Params type named by registry.describe (int, float, str, tuple[...])."""
    text = text.strip()
    if type_name.startswith("tuple"):
        item = float if "float" in type_name else int if "int" in type_name else str
        return [item(part.strip()) for part in text.split(",") if part.strip()]
    if type_name == "int":
        return int(text)
    if type_name == "float":
        return float(text)
    return text


def format_param(value) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(repr(item) for item in value)
    return "" if value is None else repr(value) if isinstance(value, float) else str(value)


def metric_columns(table: pd.DataFrame, preferred: list[str]) -> list[str]:
    """Preferred columns that exist, plus any top-k recall column (its name depends on top_k)."""
    recall = [column for column in table.columns if column.startswith("top") and column.endswith("_recall")]
    return [column for column in preferred if column in table.columns] + recall
