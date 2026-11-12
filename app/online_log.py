"""Parsing helpers for the append-only online evaluator log."""
from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


MAX_CALLS = 20
MAX_BATCH = 50


def _normalise(name: object) -> str:
    """Make log headers comparable without changing the displayed source table."""
    return "_".join(str(name).strip().casefold().replace("-", "_").split())


def _find_column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    names = {_normalise(column): column for column in frame.columns}
    for alias in aliases:
        if _normalise(alias) in names:
            return names[_normalise(alias)]
    return None


def read_online_log(path: str | Path) -> tuple[pd.DataFrame | None, str | None]:
    """Load and validate the T07 append-only log used by the dashboard.

    T07 owns the writer, so this reader accepts aliases for the call and score columns while
    preserving the source columns. Missing or empty files mean that no observations are ready.
    Invalid or incomplete snapshots are rejected rather than dropping rows or understating quota.
    """
    path = Path(path)
    if not path.exists():
        return None, None
    try:
        snapshot = path.read_text(encoding="utf-8-sig")
        # pandas can infer an index from extra fields or rename duplicate headers. Reject these
        # snapshots before parsing so malformed CSV cannot silently change the call or R columns.
        rows = (row for row in csv.reader(StringIO(snapshot), strict=True)
                if row and (len(row) != 1 or row[0].strip()))
        header = next(rows, [])
        if any(len(row) != len(header) for row in rows):
            return None, "在线日志的字段数量与表头不一致；请等待写入完成或核对文件。"
        frame = pd.read_csv(StringIO(snapshot))
    except pd.errors.EmptyDataError:
        return None, None
    except (OSError, ValueError, csv.Error, pd.errors.ParserError) as error:
        return None, f"无法读取在线日志：{error}"
    if frame.empty:
        return None, None
    if not snapshot.endswith(("\n", "\r")):
        return None, "日志末行缺少换行符，可能尚未完整写入；请等待下一次刷新。"

    names = [_normalise(column) for column in header]
    if len(set(names)) != len(names):
        return None, "在线日志包含重复的列名，无法确定调用号和实测 R。"

    call = _find_column(frame, ("call", "call_no", "call_number", "调用号"))
    score = _find_column(frame, ("r", "actual_r", "score", "ramping", "value", "R"))
    if call is None or score is None:
        return None, "在线日志必须包含调用号（call）和实测 R 列。"
    if any(pd.api.types.is_bool_dtype(frame[column]) for column in (call, score)):
        return None, "调用号和实测 R 必须是有限数值，不能是布尔值。"
    frame = frame.copy()
    frame["_call"] = pd.to_numeric(frame[call], errors="coerce")
    frame["_r"] = pd.to_numeric(frame[score], errors="coerce")
    if frame[["_call", "_r"]].isna().any().any() or not np.isfinite(frame[["_call", "_r"]]).all().all():
        return None, "调用号和实测 R 必须是有限数值；日志可能仍在写入。"
    if ((frame["_call"] < 1).any() or (frame["_call"] % 1 != 0).any()
            or (frame["_call"] >= 2 ** 63).any()):
        return None, "调用号必须是从 1 开始的正整数。"
    frame["_call"] = frame["_call"].astype(np.int64)
    if not frame["_call"].is_monotonic_increasing:
        return None, "调用号必须按写入顺序递增，同一次调用的记录必须相邻。"
    calls = frame["_call"].unique()
    if not np.array_equal(calls, np.arange(1, len(calls) + 1)):
        return None, "调用号必须从 1 连续递增；日志存在缺失调用，无法确认已用额度。"
    oversized = frame.groupby("_call").size()
    oversized = oversized[oversized > MAX_BATCH]
    if not oversized.empty:
        return None, f"第 {int(oversized.index[0])} 次调用超过每批 {MAX_BATCH} 组权重的上限。"
    return frame, None


def check_columns(frame: pd.DataFrame) -> list[str]:
    """Return optional boolean pre-submit check columns written by the online runner."""
    columns = []
    for column in frame.columns:
        name = _normalise(column)
        if name.startswith("check_") or name.endswith("_ok"):
            columns.append(column)
    return columns


def summarize_checks(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize each call's recorded checks without treating missing values as a pass."""
    columns = check_columns(frame)
    if not columns:
        return pd.DataFrame()

    def parse(value) -> bool | None:
        if pd.isna(value):
            return None
        text = str(value).strip().casefold()
        if text in {"1", "1.0", "true", "yes", "ok", "通过"}:
            return True
        if text in {"0", "0.0", "false", "no", "fail", "failed", "失败", "未通过"}:
            return False
        return None

    summaries = []
    for call, batch in frame.groupby("_call", sort=True):
        values = {}
        for column in columns:
            parsed = [parse(value) for value in batch[column]]
            values[column] = "未通过" if False in parsed else "未知" if None in parsed else "通过"
        overall = ("未通过" if "未通过" in values.values()
                   else "未知" if "未知" in values.values() else "通过")
        summaries.append({"调用号": int(call), **values, "全部通过": overall})
    return pd.DataFrame(summaries)
