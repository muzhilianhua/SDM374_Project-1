"""Parsing helpers for the append-only online evaluator log."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


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

    T07 owns the writer, so this reader accepts a few harmless aliases for the call and score
    columns while keeping all original columns available for display. A malformed row is reported
    instead of being silently dropped, since hiding a failed evaluator call would corrupt the quota.
    """
    path = Path(path)
    if not path.exists():
        return None, None
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        return None, f"Unable to read online log: {error}"
    if frame.empty:
        return None, "Online log is empty"

    call = _find_column(frame, ("call", "call_no", "call_number", "调用号"))
    score = _find_column(frame, ("r", "actual_r", "score", "ramping", "value", "R"))
    if call is None or score is None:
        return None, "Online log must contain call and R columns"
    frame = frame.copy()
    frame["_call"] = pd.to_numeric(frame[call], errors="coerce")
    frame["_r"] = pd.to_numeric(frame[score], errors="coerce")
    if frame[["_call", "_r"]].isna().any().any() or not np.isfinite(frame[["_call", "_r"]]).all().all():
        return None, "Call and R values in online log must be finite numbers"
    if (frame["_call"] < 1).any() or (frame["_call"] % 1 != 0).any():
        return None, "Call numbers in online log must be integers starting at 1"
    frame["_call"] = frame["_call"].astype(int)
    return frame, None


def check_columns(frame: pd.DataFrame) -> list[str]:
    """Return optional boolean pre-submit check columns written by the online runner."""
    columns = []
    for column in frame.columns:
        name = _normalise(column)
        if name.startswith("check_") or name.endswith("_ok"):
            columns.append(column)
    return columns
