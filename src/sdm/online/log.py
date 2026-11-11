"""Append-only records for the sequential online evaluator calls (T07)."""
from __future__ import annotations

from dataclasses import dataclass
import csv
import io
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..simplex import check_weights

N_WEIGHTS = 21
MAX_POINTS_PER_CALL = 50
WEIGHT_COLUMNS = tuple(f"weight_{i:02d}" for i in range(N_WEIGHTS))
FIELDS = ("call", "point", "n", *WEIGHT_COLUMNS, "R", "source", "run_id", "timestamp")


class OnlineLogError(ValueError):
    """The on-disk log is malformed or an append would violate its contract."""


@dataclass(frozen=True)
class LogRow:
    call: int
    point: int
    n: int
    weights: tuple[float, ...]
    R: float
    source: str
    run_id: str
    timestamp: str


def _number(value: str, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise OnlineLogError(f"{name} must be numeric") from error
    if not np.isfinite(number):
        raise OnlineLogError(f"{name} must be finite")
    return number


class OnlineLog:
    """Read and append a validated online log without rewriting prior rows."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def rows(self) -> list[LogRow]:
        if not self.path.exists():
            return []
        with self.path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise OnlineLogError(f"unexpected header; expected {FIELDS!r}")
            result = []
            for index, raw in enumerate(reader, 2):
                if None in raw:
                    raise OnlineLogError(f"unexpected fields in row {index}")
                try:
                    call = int(raw["call"])
                    point = int(raw["point"])
                    batch_size = int(raw["n"])
                    weights = tuple(_number(raw[name], name) for name in WEIGHT_COLUMNS)
                    score = _number(raw["R"], "R")
                    source, run_id, timestamp = raw["source"], raw["run_id"], raw["timestamp"]
                except (KeyError, TypeError, ValueError) as error:
                    raise OnlineLogError(f"invalid row {index}") from error
                if call < 1 or point < 1 or not source or not run_id or not timestamp:
                    raise OnlineLogError(f"invalid identifiers in row {index}")
                if not 1 <= batch_size <= MAX_POINTS_PER_CALL:
                    raise OnlineLogError(f"invalid batch size in row {index}")
                try:
                    check_weights(np.asarray(weights, dtype=float))
                except ValueError as error:
                    raise OnlineLogError(f"invalid weights in row {index}") from error
                result.append(LogRow(call, point, batch_size, weights, score, source, run_id, timestamp))
        self._validate_sequence(result)
        return result

    @staticmethod
    def _validate_sequence(rows: list[LogRow]) -> None:
        if rows and (rows[0].call != 1 or rows[0].point != 1):
            raise OnlineLogError("log must start at call 1, point 1")
        seen = set()
        index = 0
        expected_call = 1
        while index < len(rows):
            first = rows[index]
            batch_size = first.n
            if first.call != expected_call or first.point != 1:
                raise OnlineLogError("call and point numbers must be contiguous")
            if index + batch_size > len(rows):
                raise OnlineLogError(f"incomplete batch for call {expected_call}")
            for offset in range(batch_size):
                row = rows[index + offset]
                if row.call != expected_call or row.point != offset + 1 or row.n != batch_size:
                    raise OnlineLogError(f"incomplete batch for call {expected_call}")
                key = row.weights
                if key in seen:
                    raise OnlineLogError("duplicate weights in online log")
                seen.add(key)
            index += batch_size
            expected_call += 1

    @property
    def next_call(self) -> int:
        rows = self.rows()
        return 1 if not rows else rows[-1].call + 1

    def append(self, weights, scores, *, source: str, run_id: str, call: int | None = None) -> list[LogRow]:
        batch = check_weights(np.asarray(weights, dtype=float))
        scores = np.asarray(scores, dtype=float).reshape(-1)
        if len(batch) == 0 or len(batch) > MAX_POINTS_PER_CALL:
            raise OnlineLogError("a call must contain 1 to 50 points")
        if len(batch) != len(scores) or not np.isfinite(scores).all():
            raise OnlineLogError("weights and finite scores must have the same length")
        if not source or not run_id:
            raise OnlineLogError("source and run_id are required")
        old = self.rows()
        expected_call = 1 if not old else old[-1].call + 1
        actual_call = expected_call if call is None else call
        if actual_call != expected_call:
            raise OnlineLogError(f"expected call {expected_call}, got {actual_call}")
        previous = {row.weights for row in old}
        new_weights = [tuple(float(value) for value in row) for row in batch]
        if len(set(new_weights)) != len(new_weights) or previous.intersection(new_weights):
            raise OnlineLogError("duplicate weights in online log")
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not self.path.exists() or self.path.stat().st_size == 0
        new_rows = [LogRow(actual_call, point, len(new_weights), row, float(score), source, run_id, timestamp)
                    for point, (row, score) in enumerate(zip(new_weights, scores), 1)]
        payload = io.StringIO()
        writer = csv.DictWriter(payload, fieldnames=FIELDS, lineterminator="\n")
        if needs_header:
            writer.writeheader()
        for row in new_rows:
            writer.writerow({"call": row.call, "point": row.point, "n": row.n,
                             **dict(zip(WEIGHT_COLUMNS, (f"{x:.17g}" for x in row.weights))),
                             "R": f"{row.R:.17g}", "source": row.source,
                             "run_id": row.run_id, "timestamp": row.timestamp})
        with self.path.open("a", newline="", encoding="utf-8") as stream:
            stream.write(payload.getvalue())
            stream.flush()
            os.fsync(stream.fileno())
        return new_rows

    def best(self) -> LogRow | None:
        rows = self.rows()
        return min(rows, key=lambda row: row.R) if rows else None
