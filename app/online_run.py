"""Read-only T08 run-folder snapshots for the T11 console."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd

from online_log import MAX_CALLS, read_online_log, summarize_checks
from sdm.simplex import check_weights


def discover_online_runs(root: Path) -> list[Path]:
    """Include empty newly-created online runs, direct run folders and legacy root logs."""
    def is_online(path):
        if (path / "online_log.csv").exists() or (path / "attempts.jsonl").exists():
            return True
        try:
            return json.loads((path / "meta.json").read_text(encoding="utf-8"))["kind"] == "online"
        except (OSError, ValueError, KeyError, TypeError):
            return False
    paths = ([root] if is_online(root) else [])
    if root.is_dir():
        paths += [path for path in root.iterdir() if path.is_dir() and is_online(path)]
    return sorted(paths, key=lambda path: path.name, reverse=True)


def read_attempts(path: Path) -> dict | None:
    """Validate journal sequence. Pending starts count; only local rejections release quota."""
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        raise ValueError("额度台账末行不完整，已用额度未知，请等待写入完成。")
    starts, outcomes = {}, {}
    latest = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError("额度台账记录必须是对象。")
        attempt, call, event = (record.get(key) for key in ("attempt", "call", "event"))
        if any(type(value) is not int or value < 1 for value in (attempt, call)):
            raise ValueError("额度台账的 attempt/call 必须是正整数。")
        if event == "started":
            if attempt <= latest:
                raise ValueError("额度台账 attempt 必须唯一且递增。")
            starts[attempt] = record
            latest = attempt
        elif event in {"ok", "failed", "lost", "rejected"}:
            if attempt not in starts or attempt in outcomes or starts[attempt]["call"] != call:
                raise ValueError("额度台账结果缺少对应开始记录，或结果重复。")
            outcomes[attempt] = event
        else:
            raise ValueError("额度台账包含未知事件。")
    return {"used": sum(outcomes.get(attempt) != "rejected" for attempt in starts),
            "pending": [attempt for attempt in starts if attempt not in outcomes],
            "failed": sum(event in {"failed", "lost"} for event in outcomes.values()),
            "ok_calls": {starts[attempt]["call"] for attempt, event in outcomes.items() if event == "ok"},
            "started_calls": {record["call"] for record in starts.values()}}


def read_call_checks(path: Path, log: pd.DataFrame | None) -> pd.DataFrame:
    """Report evidence from screening rounds; dropped candidates are distinct from batch validity."""
    records = []
    for file in sorted((path / "calls").glob("call_*.json")):
        saved = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(saved, dict):
            raise ValueError(f"{file.name} 不是有效调用记录。")
        call = saved.get("call")
        weights = np.asarray(saved.get("weights", []), dtype=float)
        rounds, rejections = saved.get("rounds"), saved.get("rejections")
        evidence = isinstance(rounds, list) and bool(rounds) and isinstance(rejections, list)
        valid = False
        if weights.size:
            weights = weights.reshape(-1, 21)
            try:
                check_weights(weights)
                valid = (len(weights) <= 50 and len(np.unique(np.round(weights, 9), axis=0)) == len(weights)
                         and evidence and sum(item["accepted"] for item in rounds) == len(weights))
            except (ValueError, KeyError, TypeError):
                valid = False
        records.append({"调用号": call, "记录": file.stem, "状态": saved.get("status", "未知"),
                        "接收点数": len(weights) if weights.size else 0,
                        "剔除候选": len(rejections) if isinstance(rejections, list) else 0,
                        "全部通过": "通过" if valid else "未知",
                        "信赖域比例": saved.get("diagnostics", {}).get("state_radius_ratio")})
    if records:
        return pd.DataFrame(records)
    return summarize_checks(log) if log is not None else pd.DataFrame()


@dataclass
class Snapshot:
    log: pd.DataFrame | None
    used: int | None
    budget: int
    journal: dict | None
    checks: pd.DataFrame
    error: str | None = None


def read_run(path: Path) -> Snapshot:
    """Do not fall back to success-only counts after a corrupt or missing T08 journal."""
    log, error = read_online_log(path / "online_log.csv")
    result = Snapshot(log, None, MAX_CALLS, None, pd.DataFrame(), error)
    if error:
        return result
    try:
        config = path / "config.toml"
        if config.exists():
            online = tomllib.loads(config.read_text(encoding="utf-8")).get("online", {})
            result.budget = online.get("calls", MAX_CALLS)
            if type(result.budget) is not int or result.budget < 1:
                raise ValueError("配置中的调用额度无效。")
        result.journal = read_attempts(path / "attempts.jsonl")
        if result.journal is not None:
            result.used = result.journal["used"]
            completed = set(log["_call"].unique()) if log is not None else set()
            if (not completed.issubset(result.journal["started_calls"])
                    or not result.journal["ok_calls"].issubset(completed)):
                raise ValueError("额度台账与实测日志暂不一致，请等待刷新或核对恢复状态。")
        elif log is not None:
            # Legacy logs carry no failed attempts. Keep their measured-call count visible with an
            # explicit warning in the view; a T08 run missing its journal has unknown quota.
            meta = path / "meta.json"
            if meta.exists() and json.loads(meta.read_text(encoding="utf-8")).get("kind") == "online":
                raise ValueError("在线运行缺少 attempts.jsonl，无法确认失败或丢失调用的额度。")
            result.used = int(log["_call"].max())
        result.checks = read_call_checks(path, log)
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        result.used = None
        result.error = f"无法可靠读取在线运行：{error}"
    return result
