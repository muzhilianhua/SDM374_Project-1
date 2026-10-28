"""Read-only tabular views of recorded runs, for the GUI and other reporting. Never writes to runs/."""
from __future__ import annotations

from pathlib import Path
import tomllib

import pandas as pd

from .runs import list_runs

# W* farther than this multiple of the table's typical nearest-neighbour spacing is an extrapolation.
EXTRAPOLATION_RATIO = 1.5

RUN_COLUMNS = ["run", "kind", "name", "created", "model", "optimizer", "protocol", "model_run",
               "commit", "dirty", "complete", "metrics", "path"]


def run_records(root: Path | None = None) -> pd.DataFrame:
    """One row per run, oldest first; `metrics` holds the run's metrics.json dict (empty if incomplete)."""
    rows = []
    for run in list_runs(root):
        meta = run.read_json("meta.json")
        config = tomllib.loads(run.config_path.read_text(encoding="utf-8"))
        is_optimize = meta["kind"] == "optimize"
        rows.append({
            "run": run.id,
            "kind": meta["kind"],
            "name": meta["name"],
            "created": pd.Timestamp(meta["created"]),
            "model": config.get("model", {}).get("name"),
            "optimizer": config.get("optimizer", {}).get("name") if is_optimize else None,
            "protocol": config.get("evaluate", {}).get("protocol", "kfold") if meta["kind"] == "evaluate" else None,
            "model_run": meta["options"].get("model_run"),
            "commit": (meta["git"].get("commit") or "")[:7],
            "dirty": meta["git"].get("dirty"),
            "complete": run.has("metrics.json"),
            "metrics": run.read_json("metrics.json") if run.has("metrics.json") else {},
            "path": str(run.path),
        })
    return pd.DataFrame(rows, columns=RUN_COLUMNS)


def _with_metrics(records: pd.DataFrame, kind: str) -> pd.DataFrame:
    selected = records[(records.kind == kind) & records.complete]
    metrics = pd.DataFrame(list(selected.metrics), index=selected.index)
    return pd.concat([selected.drop(columns=["metrics"]), metrics], axis=1)


def leaderboard(records: pd.DataFrame) -> pd.DataFrame:
    """Completed evaluate runs: frozen-v32 runs by v3.2 score (higher is better), then k-fold runs by MAE.

    The two protocols use different splits, so their rows are ranked separately, never against each other.
    """
    table = _with_metrics(records, "evaluate")
    if table.empty:
        return table
    v32 = table[table.protocol == "frozen-v32"]
    if not v32.empty:
        v32 = v32.sort_values("v32_score", ascending=False, kind="stable")
    kfold = table[table.protocol != "frozen-v32"].sort_values("mae", kind="stable")
    return pd.concat([v32, kfold]).reset_index(drop=True)


def candidates(records: pd.DataFrame) -> pd.DataFrame:
    """Completed optimize runs ranked by predicted ramping, with an extrapolation flag."""
    table = _with_metrics(records, "optimize")
    if table.empty:
        return table
    table["distance_ratio"] = table.nearest_train_distance / table.train_median_nn_distance
    table["extrapolating"] = table.distance_ratio > EXTRAPOLATION_RATIO
    return table.sort_values("predicted_mean", kind="stable").reset_index(drop=True)
