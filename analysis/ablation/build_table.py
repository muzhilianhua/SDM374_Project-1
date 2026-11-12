"""Build reproducible tables from ``sdm`` run records for the report.

The script reads only completed ``evaluate`` runs. It never edits ``runs/`` and
keeps the run id beside every metric so report numbers remain traceable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib

import pandas as pd

from sdm.catalog import run_records
from sdm.paths import REPO_ROOT


METRICS = ["v32_score", "v32_top1_regret", "mae", "rmse", "spearman", "top50_recall", "std_coverage"]


def build_table(runs_dir: Path, *, include_unverified: bool = False,
                project_root: Path = REPO_ROOT) -> pd.DataFrame:
    records = run_records(runs_dir)
    records = records[(records.kind == "evaluate") & records.complete].copy()
    rows = []
    for _, record in records.iterrows():
        config = tomllib.loads(Path(record.path, "config.toml").read_text(encoding="utf-8"))
        meta = json.loads(Path(record.path, "meta.json").read_text(encoding="utf-8"))
        commit = meta.get("git", {}).get("commit")
        current_config = project_root / "experiments" / (str(record['name']) + ".toml")
        config_available = current_config.is_file() and tomllib.loads(
            current_config.read_text(encoding="utf-8")) == config
        verified = bool(commit) and meta.get("git", {}).get("dirty") is False and config_available
        if not include_unverified and not verified:
            continue
        params = config.get("model", {}).get("params", {})
        variant = ", ".join(f"{key}={value}" for key, value in sorted(params.items()))
        metrics = record.metrics
        row = {
            "run": record.run,
            "created": record.created.isoformat(),
            "model": record.model,
            "model_variant": variant,
            "protocol": record.protocol,
            "commit": commit,
            "dirty": record.dirty if commit else None,
            "current_config_matches": config_available,
        }
        row.update({metric: metrics.get(metric) for metric in METRICS})
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values(["protocol", "v32_score", "mae"], ascending=[True, False, True],
                             na_position="last", kind="stable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("analysis/ablation/evidence"))
    parser.add_argument("--out", type=Path, default=Path("analysis/ablation/results"))
    parser.add_argument("--include-unverified", action="store_true",
                        help="include dirty, unknown-source or unavailable-config historical runs")
    args = parser.parse_args()
    table = build_table(args.runs_dir, include_unverified=args.include_unverified)
    if table.empty:
        raise SystemExit("No completed evaluate runs found")
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "evaluation_table.csv", index=False, float_format="%.10g", lineterminator="\n")
    columns = list(table.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in table.iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append("" if pd.isna(value) else f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    (args.out / "evaluation_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    provenance = {
        "runs_dir": str(args.runs_dir),
        "scope": "evaluation-only; optimizer and feature ablations remain pending (T24)",
        "include_unverified": args.include_unverified,
        "rows": len(table),
        "run_ids": table["run"].tolist(),
        "protocols": sorted(table.protocol.dropna().unique().tolist()),
        "dirty_runs": table.loc[table.dirty.eq(True), "run"].tolist(),
        "unknown_provenance_runs": table.loc[table.dirty.isna(), "run"].tolist(),
    }
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {len(table)} completed evaluations to {args.out}")


if __name__ == "__main__":
    main()
