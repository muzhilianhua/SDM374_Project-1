"""Append locally measured points to ``eval_log.jsonl`` (the measured-point database).

Only real evaluator results are accepted: every record must carry 21 finite, non-negative
weights with each building summing to 1, and a finite positive ``R``. Points already in the
log are skipped, so the script is idempotent and safe to re-run.

Usage (from this folder):
    python ingest_local_measured.py <run_dir_or_eval_log> [more...] --tag <tag>

Each argument is either a run directory containing ``eval_log.jsonl`` or the file itself.
A provenance report ``ingest_report_<UTC timestamp>.json`` is written next to the log.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pool import LOG, key_of  # noqa: E402  (same dedupe semantics as the pool)

SUM_TOL = 1e-9
MIN_R, MAX_R = 1.0, 1e4        # sanity band for the evaluator's mean absolute ramp metric


def resolve(arg: str) -> Path:
    p = Path(arg)
    return p / "eval_log.jsonl" if p.is_dir() else p


def read_rows(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def check(rec: dict) -> tuple[np.ndarray | None, str]:
    """Return (weights, '') when the record is a usable measurement, else (None, reason)."""
    try:
        w = np.asarray(rec["weights"], dtype=float).reshape(-1)
        r = float(rec["R"])
    except (KeyError, TypeError, ValueError):
        return None, "unreadable"
    if w.size != 21:
        return None, "wrong_size"
    if not np.all(np.isfinite(w)):
        return None, "non_finite_weights"
    if (w < 0).any():
        return None, "negative_weights"
    if np.abs(w.reshape(3, 7).sum(axis=1) - 1.0).max() > SUM_TOL:
        return None, "rows_do_not_sum_to_one"
    if not np.isfinite(r):
        return None, "non_finite_R"
    if not (MIN_R <= r <= MAX_R):
        return None, "R_out_of_band"
    return w, ""


def sha256_of(vectors: list[np.ndarray]) -> str:
    h = hashlib.sha256()
    for w in vectors:
        h.update(np.asarray(w, dtype="<f8").tobytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--tag", required=True, help="tag stored on every appended record")
    ap.add_argument("--log", default=str(LOG))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    log = Path(args.log)
    existing: set[str] = set()
    next_id = 0
    if log.exists():
        with log.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    existing.add(key_of(np.asarray(rec["weights"], dtype=float)))
                    next_id = max(next_id, int(rec.get("eval_id", -1)) + 1)

    added: list[dict] = []
    report = {"tag": args.tag, "log": str(log), "sources": {}, "rejected": {}}
    for arg in args.sources:
        path = resolve(arg)
        if not path.exists():
            raise SystemExit(f"missing log: {path}")
        n_seen = n_new = n_dup = 0
        for rec in read_rows(path):
            n_seen += 1
            w, why = check(rec)
            if w is None:
                report["rejected"][why] = report["rejected"].get(why, 0) + 1
                continue
            k = key_of(w)
            if k in existing:
                n_dup += 1
                continue
            existing.add(k)
            n_new += 1
            added.append({
                "eval_id": next_id,
                "tag": args.tag,
                "weights": [float(v) for v in w],
                "R": float(rec["R"]),
                "time": rec.get("timestamp") or rec.get("time") or time.strftime("%Y-%m-%dT%H:%M:%S"),
                "worker": None,
                "call_secs": None,
                "call_sets": 1,
                "source": path.parent.name if path.name == "eval_log.jsonl" else path.name,
                "source_phase": rec.get("phase"),
                "source_eval_id": rec.get("eval_id"),
            })
            next_id += 1
        report["sources"][str(path)] = {"seen": n_seen, "added": n_new, "duplicate": n_dup}

    report["added_total"] = len(added)
    report["sha256_weights_appended"] = sha256_of([np.asarray(r["weights"]) for r in added])
    report["R_range_added"] = [min((r["R"] for r in added), default=None),
                              max((r["R"] for r in added), default=None)]
    report["env"] = {"python": sys.version.split()[0], "numpy": np.__version__,
                     "platform": platform.platform()}
    report["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    print(json.dumps({k: report[k] for k in ("sources", "added_total", "rejected",
                                             "R_range_added", "sha256_weights_appended")}, indent=2))
    if args.dry_run:
        print("dry run: nothing written")
        return 0
    if added:
        with log.open("a", encoding="utf-8") as fh:
            for rec in added:
                fh.write(json.dumps(rec) + "\n")
    out = HERE / f"ingest_report_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"appended {len(added)} -> {log}\nreport -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
