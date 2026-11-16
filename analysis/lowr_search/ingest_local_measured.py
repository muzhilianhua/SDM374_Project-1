"""Import locally measured points into a standalone JSONL file, never the T36 log.

Only real evaluator results are accepted: every record must carry 21 finite, non-negative
weights with each building summing to 1, and a finite positive ``R``. Points already in the
read-only T36 log or an earlier source are skipped.

Usage (from this folder):
    python ingest_local_measured.py <run_dir_or_eval_log> [more...] --tag <tag>
        [--output imported/<tag>.jsonl] [--source-script <script> ...]

``--source-script`` accepts one archived generator per source, in the same order. The output
and an adjacent ``.provenance.json`` are created exclusively; existing evidence is never
overwritten. All recorded paths are relative to the repository root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from pool import LOG, key_of  # noqa: E402  (same dedupe semantics as the pool)

SUM_TOL = 1e-9
MIN_R, MAX_R = 1.0, 1e4


def resolve(arg: str) -> Path:
    p = Path(arg).resolve()
    return p / "eval_log.jsonl" if p.is_dir() else p


def relative(path: Path) -> str:
    return Path(os.path.relpath(path.resolve(), REPO_ROOT)).as_posix()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    ap.add_argument("--tag", required=True, help="tag stored on every imported record")
    ap.add_argument("--output", help="standalone JSONL path; defaults to imported/<tag>.jsonl")
    ap.add_argument("--source-script", action="append", default=[],
                    help="archived generator for each source, in source order")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.tag):
        raise SystemExit("tag may contain only letters, digits, dot, underscore and hyphen")
    if args.source_script and len(args.source_script) != len(args.sources):
        raise SystemExit("provide either no --source-script values or exactly one per source")

    output = Path(args.output) if args.output else HERE / "imported" / f"{args.tag.replace('-', '_')}.jsonl"
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()
    if output == LOG.resolve():
        raise SystemExit("refusing to write the T36 eval_log.jsonl; choose a standalone output")
    provenance_path = output.with_suffix(".provenance.json")

    existing: set[str] = set()
    next_id = 0
    with LOG.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                existing.add(key_of(np.asarray(rec["weights"], dtype=float)))
                next_id = max(next_id, int(rec.get("eval_id", -1)) + 1)

    added: list[dict] = []
    report = {
        "tag": args.tag,
        "output": relative(output),
        "t36_log_read_only": {
            "path": relative(LOG),
            "sha256": sha256_file(LOG),
            "records": next_id,
        },
        "sources": [],
        "rejected": {},
        "verification_status": "unverified imported measurements; not an established team best",
    }
    for index, arg in enumerate(args.sources):
        path = resolve(arg)
        if not path.exists():
            raise SystemExit(f"missing log: {path}")
        source = {
            "path": relative(path),
            "sha256": sha256_file(path),
            "seen": 0,
            "imported": 0,
            "duplicate": 0,
        }
        if args.source_script:
            script = Path(args.source_script[index]).resolve()
            source["generator"] = {"path": relative(script), "sha256": sha256_file(script)}
        for rec in read_rows(path):
            source["seen"] += 1
            w, why = check(rec)
            if w is None:
                report["rejected"][why] = report["rejected"].get(why, 0) + 1
                continue
            k = key_of(w)
            if k in existing:
                source["duplicate"] += 1
                continue
            existing.add(k)
            source["imported"] += 1
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
        report["sources"].append(source)

    payload = "".join(json.dumps(rec) + "\r\n" for rec in added).encode("utf-8")
    report.update({
        "imported_total": len(added),
        "output_sha256": hashlib.sha256(payload).hexdigest(),
        "sha256_weights_imported": sha256_of([np.asarray(r["weights"]) for r in added]),
        "R_range_imported": [min((r["R"] for r in added), default=None),
                             max((r["R"] for r in added), default=None)],
        "env": {"python": sys.version.split()[0], "numpy": np.__version__,
                "platform": platform.platform()},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })

    print(json.dumps({k: report[k] for k in ("sources", "imported_total", "rejected",
                                             "R_range_imported", "sha256_weights_imported")}, indent=2))
    if args.dry_run:
        print("dry run: nothing written")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as fh:
        fh.write(payload)
    with provenance_path.open("x", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")
    print(f"imported {len(added)} -> {relative(output)}\nprovenance -> {relative(provenance_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
