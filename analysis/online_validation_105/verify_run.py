"""Check a completed online run without touching the solver or its quota.

Usage: python verify_run.py [run_id_or_path]

Exits 0 and prints a one-line summary when every invariant holds; otherwise raises AssertionError
with the first violation. Pure pandas/numpy/json, no sklearn, no subprocess.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
EXPECTED_BEST_R = 105.9905449313228
EXPECTED_BEST_CALL = 20
EXPECTED_BEST_POINT = 9


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(argument: str) -> Path:
    path = Path(argument)
    candidate = path if path.is_absolute() else ROOT / "runs" / path
    if not candidate.is_dir():
        raise SystemExit(f"run directory not found: {candidate}")
    return candidate


def verify(run_dir: Path, *, expected_best: float | None = None) -> dict:
    evaluations = pd.read_csv(run_dir / "evaluations.csv", encoding="utf-8-sig")
    weights = evaluations.filter(regex=r"^b[123]_").to_numpy(float)
    assert len(evaluations) == 1000, f"expected 1000 rows, got {len(evaluations)}"
    assert evaluations["call"].nunique() == 20, "expected 20 distinct calls"
    assert (evaluations.groupby("call").size() == 50).all(), "every call must have exactly 50 points"
    assert np.isfinite(weights).all(), "non-finite weight"
    assert (weights >= 0).all(), "negative weight"
    assert np.abs(weights.reshape(-1, 3, 7).sum(2) - 1).max() < 1e-12, "each building must sum to 1"
    assert len(np.unique(np.round(weights, 9), axis=0)) == 1000, "duplicate points at atol=1e-9"
    assert evaluations["actual_R"].notna().all(), "missing actual R"

    best_index = evaluations["actual_R"].idxmin()
    best_r = float(evaluations.loc[best_index, "actual_R"])
    best_call = int(evaluations.loc[best_index, "call"])
    best_point = int(evaluations.loc[best_index, "point"])
    if expected_best is not None:
        assert abs(best_r - expected_best) < 1e-9, f"best R {best_r} != expected {expected_best}"
        assert (best_call, best_point) == (EXPECTED_BEST_CALL, EXPECTED_BEST_POINT), \
            f"best point moved to call {best_call} point {best_point}"

    running = evaluations.groupby("call")["actual_R"].min().sort_index().cummin().to_numpy()
    assert np.all(np.diff(running) <= 0), "running best is not monotone non-increasing"

    attempts = [json.loads(line) for line in
                (run_dir / "attempts.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    started = [record for record in attempts if record["status"] == "started"]
    ok = [record for record in attempts if record["status"] == "ok"]
    failed = [record for record in attempts if record["status"] == "failed"]
    unresolved = {record["call"] for record in started} - {record["call"] for record in ok + failed}
    assert len(started) == 20 and len(ok) == 20, f"expected 20 started/20 ok, got {len(started)}/{len(ok)}"
    assert not failed, f"{len(failed)} failed calls in the log"
    assert not unresolved, f"calls without a terminal record: {sorted(unresolved)}"

    # Every log must describe the same weights and R values for the same (call, point).
    calls = [json.loads(path.read_text(encoding="utf-8")) for path in
             sorted((run_dir / "calls").glob("call_*.json"))]
    ok_calls = {int(record["call"]): record for record in calls if record.get("status") == "ok"}
    assert sorted(ok_calls) == sorted(int(record["call"]) for record in ok), \
        "calls/*.json and attempts.jsonl disagree about which calls succeeded"
    for call, record in sorted(ok_calls.items()):
        rows = evaluations[evaluations["call"] == call].sort_values("point")
        weights = rows.filter(regex=r"^b[123]_").to_numpy(float)
        assert len(rows) == len(record["weights"]), f"call {call} has {len(rows)} CSV rows"
        np.testing.assert_allclose(np.asarray(record["weights"], float), weights, rtol=0, atol=1e-12,
                                   err_msg=f"call {call} weights differ between calls/ and evaluations.csv")
        np.testing.assert_allclose(np.asarray(record["actual_R"], float),
                                   rows["actual_R"].to_numpy(float), rtol=0, atol=1e-12,
                                   err_msg=f"call {call} R values differ between calls/ and evaluations.csv")
        np.testing.assert_allclose(np.asarray(record["predicted_mean"], float),
                                   rows["predicted_mean"].to_numpy(float), rtol=0, atol=1e-12,
                                   err_msg=f"call {call} predictions differ between calls/ and evaluations.csv")
        started_weights = [item["weights"] for item in attempts
                           if int(item["call"]) == call and item["status"] == "started"]
        np.testing.assert_allclose(np.asarray(started_weights[0], float), weights, rtol=0, atol=1e-12,
                                   err_msg=f"call {call} weights differ between attempts.jsonl and evaluations.csv")

    # The manifest must cover the call payloads too, not only the plotted outputs.
    manifest = json.loads((run_dir / "output_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest.items():
        path = run_dir / relative
        assert path.is_file(), f"manifest lists a missing file: {relative}"
        assert sha256_file(path) == expected, f"manifest hash mismatch: {relative}"
    assert "attempts.jsonl" in manifest, "output_manifest.json does not cover attempts.jsonl"
    for path in sorted((run_dir / "calls").glob("call_*.json")):
        assert f"calls/{path.name}" in manifest, f"output_manifest.json does not cover calls/{path.name}"

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert abs(summary["online_best_R"] - best_r) < 1e-9, "summary.json disagrees with evaluations.csv"
    assert summary["successful_calls"] == 20 and not summary["failed_calls"], "summary reports failures"

    return {"run": run_dir.name, "points": len(evaluations), "calls": int(evaluations["call"].nunique()),
            "best_R": best_r, "best_call": best_call, "best_point": best_point}


def main() -> None:
    argument = sys.argv[1] if len(sys.argv) > 1 else "independent_gp_2026"
    run_dir = resolve(argument)
    pinned = argument in ("independent_gp_2026", str(run_dir))
    result = verify(run_dir, expected_best=EXPECTED_BEST_R if pinned else None)
    print(json.dumps(result, ensure_ascii=False))
    print("PASS")


if __name__ == "__main__":
    main()
