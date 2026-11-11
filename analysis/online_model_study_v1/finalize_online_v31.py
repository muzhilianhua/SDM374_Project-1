"""Validate completed V3.1 trajectories and score all unique global candidates."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
OUT = STUDY / "pilot" / "online_v3_1"
AUDIT = OUT / "global_audit"
SOLVER_DIR = ROOT / "solver"
SOLVER = SOLVER_DIR / "solver.exe"
MODELS = ("local-gp", "gp-ard", "histgb", "xgboost", "boosted-consensus")
ROUNDS = 10

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(STUDY))
from sdm.data import load_dataset
from run_online_v31 import point_key, read_csv, write_csv, write_json
from sdm.simplex import FEATURES, check_weights


def validate_all() -> dict:
    dataset = load_dataset(ROOT / "dataset_same8")
    if len(dataset.y) != 2048:
        raise RuntimeError(f"expected 2048 cold-start rows, got {len(dataset.y)}")
    report = {"replicates": [], "models": list(MODELS), "rounds_per_replicate": ROUNDS}
    for rep in range(1, 9):
        repdir = OUT / f"replicate_{rep:02d}"
        checkpoint = json.loads((repdir / "checkpoint.json").read_text(encoding="utf-8"))
        if checkpoint.get("completed_rounds") != ROUNDS or checkpoint.get("cold_start_rows") != 2048:
            raise RuntimeError(f"replicate {rep} checkpoint incomplete or cold start mismatch")
        rows = read_csv(repdir / "trajectories.csv")
        if len(rows) != len(MODELS) * ROUNDS:
            raise RuntimeError(f"replicate {rep}: expected 50 rows, got {len(rows)}")
        if {r["model"] for r in rows} != set(MODELS):
            raise RuntimeError(f"replicate {rep}: model roster mismatch")
        for model in MODELS:
            state = checkpoint["states"][model]
            X = np.asarray(state["X"], dtype=float)
            y = np.asarray(state["y"], dtype=float)
            if X.shape != (2058, 21) or y.shape != (2058,):
                raise RuntimeError(f"replicate {rep} {model}: invalid final training state shape")
            if not np.array_equal(X[:2048], dataset.X) or not np.array_equal(y[:2048], dataset.y):
                raise RuntimeError(f"replicate {rep} {model}: cold start changed")
            if len(state["history"]) != ROUNDS:
                raise RuntimeError(f"replicate {rep} {model}: missing own feedback records")
        selections_by_round = {}
        for row in rows:
            selections_by_round.setdefault(int(row["round"]), []).append(row)
        unique_selected = set()
        for rnd in range(1, ROUNDS + 1):
            selected_rows = selections_by_round.get(rnd, [])
            if len(selected_rows) != len(MODELS):
                raise RuntimeError(f"replicate {rep} round {rnd}: expected five model selections")
            pool_path = repdir / "candidate_pools" / f"round_{rnd:02d}.csv"
            pool_rows = read_csv(pool_path)
            if len(pool_rows) != 300:
                raise RuntimeError(f"replicate {rep} round {rnd}: candidate pool size is not 300")
            pool_by_id = {r["candidate_id"]: np.asarray([float(r[f]) for f in FEATURES]) for r in pool_rows}
            check_weights(np.asarray(list(pool_by_id.values())))
            score_path = repdir / "round_scores" / f"round_{rnd:02d}.json"
            score_doc = json.loads(score_path.read_text(encoding="utf-8"))
            score_weights = np.asarray(score_doc["weights"], dtype=float).reshape(-1, 21)
            score_values = np.asarray(score_doc["R"], dtype=float)
            if score_values.shape != (len(score_weights),) or not np.isfinite(score_values).all():
                raise RuntimeError(f"replicate {rep} round {rnd}: invalid solver score record")
            score_map = {point_key(x): float(y) for x, y in zip(score_weights, score_values)}
            for row in selected_rows:
                cid = row["candidate_id"]
                if cid not in pool_by_id:
                    raise RuntimeError(f"selected candidate {cid} not found in its pool")
                x = pool_by_id[cid]
                key = point_key(x)
                if key not in score_map or not np.isclose(float(row["actual_R"]), score_map[key], atol=1e-12, rtol=0):
                    raise RuntimeError(f"selected candidate {cid} lacks matching solver label")
                if not np.isfinite(float(row["prediction_error_actual_minus_predicted"])):
                    raise RuntimeError(f"selected point residual is non-finite: {cid}")
                unique_selected.add(key)
            credit = sum(float(r["round_first_place_credit"]) for r in selected_rows)
            if not np.isclose(credit, 1.0, atol=1e-12):
                raise RuntimeError(f"replicate {rep} round {rnd}: winner credit sums to {credit}")
        report["replicates"].append({"replicate": rep, "trajectory_rows": len(rows), "unique_selected_points": len(unique_selected), "status": "passed"})
    report["total_trajectory_rows"] = sum(x["trajectory_rows"] for x in report["replicates"])
    report["status"] = "passed"
    write_json(OUT / "final_protocol_validation.json", report)
    return report


def load_score_cache() -> dict[bytes, float]:
    cache: dict[bytes, float] = {}
    paths = list(OUT.glob("replicate_*/solver_calls/round_*.json")) + list(AUDIT.glob("solver_calls/call_*.json"))
    for p in sorted(paths):
        doc = json.loads(p.read_text(encoding="utf-8"))
        if doc.get("returncode") != 0:
            continue
        y = np.asarray(json.loads(doc["stdout"])["R"], dtype=float)
        X = np.asarray(doc["weights"], dtype=float).reshape(-1, 21)
        for x, val in zip(X, y):
            key = point_key(x)
            if key in cache and not np.isclose(cache[key], val, atol=1e-9, rtol=0):
                raise RuntimeError(f"inconsistent duplicate solver labels: {p}")
            cache[key] = float(val)
    return cache


def audit_global() -> dict:
    global_points: dict[bytes, tuple[np.ndarray, list[str]]] = {}
    for rep in range(1, 9):
        repdir = OUT / f"replicate_{rep:02d}"
        for rnd in range(1, ROUNDS + 1):
            rows = read_csv(repdir / "candidate_pools" / f"round_{rnd:02d}.csv")
            for row in rows:
                if row["source"] != "global":
                    continue
                x = np.asarray([float(row[f]) for f in FEATURES], dtype=float)
                k = point_key(x)
                global_points.setdefault(k, (x, []))[1].append(row["candidate_id"])
    cache = load_score_cache()
    audit_rows: list[dict] = []
    missing: list[tuple[bytes, np.ndarray, list[str]]] = []
    for key, (x, ids) in global_points.items():
        if key in cache:
            audit_rows.append({"weight_key": ",".join(f"{v:.12f}" for v in np.round(x, 12)), "candidate_ids": ";".join(ids), "actual_R": cache[key], "source": "reused_model_selection_solver_result"})
        else:
            missing.append((key, x, ids))
    AUDIT.mkdir(parents=True, exist_ok=True)
    marker = OUT / "quota_calls_since_reset.txt"
    used = int(marker.read_text(encoding="utf-8").strip()) if marker.exists() else 0
    invocation = 0
    for start in range(0, len(missing), 50):
        chunk = missing[start:start + 50]
        if used >= 20:
            (SOLVER_DIR / "usage.count").unlink(missing_ok=True)
            used = 0
        invocation += 1
        weights = [x.reshape(3, 7).tolist() for _, x, _ in chunk]
        t0 = time.perf_counter()
        result = subprocess.run([str(SOLVER)], cwd=SOLVER_DIR, input=json.dumps({"weights": weights}, separators=(",", ":")), text=True, capture_output=True, timeout=1800)
        elapsed = time.perf_counter() - t0
        call_path = AUDIT / "solver_calls" / f"call_{invocation:02d}.json"
        doc = {"purpose": "global_candidate_audit_only_not_training_feedback", "weights": weights, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr, "elapsed_seconds": elapsed}
        write_json(call_path, doc)
        used += 1
        marker.write_text(str(used), encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(f"global solver audit failed at {call_path}: {result.stderr}")
        values = np.asarray(json.loads(result.stdout)["R"], dtype=float)
        if values.shape != (len(chunk),) or not np.isfinite(values).all():
            raise RuntimeError(f"global solver audit returned invalid labels: {call_path}")
        for (_, x, ids), val in zip(chunk, values):
            key = point_key(x)
            cache[key] = float(val)
            audit_rows.append({"weight_key": ",".join(f"{v:.12f}" for v in np.round(x, 12)), "candidate_ids": ";".join(ids), "actual_R": float(val), "source": "global_audit_solver_call"})
        print(f"global audit {min(start + 50, len(missing))}/{len(missing)} unique unsolved points; call={len(chunk)}; {elapsed:.1f}s", flush=True)
    audit_rows.sort(key=lambda r: r["weight_key"])
    write_csv(AUDIT / "global_candidate_results.csv", audit_rows)
    summary = {
        "unique_global_candidates": len(global_points),
        "already_scored_as_model_selections": len(global_points) - len(missing),
        "new_solver_points": len(missing),
        "total_result_rows": len(audit_rows),
        "minimum_global_R": min((float(r["actual_R"]) for r in audit_rows), default=None),
        "solver_invocations": invocation,
        "training_feedback_added": False,
        "status": "complete",
    }
    write_json(AUDIT / "global_audit_summary.json", summary)
    return summary


if __name__ == "__main__":
    validation = validate_all()
    print("Protocol validation passed:", json.dumps(validation, ensure_ascii=False), flush=True)
    print("Starting global-candidate audit (results are kept out of model training).", flush=True)
    print(json.dumps(audit_global(), ensure_ascii=False, indent=2), flush=True)


