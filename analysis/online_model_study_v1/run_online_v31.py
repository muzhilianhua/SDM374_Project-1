"""Run the staged V3.1 online surrogate comparison with durable round checkpoints."""
from __future__ import annotations

import csv
import json
import os
import sqlite3
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
SOLVER_DIR = ROOT / "solver"
SOLVER = SOLVER_DIR / "solver.exe"
POOL_SIZE = 300
ROUNDS = 10
SEED = 2026
MODELS = ("local-gp", "gp-ard", "histgb", "xgboost", "boosted-consensus")
MODEL_PARAMS = {
    "local-gp": {"seed": SEED},
    "gp-ard": {"family": "gp", "kernel": "additive-ard", "lengths": (0.5,), "noises": (0.03,), "seed": SEED},
    "histgb": {"seed": SEED},
    "xgboost": {"seed": SEED},
    "boosted-consensus": {"seed": SEED},
}

sys.path.insert(0, str(ROOT / "src"))
from sdm.data import load_dataset
from sdm.registry import build_model, models
from sdm.simplex import FEATURES, check_weights


def point_key(point: np.ndarray) -> bytes:
    return np.round(np.asarray(point, dtype=np.float64), 12).tobytes()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def pool_rows(rep: int, round_number: int) -> tuple[np.ndarray, list[dict]]:
    path = OUT / f"replicate_{rep:02d}" / "candidate_pools" / f"round_{round_number:02d}.csv"
    rows = read_csv(path)
    if len(rows) != POOL_SIZE:
        raise RuntimeError(f"{path}: expected {POOL_SIZE} candidates, got {len(rows)}")
    X = np.asarray([[float(row[name]) for name in FEATURES] for row in rows], dtype=float)
    check_weights(X)
    return X, [{k: v for k, v in row.items() if k not in FEATURES} for row in rows]


def model_fit(name: str, X: np.ndarray, y: np.ndarray):
    model = build_model("bayes-search" if name == "gp-ard" else name, MODEL_PARAMS[name])
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        model.fit(X, y)
    return model, time.perf_counter() - start


def load_v31_score_cache() -> dict[bytes, float]:
    cache: dict[bytes, float] = {}
    for path in sorted(OUT.glob("replicate_*/solver_calls/round_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("returncode") != 0:
            continue
        values = np.asarray(json.loads(record["stdout"])["R"], dtype=float)
        points = np.asarray(record["weights"], dtype=float).reshape(-1, 21)
        if values.shape != (len(points),) or not np.isfinite(values).all():
            raise RuntimeError(f"invalid cached solver record: {path}")
        for point, value in zip(points, values):
            key = point_key(point)
            if key in cache and not np.isclose(cache[key], value, rtol=0, atol=1e-9):
                raise RuntimeError(f"inconsistent duplicate solver scores for {path}")
            cache[key] = float(value)
    return cache


def solve_unique(points: list[np.ndarray], rep: int, round_number: int) -> tuple[dict[bytes, float], dict]:
    cache = load_v31_score_cache()
    calls = OUT / f"replicate_{rep:02d}" / "solver_calls"
    calls.mkdir(parents=True, exist_ok=True)
    path = calls / f"round_{round_number:02d}.json"
    missing = [point for point in points if point_key(point) not in cache]
    if missing:
        # solver.exe grants 20 invocations per usage.count cycle. Reset between calls
        # only when the recorded invocation count has reached the documented limit.
        marker = OUT / "quota_calls_since_reset.txt"
        used = int(marker.read_text(encoding="utf-8").strip()) if marker.exists() else 0
        if used >= 20:
            (SOLVER_DIR / "usage.count").unlink(missing_ok=True)
            used = 0
        marker.write_text(str(used + 1), encoding="utf-8")
        payload = {"weights": [point.reshape(3, 7).tolist() for point in missing]}
        start = time.perf_counter()
        result = subprocess.run(
            [str(SOLVER)], cwd=SOLVER_DIR,
            input=json.dumps(payload, separators=(",", ":")),
            text=True, capture_output=True, timeout=900,
        )
        elapsed = time.perf_counter() - start
        record = {
            "replicate": rep, "round": round_number, "weights": payload["weights"],
            "returncode": result.returncode, "stdout": result.stdout,
            "stderr": result.stderr, "elapsed_seconds": elapsed,
        }
        write_json(path, record)
        if result.returncode != 0:
            raise RuntimeError(f"solver failed for replicate {rep}, round {round_number}: {result.stderr}")
        results = np.asarray(json.loads(result.stdout)["R"], dtype=float)
        if results.shape != (len(missing),) or not np.isfinite(results).all():
            raise RuntimeError(f"solver result shape/value mismatch in {path}: {results.shape}")
        for point, score in zip(missing, results):
            cache[point_key(point)] = float(score)
    elapsed = float(record["elapsed_seconds"]) if missing else 0.0
    mapping = {point_key(point): cache[point_key(point)] for point in points}
    score_path = OUT / f"replicate_{rep:02d}" / "round_scores" / f"round_{round_number:02d}.json"
    write_json(score_path, {
        "replicate": rep, "round": round_number,
        "weights": [point.reshape(3, 7).tolist() for point in points],
        "R": [mapping[point_key(point)] for point in points],
        "new_solver_points": len(missing),
        "solver_call_record": str(path.relative_to(OUT)) if missing else "reused_prior_v31_solver_result",
        "elapsed_seconds": elapsed,
    })
    return mapping, {"elapsed_seconds": elapsed, "new_solver_points": len(missing)}


def run_replicate(rep: int, dataset) -> dict:
    repdir = OUT / f"replicate_{rep:02d}"
    checkpoint = repdir / "checkpoint.json"
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        if saved.get("models") != list(MODELS) or saved.get("cold_start_rows") != len(dataset.y):
            raise RuntimeError(f"checkpoint protocol mismatch: {checkpoint}")
        states = {
            name: {
                "X": np.asarray(state["X"], dtype=float),
                "y": np.asarray(state["y"], dtype=float),
                "history": state["history"],
                "best": float(state["best"]),
            }
            for name, state in saved["states"].items()
        }
        start_round = int(saved["completed_rounds"])
    else:
        states = {
            name: {"X": dataset.X.copy(), "y": dataset.y.copy(), "history": [], "best": float(dataset.y.min())}
            for name in MODELS
        }
        start_round = 0
    if start_round > ROUNDS:
        raise RuntimeError(f"invalid completed round count {start_round}")

    for round_number in range(start_round + 1, ROUNDS + 1):
        Xpool, metadata = pool_rows(rep, round_number)
        selections = []
        for name in MODELS:
            state = states[name]
            model, fit_seconds = model_fit(name, state["X"], state["y"])
            with threadpool_limits(limits=1):
                means, _ = model.predict(Xpool)
            means = np.asarray(means, dtype=float)
            if means.shape != (POOL_SIZE,) or not np.isfinite(means).all():
                raise RuntimeError(f"{name} produced invalid predictions in replicate {rep}, round {round_number}")
            selected = min(range(POOL_SIZE), key=lambda i: (float(means[i]), metadata[i]["candidate_id"]))
            selections.append({
                "model": name, "point": Xpool[selected], "metadata": metadata[selected],
                "predicted_mean": float(means[selected]), "fit_seconds": fit_seconds,
            })
            del model
            print(
                f"rep {rep}/{8} round {round_number}/{ROUNDS} {name}: "
                f"fit={fit_seconds:.1f}s, selected={metadata[selected]['candidate_id']}, "
                f"pred={means[selected]:.6f}", flush=True,
            )

        unique_points = []
        index_by_key = {}
        for selection in selections:
            point = selection["point"]
            key = point_key(point)
            if key not in index_by_key:
                index_by_key[key] = len(unique_points)
                unique_points.append(point)
        actual, call_record = solve_unique(unique_points, rep, round_number)
        current_scores = [actual[point_key(item["point"])] for item in selections]
        best_score = min(current_scores)
        winners = [i for i, score in enumerate(current_scores) if np.isclose(score, best_score, rtol=0, atol=1e-12)]
        win_share = 1.0 / len(winners)

        for i, selection in enumerate(selections):
            name = selection["model"]
            state = states[name]
            score = current_scores[i]
            state["X"] = np.vstack((state["X"], selection["point"]))
            state["y"] = np.append(state["y"], score)
            state["best"] = min(state["best"], score)
            state["history"].append({
                "replicate": rep, "round": round_number, "model": name,
                "candidate_id": selection["metadata"]["candidate_id"],
                "source": selection["metadata"]["source"],
                "anchor_id": selection["metadata"]["anchor_id"],
                "step": selection["metadata"]["step"],
                "predicted_mean": selection["predicted_mean"],
                "actual_R": score,
                "prediction_error_actual_minus_predicted": score - selection["predicted_mean"],
                "best_R_after_round": state["best"],
                "fit_seconds": selection["fit_seconds"],
                "round_first_place_credit": win_share if i in winners else 0.0,
                "shared_solver_points": len(unique_points),
                "new_solver_points": call_record["new_solver_points"],
                "solver_elapsed_seconds": call_record["elapsed_seconds"],
            })
        write_csv(repdir / "trajectories.csv", [row for name in MODELS for row in states[name]["history"]])
        write_json(checkpoint, {
            "replicate": rep, "completed_rounds": round_number,
            "models": list(MODELS), "cold_start_rows": len(dataset.y),
            "initial_best_R": float(dataset.y.min()),
            "states": {
                name: {"X": state["X"].tolist(), "y": state["y"].tolist(),
                       "history": state["history"], "best": state["best"]}
                for name, state in states.items()
            },
        })
        print(
            f"rep {rep} round {round_number}: solver scored {call_record['new_solver_points']} new point(s), "
            f"best={best_score:.6f}; checkpoint saved", flush=True,
        )
    return states


def validate_completed_replicate(rep: int, dataset) -> dict:
    """Check protocol integrity before advancing from the first replicate."""
    repdir = OUT / f"replicate_{rep:02d}"
    checkpoint = json.loads((repdir / "checkpoint.json").read_text(encoding="utf-8"))
    if checkpoint.get("completed_rounds") != ROUNDS:
        raise RuntimeError("replicate did not finish all ten rounds")
    if checkpoint.get("cold_start_rows") != len(dataset.y) or len(dataset.y) != 2048:
        raise RuntimeError("cold-start row count is not 2048")
    rows = read_csv(repdir / "trajectories.csv")
    if len(rows) != len(MODELS) * ROUNDS:
        raise RuntimeError(f"expected 50 trajectory rows, got {len(rows)}")
    if {row["model"] for row in rows} != set(MODELS):
        raise RuntimeError("trajectory model roster mismatch")

    for name in MODELS:
        state = checkpoint["states"][name]
        X = np.asarray(state["X"], dtype=float)
        y = np.asarray(state["y"], dtype=float)
        if X.shape != (2048 + ROUNDS, 21) or y.shape != (2048 + ROUNDS,):
            raise RuntimeError(f"{name}: expected 2048 cold-start rows plus 10 own feedback rows")
        if not np.array_equal(X[:2048], dataset.X) or not np.array_equal(y[:2048], dataset.y):
            raise RuntimeError(f"{name}: initial 2048 rows/labels changed")
        if len(state["history"]) != ROUNDS:
            raise RuntimeError(f"{name}: expected ten feedback records")

    database_path = STUDY / "solver_database" / "solver_observations.sqlite3"
    with sqlite3.connect(database_path) as conn:
        known_keys = {
            np.round([float(v) for v in row[0].split(",")], 12).tobytes()
            for row in conn.execute("SELECT weight_key FROM solver_observations")
        }
    selected_keys = set()
    for round_number in range(1, ROUNDS + 1):
        pool, metadata = pool_rows(rep, round_number)
        pool_by_id = {meta["candidate_id"]: pool[i] for i, meta in enumerate(metadata)}
        score_path = repdir / "round_scores" / f"round_{round_number:02d}.json"
        score_record = json.loads(score_path.read_text(encoding="utf-8"))
        returned = np.asarray(score_record["R"], dtype=float)
        sent = np.asarray(score_record["weights"], dtype=float).reshape(-1, 21)
        if returned.shape != (len(sent),) or not np.isfinite(returned).all():
            raise RuntimeError(f"solver score record mismatch in round {round_number}")
        scored = {point_key(point): float(score) for point, score in zip(sent, returned)}
        for row in [row for row in rows if int(row["round"]) == round_number]:
            candidate_id = row["candidate_id"]
            if candidate_id not in pool_by_id:
                raise RuntimeError(f"selected candidate {candidate_id} is not in the shared pool")
            point = pool_by_id[candidate_id]
            key = point_key(point)
            if key not in scored:
                raise RuntimeError(f"selected point was not in the saved solver call: {candidate_id}")
            if not np.isclose(float(row["actual_R"]), scored[key], rtol=0, atol=1e-12):
                raise RuntimeError(f"trajectory/solver label mismatch for {candidate_id}")
            if not np.isfinite(float(row["prediction_error_actual_minus_predicted"])):
                raise RuntimeError(f"non-finite selected-point residual for {candidate_id}")
            selected_keys.add(key)
        if set(scored) & known_keys:
            raise RuntimeError(f"replicate {rep} solver call duplicated a point in the canonical database")
    if len(selected_keys) > ROUNDS * len(MODELS):
        raise RuntimeError("unique selected-point count exceeds the number of selections")
    return {
        "replicate": rep, "status": "passed", "trajectory_rows": len(rows),
        "model_count": len(MODELS), "rounds": ROUNDS,
        "solver_calls": ROUNDS, "unique_solver_points": len(selected_keys),
        "cold_start_rows_per_model": 2048, "post_run_rows_per_model": 2048 + ROUNDS,
    }


def main() -> None:
    available = models()
    required = {"local-gp", "bayes-search", "histgb", "xgboost", "boosted-consensus"}
    missing = required - set(available)
    if missing:
        raise RuntimeError(f"missing registered models: {sorted(missing)}")
    if not SOLVER.exists():
        raise FileNotFoundError(SOLVER)
    manifest = json.loads((OUT / "candidate_generation_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("solver_called") is not False or len(manifest.get("summary", [])) != 80:
        raise RuntimeError("candidate generation manifest is missing or incompatible")
    dataset = load_dataset(ROOT / "dataset_same8")
    marker = OUT / "quota_calls_since_reset.txt"
    if not marker.exists():
        previous = STUDY / "pilot" / "online_v3_directional" / "quota_calls_since_reset.txt"
        marker.write_text(previous.read_text(encoding="utf-8").strip() if previous.exists() else "0", encoding="utf-8")

    # The first replicate is a gated run: protocol integrity is checked before the
    # remaining seven are allowed to proceed.
    if not (OUT / "replicate_01" / "checkpoint.json").exists():
        print("Starting V3.1 replicate 1/8 from its 2048-row cold start", flush=True)
        run_replicate(1, dataset)
    report = validate_completed_replicate(1, dataset)
    write_json(OUT / "first_replicate_validation.json", report)
    print(f"Replicate 1 passed protocol checks: {json.dumps(report)}", flush=True)

    for rep in range(2, 9):
        print(f"Starting V3.1 replicate {rep}/8 from its 2048-row cold start", flush=True)
        run_replicate(rep, dataset)
    print("All eight V3.1 replicates completed.", flush=True)


if __name__ == "__main__":
    main()
