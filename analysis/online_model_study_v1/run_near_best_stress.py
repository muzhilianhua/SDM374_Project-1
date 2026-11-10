"""Run a small, model-independent test cohort concentrated near known low-R weights."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
OUT = STUDY / "pilot" / "near_best_stress"
SOLVER_DIR = ROOT / "solver"
SOLVER = SOLVER_DIR / "solver.exe"
sys.path.insert(0, str(ROOT / "src"))

from sdm.data import load_dataset
from sdm.registry import build_model
from sdm.simplex import FEATURES

MODELS = ("local-gp", "histgb", "xgboost", "boosted-consensus", "ridge2")
SEED = 4103
MODEL_SEED = 2026
N_TEST = 50
LOW_TARGET = 115.68
NEAR_BEST_BAND = 5.0


def point_key(point: np.ndarray) -> bytes:
    return np.round(np.asarray(point, dtype=float), 12).tobytes()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_candidate_pool_points() -> np.ndarray:
    points = []
    with (STUDY / "candidate_pools" / "round_01.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            points.append([float(row[name]) for name in FEATURES])
    for round_number in range(2, 21):
        with (STUDY / "candidate_pools" / f"round_{round_number:02d}.csv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                points.append([float(row[name]) for name in FEATURES])
    return np.asarray(points, dtype=float)


def generate_stress_points(dataset, blocked: set[bytes]) -> tuple[np.ndarray, list[dict]]:
    rng = np.random.default_rng(SEED)
    elite = np.argsort(dataset.y, kind="stable")[:100]
    anchors = dataset.X[elite]
    anchor_ids = tuple(dataset.ids[index] for index in elite)
    points, metadata = [], []
    while len(points) < N_TEST:
        index = int(rng.integers(0, len(anchors)))
        alpha = float(rng.uniform(0.005, 0.05))
        diffuse = rng.dirichlet(np.ones(7), size=3)
        point = ((1.0 - alpha) * anchors[index].reshape(3, 7) + alpha * diffuse).reshape(21)
        key = point_key(point)
        if key in blocked:
            continue
        blocked.add(key)
        points.append(point)
        metadata.append({"candidate_id": f"near_best_{len(points):03d}",
                         "anchor_id": anchor_ids[index], "alpha": alpha})
    return np.asarray(points, dtype=float), metadata


def main() -> None:
    if not SOLVER.is_file():
        raise SystemExit(f"missing solver: {SOLVER}")
    result_file = OUT / "results.json"
    if result_file.exists():
        raise SystemExit(f"refusing to overwrite existing run: {result_file}")

    dataset = load_dataset(ROOT / "dataset_same8")
    candidate_points = load_candidate_pool_points()
    blocked = {point_key(row) for row in dataset.X}
    blocked.update(point_key(row) for row in candidate_points)
    states = {}
    for name in MODELS:
        rows = list(csv.DictReader((STUDY / "pilot" / "online_runs" / name / "trajectory.csv")
                                   .open(encoding="utf-8", newline="")))
        X_new = np.asarray([[float(row[feature]) for feature in FEATURES] for row in rows])
        y_new = np.asarray([float(row["actual_R"]) for row in rows])
        blocked.update(point_key(row) for row in X_new)
        states[name] = {"X": np.vstack([dataset.X, X_new]), "y": np.r_[dataset.y, y_new]}

    # Train all five final pilot models before creating or exposing the stress-test inputs.
    models = {}
    for name in MODELS:
        model = build_model(name, {"seed": MODEL_SEED})
        started = time.perf_counter()
        model.fit(states[name]["X"], states[name]["y"])
        models[name] = model
        states[name]["fit_seconds"] = time.perf_counter() - started
        print(f"refit {name}: {states[name]['fit_seconds']:.1f}s", flush=True)

    test_X, test_meta = generate_stress_points(dataset, blocked)
    pred_dir = OUT / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    prediction_map = {}
    for name, model in models.items():
        mean, std = model.predict(test_X)
        mean, std = np.asarray(mean, dtype=float), np.asarray(std, dtype=float)
        prediction_map[name] = mean
        write_csv(pred_dir / f"{name}.csv", [
            {**meta, "predicted_mean": float(m), "predicted_std": float(s),
             **{feature: float(value) for feature, value in zip(FEATURES, point)}}
            for point, meta, m, s in zip(test_X, test_meta, mean, std)
        ])
    models.clear()

    # Persist test inputs only after every model has predicted them.
    write_csv(OUT / "weights.csv", [
        {**meta, **{feature: float(value) for feature, value in zip(FEATURES, point)}}
        for point, meta in zip(test_X, test_meta)
    ])
    call_dir = OUT / "solver_calls"
    call_dir.mkdir(parents=True, exist_ok=True)
    truth_values = []
    elapsed_total = 0.0
    batch_size = 5
    for start in range(0, N_TEST, batch_size):
        stop = min(start + batch_size, N_TEST)
        call_path = call_dir / f"near_best_{start + 1:03d}_{stop:03d}.json"
        if call_path.exists():
            record = json.loads(call_path.read_text(encoding="utf-8"))
            if record.get("returncode") != 0:
                raise RuntimeError(f"previous solver batch failed; inspect {call_path}")
            truth_values.extend(float(value) for value in json.loads(record["stdout"])["R"])
            elapsed_total += float(record.get("elapsed_seconds", 0.0))
            continue
        batch = test_X[start:stop]
        payload = {"weights": [point.reshape(3, 7).tolist() for point in batch]}
        started = time.perf_counter()
        try:
            process = subprocess.run([str(SOLVER)], cwd=SOLVER_DIR,
                input=json.dumps(payload, separators=(",", ":")), text=True,
                capture_output=True, timeout=600)
        except subprocess.TimeoutExpired as error:
            elapsed = time.perf_counter() - started
            write_json(call_path, {"weights": payload["weights"], "returncode": None,
                "stdout": error.stdout, "stderr": error.stderr, "elapsed_seconds": elapsed,
                "timed_out": True})
            raise RuntimeError(f"solver batch {start + 1}-{stop} timed out after {elapsed:.1f}s") from error
        elapsed = time.perf_counter() - started
        record = {"weights": payload["weights"], "returncode": process.returncode,
            "stdout": process.stdout, "stderr": process.stderr, "elapsed_seconds": elapsed}
        write_json(call_path, record)
        if process.returncode != 0:
            raise RuntimeError(f"solver batch {start + 1}-{stop} failed: {process.stderr.strip()}")
        values = np.asarray(json.loads(process.stdout)["R"], dtype=float)
        if values.shape != (stop-start,) or not np.isfinite(values).all():
            raise RuntimeError(f"solver batch {start + 1}-{stop} returned invalid scores")
        truth_values.extend(float(value) for value in values)
        elapsed_total += elapsed
        print(f"solver batch {start + 1}-{stop}/{N_TEST}: {elapsed:.1f}s", flush=True)
    truth = np.asarray(truth_values, dtype=float)
    if truth.shape != (N_TEST,) or not np.isfinite(truth).all():
        raise RuntimeError("solver returned invalid or incomplete R values")
    write_csv(OUT / "solver_results.csv", [
        {"candidate_id": meta["candidate_id"], "actual_R": float(actual)}
        for meta, actual in zip(test_meta, truth)
    ])

    baseline = float(dataset.y.min())
    summary = []
    for name in MODELS:
        pred = prediction_map[name]
        chosen = int(np.argmin(pred))
        near_mask = truth <= baseline + NEAR_BEST_BAND
        summary.append({"model": name, "fit_seconds": states[name]["fit_seconds"],
            "overall_MAE": float(np.mean(np.abs(pred-truth))),
            "near_best_band_R_le_baseline_plus_5_n": int(near_mask.sum()),
            "near_best_band_MAE": float(np.mean(np.abs(pred[near_mask]-truth[near_mask]))) if near_mask.any() else "",
            "target_hit_count_R_le_115_68": int(np.sum(truth <= LOW_TARGET)),
            "predicted_best_test_R": float(truth[chosen]), "oracle_best_test_R": float(truth.min()),
            "selection_regret": float(truth[chosen]-truth.min()),
            "selected_candidate_id": test_meta[chosen]["candidate_id"]})
    write_csv(OUT / "model_comparison.csv", summary)
    write_json(result_file, {"status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "models": list(MODELS), "training_rows_per_model": 2048+5, "stress_test_points": N_TEST,
        "seed": SEED, "perturbation_alpha": [0.005, 0.05], "near_best_band": baseline+NEAR_BEST_BAND,
        "target_R": LOW_TARGET, "solver_seconds": elapsed_total,
        "solver_invocations": (N_TEST + batch_size - 1) // batch_size,
        "solver_sha256": file_sha256(SOLVER),
        "training_data_sha256": dataset.sha256, "spearman_computed": False,
        "test_labels_used_after_predictions": True})
    print(f"Near-best stress comparison saved to {OUT / 'model_comparison.csv'}", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
