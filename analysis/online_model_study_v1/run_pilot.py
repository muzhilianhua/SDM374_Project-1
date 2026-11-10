"""Run the five-model, five-round local online-learning pilot."""
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
SOLVER_DIR = ROOT / "solver"
SOLVER = SOLVER_DIR / "solver.exe"
sys.path.insert(0, str(ROOT / "src"))

from sdm.data import load_dataset
from sdm.registry import build_model
from sdm.simplex import FEATURES, sample

MODELS = ("local-gp", "histgb", "xgboost", "boosted-consensus", "ridge2")
PILOT_ROUNDS = 5
POOL_COUNT = 20
POOL_SIZE = 100
GLOBAL_PER_POOL = 50
LOCAL_PER_POOL = 50
PILOT_TEST_SIZE = 100
LOW_R_THRESHOLD = 115.68
POOL_SEED = 4101
TEST_SEED = 4102
MODEL_SEED = 2026


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def point_key(point: np.ndarray) -> bytes:
    return np.round(np.asarray(point, dtype=np.float64), 12).tobytes()


def generate_mixed_pool(n: int, rng: np.random.Generator, anchors: np.ndarray, anchor_ids: tuple[str, ...],
                        blocked: set[bytes], round_number: int) -> tuple[np.ndarray, list[dict]]:
    """Generate half global simplex samples and half perturbations around known low-R rows."""
    points: list[np.ndarray] = []
    metadata: list[dict] = []
    quota = (n // 2, n - n // 2)
    for source, count in (("global", quota[0]), ("local", quota[1])):
        accepted = 0
        attempts = 0
        while accepted < count:
            attempts += 1
            if attempts > count * 100:
                raise RuntimeError(f"could not produce enough unique {source} candidates")
            if source == "global":
                point = sample(1, rng)[0]
                anchor_id, alpha = "", ""
            else:
                anchor_index = int(rng.integers(0, len(anchors)))
                anchor = anchors[anchor_index].reshape(3, 7)
                diffuse = rng.dirichlet(np.ones(7), size=3)
                alpha = float(rng.uniform(0.10, 0.25))
                point = ((1.0 - alpha) * anchor + alpha * diffuse).reshape(21)
                anchor_id = anchor_ids[anchor_index]
            key = point_key(point)
            if key in blocked:
                continue
            blocked.add(key)
            points.append(point)
            prefix = "g" if source == "global" else "l"
            metadata.append({
                "candidate_id": f"r{round_number:02d}_{prefix}{accepted + 1:03d}",
                "source": source,
                "anchor_id": anchor_id,
                "perturbation_alpha": alpha,
            })
            accepted += 1
    return np.asarray(points, dtype=float), metadata


def generate_candidate_pools(dataset) -> tuple[list[np.ndarray], list[list[dict]]]:
    rng = np.random.default_rng(POOL_SEED)
    elite_order = np.argsort(dataset.y, kind="stable")[:100]
    anchors = dataset.X[elite_order]
    anchor_ids = tuple(dataset.ids[index] for index in elite_order)
    blocked = {point_key(row) for row in dataset.X}
    pools, metadata = [], []
    for round_number in range(1, POOL_COUNT + 1):
        pool, rows = generate_mixed_pool(POOL_SIZE, rng, anchors, anchor_ids, blocked, round_number)
        pools.append(pool)
        metadata.append(rows)
        write_csv(STUDY / "candidate_pools" / f"round_{round_number:02d}.csv", [
            {**meta, **{feature: float(value) for feature, value in zip(FEATURES, point)}}
            for point, meta in zip(pool, rows)
        ])
    write_json(STUDY / "candidate_pools" / "generation_manifest.json", {
        "seed": POOL_SEED,
        "pools": POOL_COUNT,
        "points_per_pool": POOL_SIZE,
        "global_points_per_pool": GLOBAL_PER_POOL,
        "local_points_per_pool": LOCAL_PER_POOL,
        "local_anchor_ids": list(anchor_ids),
        "local_anchors": "100 lowest-R rows in the initial 2048-row table; candidate metadata records the anchor candidate_id",
        "local_generation": "per-building convex blend: (1-alpha)*anchor + alpha*Dirichlet(1,...,1), alpha uniform on [0.10, 0.25]",
        "global_generation": "independent Dirichlet(1,...,1) draw for each building",
        "deduplication": "rounded to 12 decimal places against training rows and all earlier candidate pools",
    })
    return pools, metadata


def solver_call(points: np.ndarray, call_name: str, call_dir: Path) -> tuple[np.ndarray, float]:
    payload = {"weights": [point.reshape(3, 7).tolist() for point in points]}
    started = time.perf_counter()
    result = subprocess.run(
        [str(SOLVER)], cwd=SOLVER_DIR, input=json.dumps(payload, separators=(",", ":")),
        text=True, capture_output=True, timeout=600,
    )
    elapsed = time.perf_counter() - started
    record = {"call": call_name, "weights": payload["weights"], "returncode": result.returncode,
              "stdout": result.stdout, "stderr": result.stderr, "elapsed_seconds": elapsed}
    write_json(call_dir / f"{call_name}.json", record)
    if result.returncode != 0:
        raise RuntimeError(f"solver failed for {call_name}: {result.stderr.strip()}")
    try:
        values = np.asarray(json.loads(result.stdout)["R"], dtype=float)
    except (ValueError, KeyError, TypeError) as error:
        raise RuntimeError(f"solver returned invalid JSON for {call_name}: {result.stdout!r}") from error
    if values.shape != (len(points),) or not np.isfinite(values).all():
        raise RuntimeError(f"solver returned {values.shape}; expected {len(points)} finite scores")
    return values, elapsed


def model_params(name: str) -> dict:
    return {"seed": MODEL_SEED}


def fit_model(name: str, X: np.ndarray, y: np.ndarray):
    model = build_model(name, model_params(name))
    started = time.perf_counter()
    model.fit(X, y)
    return model, time.perf_counter() - started


def dump_metrics(path: Path, model) -> None:
    metrics = getattr(model, "metrics", {})
    write_json(path, metrics)


def run_online_pilot(dataset, pools, pool_meta) -> dict[str, dict]:
    states = {name: {"X": dataset.X.copy(), "y": dataset.y.copy(), "history": [], "best": float(dataset.y.min())}
              for name in MODELS}
    run_root = STUDY / "pilot" / "online_runs"
    call_root = run_root / "solver_calls"
    call_root.mkdir(parents=True, exist_ok=True)
    for name in MODELS:
        (run_root / name).mkdir(parents=True, exist_ok=True)

    for round_index in range(PILOT_ROUNDS):
        round_number = round_index + 1
        selections = []
        for name in MODELS:
            state = states[name]
            model, fit_seconds = fit_model(name, state["X"], state["y"])
            means, stds = model.predict(pools[round_index])
            means, stds = np.asarray(means, dtype=float), np.asarray(stds, dtype=float)
            selected_index = min(range(POOL_SIZE), key=lambda i: (float(means[i]), pool_meta[round_index][i]["candidate_id"]))
            prediction_rows = []
            for point, meta, mean, std in zip(pools[round_index], pool_meta[round_index], means, stds):
                prediction_rows.append({**meta, "predicted_mean": float(mean), "predicted_std": float(std),
                                        **{feature: float(value) for feature, value in zip(FEATURES, point)}})
            write_csv(run_root / name / f"round_{round_number:02d}_predictions.csv", prediction_rows)
            selections.append({"model": name, "fit_seconds": fit_seconds,
                               "point": pools[round_index][selected_index],
                               "meta": pool_meta[round_index][selected_index],
                               "predicted_mean": float(means[selected_index]),
                               "predicted_std": float(stds[selected_index])})
            del model
            print(f"round {round_number}/{PILOT_ROUNDS} {name}: fit {fit_seconds:.1f}s; "
                  f"selected {selections[-1]['meta']['candidate_id']} predicted R {means[selected_index]:.3f}", flush=True)

        unique_points: list[np.ndarray] = []
        unique_index: dict[bytes, int] = {}
        for item in selections:
            key = point_key(item["point"])
            if key not in unique_index:
                unique_index[key] = len(unique_points)
                unique_points.append(item["point"])
        points = np.asarray(unique_points)
        actuals, solver_seconds = solver_call(points, f"online_round_{round_number:02d}", call_root)
        actual_by_key = {key: float(actuals[index]) for key, index in unique_index.items()}
        for item in selections:
            actual = actual_by_key[point_key(item["point"])]
            name = item["model"]
            state = states[name]
            state["X"] = np.vstack([state["X"], item["point"]])
            state["y"] = np.append(state["y"], actual)
            state["best"] = min(state["best"], float(actual))
            state["history"].append({
                "round": round_number,
                "candidate_id": item["meta"]["candidate_id"],
                "source": item["meta"]["source"],
                "predicted_mean": item["predicted_mean"],
                "predicted_std": item["predicted_std"],
                "actual_R": float(actual),
                "best_R_after_round": state["best"],
                "fit_seconds": item["fit_seconds"],
                "solver_batch_seconds": solver_seconds,
                "unique_solver_points_in_batch": len(points),
                **{feature: float(value) for feature, value in zip(FEATURES, item["point"])},
            })
            write_csv(run_root / name / "trajectory.csv", state["history"])
        print(f"round {round_number}/{PILOT_ROUNDS}: solver evaluated {len(points)} unique points for "
              f"{len(selections)} model selections "
              f"in {solver_seconds:.1f}s", flush=True)
    return states


def evaluate_pilot_test(dataset, states, all_pool_points: np.ndarray) -> None:
    test_dir = STUDY / "pilot" / "test_set"
    call_dir = test_dir / "solver_calls"
    prediction_dir = test_dir / "predictions"
    call_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    # Refit every model on its own trajectory before generating or revealing test weights.
    final_models = {}
    for name, state in states.items():
        model, fit_seconds = fit_model(name, state["X"], state["y"])
        dump_metrics(STUDY / "pilot" / "online_runs" / name / "final_fit_metrics.json", model)
        final_models[name] = model
        write_json(STUDY / "pilot" / "online_runs" / name / "final_fit.json", {
            "training_rows": int(len(state["y"])), "training_new_solver_rows": PILOT_ROUNDS,
            "fit_seconds": fit_seconds, "model": name, "seed": MODEL_SEED,
        })

    # Generate the hidden test inputs only after all five final fits are complete.
    rng = np.random.default_rng(TEST_SEED)
    elite = np.argsort(dataset.y, kind="stable")[:100]
    anchors = dataset.X[elite]
    anchor_ids = tuple(dataset.ids[index] for index in elite)
    blocked = {point_key(row) for row in dataset.X}
    blocked.update(point_key(row) for row in all_pool_points)
    test_X, test_meta = generate_mixed_pool(PILOT_TEST_SIZE, rng, anchors, anchor_ids, blocked, 0)

    for name, model in final_models.items():
        means, stds = model.predict(test_X)
        means, stds = np.asarray(means, dtype=float), np.asarray(stds, dtype=float)
        write_csv(prediction_dir / f"{name}.csv", [
            {**meta, "predicted_mean": float(mean), "predicted_std": float(std),
             **{feature: float(value) for feature, value in zip(FEATURES, point)}}
            for point, meta, mean, std in zip(test_X, test_meta, means, stds)
        ])
    final_models.clear()

    # Reveal/save the test inputs only after all final pilot models have been fitted and have predicted.
    write_csv(test_dir / "weights.csv", [
        {**meta, **{feature: float(value) for feature, value in zip(FEATURES, point)}}
        for point, meta in zip(test_X, test_meta)
    ])
    truths = []
    elapsed_total = 0.0
    for start in range(0, PILOT_TEST_SIZE, 50):
        stop = min(start + 50, PILOT_TEST_SIZE)
        scores, elapsed = solver_call(test_X[start:stop], f"test_{start + 1:03d}_{stop:03d}", call_dir)
        elapsed_total += elapsed
        truths.extend(float(value) for value in scores)
    truth = np.asarray(truths, dtype=float)
    write_csv(test_dir / "solver_results.csv", [
        {"candidate_id": meta["candidate_id"], "actual_R": float(value)}
        for meta, value in zip(test_meta, truth)
    ])

    summary = []
    for name in MODELS:
        rows = states[name]
        prediction_rows = list(csv.DictReader((prediction_dir / f"{name}.csv").open(encoding="utf-8", newline="")))
        pred = np.asarray([float(row["predicted_mean"]) for row in prediction_rows])
        selected = int(np.argmin(pred))
        low_mask = truth <= LOW_R_THRESHOLD
        summary.append({
            "model": name,
            "pilot_best_R": min(float(dataset.y.min()), *(float(row["actual_R"]) for row in rows["history"])),
            "initial_best_R": float(dataset.y.min()),
            "test_overall_MAE": float(np.mean(np.abs(pred - truth))),
            "test_low_region_n_R_le_115_68": int(low_mask.sum()),
            "test_low_region_MAE": float(np.mean(np.abs(pred[low_mask] - truth[low_mask]))) if low_mask.any() else "",
            "test_predicted_best_actual_R": float(truth[selected]),
            "test_oracle_best_R": float(truth.min()),
            "test_selection_regret": float(truth[selected] - truth.min()),
            "test_best_candidate_id": test_meta[selected]["candidate_id"],
        })
    write_csv(STUDY / "pilot" / "results" / "model_comparison.csv", summary)
    write_json(STUDY / "pilot" / "results" / "run_summary.json", {
        "status": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_rounds_per_model": PILOT_ROUNDS,
        "models": list(MODELS),
        "online_model_selections": PILOT_ROUNDS * len(MODELS),
        "online_unique_solver_points": sum(
            len({row["candidate_id"] for state in states.values() for row in state["history"]
                 if int(row["round"]) == round_number})
            for round_number in range(1, PILOT_ROUNDS + 1)),
        "online_duplicate_model_selections": (PILOT_ROUNDS * len(MODELS)) - sum(
            len({row["candidate_id"] for state in states.values() for row in state["history"]
                 if int(row["round"]) == round_number})
            for round_number in range(1, PILOT_ROUNDS + 1)),
        "pilot_test_points": PILOT_TEST_SIZE,
        "online_solver_invocations": PILOT_ROUNDS,
        "test_solver_invocations": (PILOT_TEST_SIZE + 49) // 50,
        "test_solver_seconds": elapsed_total,
        "training_dataset_sha256": dataset.sha256,
        "solver_sha256": sha256(SOLVER),
        "selection_rule": "minimum predicted_mean; ties by candidate_id",
        "spearman_computed": False,
        "test_low_region_threshold_R": LOW_R_THRESHOLD,
    })


def main() -> None:
    if not SOLVER.is_file():
        raise SystemExit(f"solver executable missing: {SOLVER}")
    result_file = STUDY / "pilot" / "results" / "run_summary.json"
    if result_file.exists():
        raise SystemExit(f"pilot output already exists; refusing to overwrite: {result_file}")
    dataset = load_dataset(ROOT / "dataset_same8")
    print(f"Loaded {len(dataset.y)} verified training rows; best R={dataset.y.min():.6f}", flush=True)
    pools, metadata = generate_candidate_pools(dataset)
    all_pool_points = np.vstack(pools)
    write_json(STUDY / "protocol" / "pilot_manifest.json", {
        "protocol": "online-surrogate-pilot-v1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "models": list(MODELS),
        "model_seed": MODEL_SEED,
        "candidate_pool_count": POOL_COUNT,
        "candidate_pool_size": POOL_SIZE,
        "pilot_rounds_per_model": PILOT_ROUNDS,
        "selection": "each model selects its lowest predicted_mean from that round's shared pool; ties by candidate_id",
        "trajectories": "each model receives only its own solver feedback",
        "test_protocol": "after all models finish 5 rounds, refit each on the initial 2048 plus its own five results; only then generate and reveal a separate 100-point test set for prediction, before obtaining solver labels",
        "initial_dataset_sha256": dataset.sha256,
        "solver_sha256": sha256(SOLVER),
        "solver_executable": str(SOLVER.relative_to(ROOT)),
        "seed": {"candidate_pools": POOL_SEED, "pilot_test": TEST_SEED},
        "metrics": ["overall MAE", "MAE where actual R <= 115.68", "test selected-point R", "test selection regret"],
        "spearman": "not computed",
        "candidate_generation": {
            "global": "50 Dirichlet(1,...,1) samples per building per pool",
            "local": "50 convex perturbations around the 100 lowest-R initial rows per pool, alpha uniform [0.10,0.25]",
            "pool_seed": POOL_SEED,
        },
    })
    with threadpool_limits(limits=1):
        states = run_online_pilot(dataset, pools, metadata)
        evaluate_pilot_test(dataset, states, all_pool_points)
    print(f"Pilot complete: {STUDY / 'pilot' / 'results' / 'model_comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()
