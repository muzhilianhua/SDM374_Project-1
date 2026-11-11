"""Generate the frozen shared candidate pools for the V3.1 model comparison."""
from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
OUT = STUDY / "pilot" / "online_v3_1"
DB = STUDY / "solver_database" / "solver_observations.sqlite3"
REFERENCE = json.loads((STUDY / "offline_direction_reference.json").read_text(encoding="utf-8-sig"))

import sys

sys.path.insert(0, str(ROOT / "src"))
from sdm.data import load_dataset
from sdm.simplex import FEATURES, sample

REPLICATES = 8
ROUNDS = 10
POOL_SIZE = 300
GLOBAL_COUNT = 10
LOCAL_COUNT = POOL_SIZE - GLOBAL_COUNT
SHORTLIST_TARGET = LOCAL_COUNT // 2
ANCHOR_ROW_COUNTS = (116, 87, 87)
STEP_COUNTS = {**{s / 100: 36 for s in range(1, 7)}, 0.07: 20, 0.08: 19, 0.09: 18, 0.10: 17}
STEPS = tuple(STEP_COUNTS)
SEEDS = tuple(73101 + i for i in range(REPLICATES))
ANCHOR_SCORES = (113.44401032605674, 112.90731669729576, 113.9512070872588)
ANCHOR_NAMES = ("v3_best_113_444", "v2_best_112_907", "v2_diverse_113_951")
TOL = 1e-10


def key(point: np.ndarray) -> bytes:
    return np.round(np.asarray(point, dtype=np.float64), 12).tobytes()


def read_database_points() -> set[bytes]:
    points: set[bytes] = set()
    if DB.exists():
        with sqlite3.connect(DB) as conn:
            for row in conn.execute("SELECT weight_key FROM solver_observations"):
                points.add(key(np.asarray([float(v) for v in row[0].split(",")])) )
    return points


def database_anchor(score: float) -> tuple[np.ndarray, float]:
    cols = ",".join(FEATURES)
    with sqlite3.connect(DB) as conn:
        row = conn.execute(
            f"SELECT {cols}, actual_R FROM solver_observations WHERE ABS(actual_R-?) < 1e-8 ORDER BY ABS(actual_R-?) LIMIT 1",
            (score, score),
        ).fetchone()
    if row is None:
        raise RuntimeError(f"Could not find solver anchor with R={score}")
    return np.asarray(row[:21], dtype=float), float(row[21])


def make_direction_bank() -> np.ndarray:
    rng = np.random.default_rng(int(REFERENCE["direction_generation"]["seed"]))
    count = int(REFERENCE["direction_generation"]["count"])
    directions = rng.normal(size=(count, 3, 7))
    directions -= directions.mean(axis=2, keepdims=True)
    directions = directions.reshape(count, 21)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return directions


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate_one_pool(
    rep: int,
    round_number: int,
    seed: int,
    anchors: list[np.ndarray],
    directions: np.ndarray,
    direction_ids: list[int],
    training_keys: set[bytes],
    solver_keys: set[bytes],
    study_broad_global_keys: set[bytes],
) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    in_pool = set(training_keys) | set(solver_keys)
    rows: list[dict] = []

    # Exact row and column margins keep every agreed anchor share and step count.
    anchor_slots = np.repeat(np.arange(3), ANCHOR_ROW_COUNTS)
    step_slots = np.concatenate([np.repeat(step, STEP_COUNTS[step]) for step in STEPS])
    rng.shuffle(anchor_slots)
    rng.shuffle(step_slots)
    cells: dict[tuple[int, float], list[int]] = {}
    for anchor_index, step in zip(anchor_slots, step_slots):
        cells.setdefault((int(anchor_index), float(step)), []).append(len(cells.get((int(anchor_index), float(step)), [])))

    # Build valid exact shortlist rays by anchor and scheduled step. Select no more than
    # half the local slots; all infeasible, duplicate, or excess slots use broad directions.
    shortlist_options: list[tuple[int, float, int, np.ndarray]] = []
    for (anchor_index, step), slots in cells.items():
        feasible = []
        for direction_id in direction_ids:
            point = anchors[anchor_index] + step * directions[direction_id]
            if point.min() < -TOL or np.max(np.abs(point.reshape(3, 7).sum(axis=1) - 1)) > TOL:
                continue
            if key(point) in in_pool:
                continue
            feasible.append((direction_id, point))
        rng.shuffle(feasible)
        shortlist_options.extend((anchor_index, step, direction_id, point) for direction_id, point in feasible[: len(slots)])

    if len(shortlist_options) > SHORTLIST_TARGET:
        keep = rng.choice(len(shortlist_options), size=SHORTLIST_TARGET, replace=False)
        selected_options = [shortlist_options[i] for i in keep]
    else:
        selected_options = shortlist_options
    selected_by_cell: dict[tuple[int, float], list[tuple[int, np.ndarray]]] = {}
    for anchor_index, step, direction_id, point in selected_options:
        selected_by_cell.setdefault((anchor_index, step), []).append((direction_id, point))

    for (anchor_index, step), slots in cells.items():
        chosen = selected_by_cell.get((anchor_index, step), [])
        rng.shuffle(chosen)
        for direction_id, point in chosen:
            in_pool.add(key(point))
            rows.append({
                "candidate_id": f"v31r{rep:02d}t{round_number:02d}l{len(rows)+1:03d}",
                "source": "offline_shortlist",
                "anchor_id": ANCHOR_NAMES[anchor_index],
                "direction_id": direction_id,
                "step": step,
                **{feature: float(value) for feature, value in zip(FEATURES, point)},
            })
        for _ in range(len(slots) - len(chosen)):
            for attempt in range(10000):
                target = sample(1, rng)[0]
                distance = float(np.linalg.norm(target - anchors[anchor_index]))
                if distance + TOL < step:
                    continue
                point = anchors[anchor_index] + (step / distance) * (target - anchors[anchor_index])
                point_key = key(point)
                if point_key in in_pool or point_key in study_broad_global_keys:
                    continue
                in_pool.add(point_key)
                study_broad_global_keys.add(point_key)
                rows.append({
                    "candidate_id": f"v31r{rep:02d}t{round_number:02d}l{len(rows)+1:03d}",
                    "source": "broad_feasible_direction",
                    "anchor_id": ANCHOR_NAMES[anchor_index],
                    "direction_id": "",
                    "step": step,
                    **{feature: float(value) for feature, value in zip(FEATURES, point)},
                })
                break
            else:
                raise RuntimeError(f"Could not fill local slot at anchor {anchor_index}, step {step}")

    # Global candidates are independent simplex draws and are unique across the full study.
    for _ in range(GLOBAL_COUNT):
        for attempt in range(10000):
            point = sample(1, rng)[0]
            point_key = key(point)
            if point_key in in_pool or point_key in study_broad_global_keys:
                continue
            in_pool.add(point_key)
            study_broad_global_keys.add(point_key)
            rows.append({
                "candidate_id": f"v31r{rep:02d}t{round_number:02d}g{sum(r['source']=='global' for r in rows)+1:02d}",
                "source": "global",
                "anchor_id": "",
                "direction_id": "",
                "step": "",
                **{feature: float(value) for feature, value in zip(FEATURES, point)},
            })
            break
        else:
            raise RuntimeError("Could not generate a unique global point")

    rng.shuffle(rows)
    return rows, {
        "replicate": rep,
        "round": round_number,
        "seed": seed,
        "pool_size": len(rows),
        "global_count": sum(row["source"] == "global" for row in rows),
        "local_count": sum(row["source"] != "global" for row in rows),
        "shortlist_count": sum(row["source"] == "offline_shortlist" for row in rows),
        "broad_count": sum(row["source"] == "broad_feasible_direction" for row in rows),
        "anchor_counts": {name: sum(row["anchor_id"] == name for row in rows) for name in ANCHOR_NAMES},
        "step_counts": {f"{step:.2f}": sum(row["step"] == step for row in rows) for step in STEPS},
    }


def main() -> None:
    dataset = load_dataset(ROOT / "dataset_same8")
    anchors = [database_anchor(score)[0] for score in ANCHOR_SCORES]
    directions = make_direction_bank()
    direction_ids = [int(value) for value in REFERENCE["direction_ids"]]
    training_keys = {key(point) for point in dataset.X}
    solver_keys = read_database_points()
    broad_global_keys: set[bytes] = set()
    summary = []
    for rep, seed in enumerate(SEEDS, start=1):
        for round_number in range(1, ROUNDS + 1):
            pool_seed = seed * 100 + round_number
            rows, info = generate_one_pool(
                rep, round_number, pool_seed, anchors, directions, direction_ids,
                training_keys, solver_keys, broad_global_keys,
            )
            pool_dir = OUT / f"replicate_{rep:02d}" / "candidate_pools"
            write_csv(pool_dir / f"round_{round_number:02d}.csv", rows)
            summary.append(info)

    manifest = {
        "experiment": "online_v3_1",
        "stage": "candidate_pool_generation_only",
        "solver_called": False,
        "test_set": None,
        "replicates": REPLICATES,
        "rounds_per_replicate": ROUNDS,
        "pool_size": POOL_SIZE,
        "models": ["local-gp", "gp-ard", "histgb", "xgboost", "boosted-consensus"],
        "cold_start_rows": 2048,
        "anchors": [
            {"name": name, "actual_R": score, "weights": anchor.reshape(3, 7).tolist()}
            for name, score, anchor in zip(ANCHOR_NAMES, ANCHOR_SCORES, anchors)
        ],
        "global_candidates_per_pool": GLOBAL_COUNT,
        "local_candidates_per_pool": LOCAL_COUNT,
        "local_anchor_counts": dict(zip(ANCHOR_NAMES, ANCHOR_ROW_COUNTS)),
        "local_step_counts": {f"{step:.2f}": count for step, count in STEP_COUNTS.items()},
        "shortlist_rule": "Use only valid, unique points on the original 17 shortlist rays; aim for at most half of local slots. Fill all remaining local slots with broad feasibility-conditioned directions.",
        "broad_rule": "Sample a valid simplex target T; for anchor A and step s, use A + (s/||T-A||)*(T-A).",
        "global_rule": "Independent Dirichlet(1,...,1) draws for each building.",
        "excluded": "all 2048 training points and all points already present in the canonical solver database",
        "seeds": list(SEEDS),
        "source_reference": str((STUDY / "offline_direction_reference.json").relative_to(ROOT)),
        "dataset_sha256": hashlib.sha256((ROOT / "dataset_same8" / "train.csv").read_bytes()).hexdigest(),
        "summary": summary,
    }
    (OUT / "candidate_generation_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_csv(OUT / "candidate_generation_summary.csv", summary)
    print(json.dumps({
        "pools": len(summary),
        "candidate_rows": sum(row["pool_size"] for row in summary),
        "shortlist_candidates": sum(row["shortlist_count"] for row in summary),
        "broad_candidates": sum(row["broad_count"] for row in summary),
        "global_candidates": sum(row["global_count"] for row in summary),
        "solver_called": False,
        "manifest": str((OUT / "candidate_generation_manifest.json").relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
