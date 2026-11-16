"""Generate a training-only candidate batch for online policy research.

This module deliberately has no access to the course solver or historical online results. It is a
candidate generator for review, not a replacement for ``sdm online``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sdm.data import load_dataset
from sdm.paths import REPO_ROOT
from sdm.registry import build_model
from sdm.simplex import DIM, FEATURES, N_BUILDINGS, N_METRICS, check_weights, sample


def _key(row: np.ndarray) -> tuple[float, ...]:
    return tuple(np.round(row, 12))


def _novel(rows: np.ndarray, reference: np.ndarray, minimum: float = 1e-9) -> np.ndarray:
    """Drop exact or numerically duplicate rows from a pool and the training table."""
    seen = {_key(row) for row in reference}
    result = []
    for row in rows:
        key = _key(row)
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return np.asarray(result, dtype=float).reshape(-1, DIM)


def _dirichlet_around(parents: np.ndarray, n: int, concentration: float,
                      rng: np.random.Generator) -> np.ndarray:
    """Sample each building around a randomly selected elite parent."""
    chosen = parents[rng.integers(0, len(parents), size=n)].reshape(n, N_BUILDINGS, N_METRICS)
    alpha = np.maximum(chosen * concentration, 0.02)
    return np.stack(
        [rng.dirichlet(alpha[i, building], size=1)[0]
         for i in range(n) for building in range(N_BUILDINGS)],
        axis=0,
    ).reshape(n, N_BUILDINGS, N_METRICS).reshape(n, DIM)


def _dirichlet_around_centroid(centroid: np.ndarray, n: int, concentration: float,
                               rng: np.random.Generator) -> np.ndarray:
    alpha = np.maximum(centroid.reshape(N_BUILDINGS, N_METRICS) * concentration, 0.02)
    return np.stack(
        [rng.dirichlet(alpha[building], size=n) for building in range(N_BUILDINGS)], axis=1
    ).reshape(n, DIM)


def _sparse_simplex_sample(n: int, concentration: float,
                           rng: np.random.Generator) -> np.ndarray:
    """Sample sparse points to probe simplex boundaries without naming a preferred objective."""
    return rng.dirichlet(np.full(N_METRICS, concentration), size=(n, N_BUILDINGS)).reshape(n, DIM)


def _greedy_quota(pool: np.ndarray, acquisition: np.ndarray, source: str, n: int,
                  reference: np.ndarray, selected: list[np.ndarray]) -> list[dict]:
    order = np.argsort(acquisition)
    seen = {_key(row) for row in reference}
    seen.update(_key(row) for row in selected)
    output = []
    for index in order:
        row = pool[index]
        key = _key(row)
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
        output.append({"weights": row.reshape(N_BUILDINGS, N_METRICS).tolist(),
                       "source": source, "predicted_mean": float(acquisition[index]),
                       "pool_index": int(index)})
        if len(output) == n:
            break
    return output


def generate(*, seed: int = 2026, n: int = 50, pool_size: int = 20_000,
             elite_quantile: float = 0.02, lcb_k: float = 0.25,
             dataset_dir: Path | None = None) -> dict:
    if n < 1 or n > 50:
        raise ValueError("n must be in [1, 50]")
    if not 0 < elite_quantile < 1:
        raise ValueError("elite_quantile must be in (0, 1)")
    dataset = load_dataset(dataset_dir or REPO_ROOT / "dataset_same8")
    rng = np.random.default_rng(seed)
    elite_threshold = float(np.quantile(dataset.y, elite_quantile))
    elite = dataset.X[dataset.y <= elite_threshold]
    # A quantile can contain fewer rows than expected when the table has ties.
    if len(elite) < 4:
        elite = dataset.X[np.argsort(dataset.y)[:4]]
    centroid = elite.reshape(-1, N_BUILDINGS, N_METRICS).mean(axis=0)

    model = build_model("extratrees", {"seed": seed, "n_estimators": 300,
                                       "min_samples_leaf": 2, "max_features": 0.5}).fit(dataset.X, dataset.y)
    per_stream = max(pool_size // 3, n * 20)
    pools = {
        "elite_local": _dirichlet_around(elite, per_stream, 220.0, rng),
        "elite_centroid": _dirichlet_around_centroid(centroid, per_stream, 220.0, rng),
        "boundary_explore": _sparse_simplex_sample(per_stream, 0.15, rng),
        "global_explore": sample(per_stream, rng),
    }
    selected: list[np.ndarray] = []
    # The quota is intentionally fixed by batch size, while the ranking is data-derived.
    local_quota = int(round(n * 0.45))
    centroid_quota = int(round(n * 0.25))
    boundary_quota = int(round(n * 0.20))
    quotas = {"elite_local": local_quota, "elite_centroid": centroid_quota,
              "boundary_explore": boundary_quota,
              "global_explore": n - local_quota - centroid_quota - boundary_quota}
    candidates: list[dict] = []
    for stream_name, pool in pools.items():
        pool = _novel(pool, dataset.X)
        mean, std = model.predict(pool)
        acquisition = mean - lcb_k * std
        rows = _greedy_quota(pool, acquisition, stream_name, quotas[stream_name], dataset.X, selected)
        for row in rows:
            # Store diagnostics after selection without changing the ranking rule.
            weights = np.asarray(row["weights"], dtype=float).reshape(1, DIM)
            predicted_mean, predicted_std = model.predict(weights)
            row["predicted_mean"] = float(predicted_mean[0])
            row["predicted_std"] = float(predicted_std[0])
            row["acquisition"] = float(predicted_mean[0] - lcb_k * predicted_std[0])
        candidates.extend(rows)
    candidates = candidates[:n]
    weights = np.asarray([row["weights"] for row in candidates], dtype=float).reshape(-1, DIM)
    check_weights(weights)
    ratios = elite.reshape(-1, N_BUILDINGS, N_METRICS)[:, :, 5] / np.maximum(
        elite.reshape(-1, N_BUILDINGS, N_METRICS)[:, :, 4], 1e-12
    )
    return {
        "method": "training-only-elite-dirichlet-lcb",
        "dataset": {"version": dataset.version, "sha256": dataset.sha256, "rows": len(dataset.X)},
        "seed": seed,
        "parameters": {"n": n, "pool_size": pool_size, "elite_quantile": elite_quantile,
                       "elite_rows": len(elite), "elite_threshold": elite_threshold, "lcb_k": lcb_k,
                       "model": "extratrees"},
        "derived_summary": {"elite_centroid": centroid.tolist(),
                            "elite_ramping_to_degradation_ratio_median": np.median(ratios, axis=0).tolist(),
                            "elite_ramping_to_degradation_ratio_q10": np.quantile(ratios, 0.1, axis=0).tolist(),
                            "elite_ramping_to_degradation_ratio_q90": np.quantile(ratios, 0.9, axis=0).tolist()},
        "candidates": candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--pool-size", type=int, default=20_000)
    parser.add_argument("--elite-quantile", type=float, default=0.02)
    parser.add_argument("--lcb-k", type=float, default=0.25)
    args = parser.parse_args()
    result = generate(seed=args.seed, n=args.n, pool_size=args.pool_size,
                      elite_quantile=args.elite_quantile, lcb_k=args.lcb_k)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Write bytes so generated artifacts keep LF line endings on Windows as well as Unix.
    args.out.write_bytes((json.dumps(result, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    print(json.dumps({"out": str(args.out), "candidates": len(result["candidates"]),
                      "elite_threshold": result["parameters"]["elite_threshold"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
