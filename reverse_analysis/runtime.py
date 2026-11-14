"""Run only the committed readable modules and bundled scenario/model assets."""
from __future__ import annotations

import concurrent.futures
import contextlib
import hashlib
import importlib
import importlib.metadata
import json
import multiprocessing
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source_view"
VENDOR = HERE / "vendor"
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
sys.path[:0] = [str(SOURCE), str(VENDOR)]


def verify_assets():
    """Check the committed original asset bytes before any evaluation."""
    manifest = json.loads((HERE / "assets_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = HERE / item["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"Missing or modified release asset: {item['path']}")
    return len(manifest["files"])


def environment_info():
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "scipy", "pandas", "scikit-learn", "CityLearn", "torch")}
    modules = {}
    for name in ("engine", "evaluate", "controller", "initialization", "disturbances",
                 "citylearn_task.evaluate", "citylearn_task.lower", "citylearn_task.temporal",
                 "citylearn_task.env", "citylearn.citylearn"):
        path = Path(importlib.import_module(name).__file__).resolve()
        if not path.is_relative_to(HERE):
            raise RuntimeError(f"Unexpected module outside this release: {name}: {path}")
        modules[name] = path.relative_to(HERE).as_posix()
    return dict(python=sys.version.split()[0], versions=versions, modules=modules)


def initialize():
    from engine import initializer
    initializer(None)


def trace_week(job):
    import numpy as np
    from engine import worker

    weights, week = job
    with contextlib.redirect_stdout(sys.stderr):
        result = worker(dict(weights=weights, scenario_path=str(
            HERE / "environment/scenarios" / f"week_{week:02}.json")))
    if "error" in result:
        raise RuntimeError(result["error"])
    nets = np.asarray([building["net"] for building in result["building_traces"]], dtype=np.float64)
    if nets.shape != (3, 168) or not np.isfinite(nets).all():
        raise RuntimeError("Expected a finite three-building, 168-hour trace")
    return nets


class Solver:
    """Reuse the process pool across all candidates in a batch."""

    def __init__(self, workers=8):
        if workers not in range(1, 9):
            raise ValueError("workers must be between 1 and 8")
        self.pool = None
        if workers == 1:
            initialize()
        else:
            self.pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize)

    def evaluate(self, weights):
        import numpy as np
        from citylearn_task.evaluate import normalize_weights

        # Validate without replacing raw weights: the original path normalizes internally.
        normalize_weights(weights)
        jobs = [(weights, week) for week in range(1, 9)]
        nets = np.asarray(list(self.pool.map(trace_week, jobs) if self.pool else map(trace_week, jobs)))
        weekly = np.abs(np.diff(nets.sum(axis=1), axis=1)).sum(axis=1)
        return dict(R=float(weekly.mean()), weekly_R=weekly.tolist(), nets=nets)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self.pool:
            self.pool.shutdown(wait=True, cancel_futures=True)
