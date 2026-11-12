"""Time each candidate surrogate on this machine and fingerprint its predictions (T09, T12).

    python -m sdm ... is not involved: run with PYTHONPATH=src from the repository root:
    python deploy/course-pc/bench_models.py [--configs a.toml b.toml] [--out DIR]

For every config it fits the [model] on the 2048 training rows, predicts a fixed set of 6000 Dirichlet
points (the size of a bayes-pool candidate set) and records fit/predict seconds plus the sha256 of the
raw prediction bytes. Comparing the fingerprints between two machines shows which models are bitwise
deterministic across platforms. Reads the training table only; never calls the evaluator.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from sdm.config import load_config  # noqa: E402
from sdm.data import load_dataset  # noqa: E402
from sdm.registry import build_model  # noqa: E402

DEFAULT_CONFIGS = ("baseline-v32", "histgb-v32", "xgboost-v32", "extratrees-v32", "local-gp-v32", "online-gp-wide")


def probe_points(n: int = 6000, seed: int = 20261112) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.dirichlet(np.ones(7), size=n) for _ in range(3)], axis=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", nargs="*", default=list(DEFAULT_CONFIGS),
                        help="config names in experiments/ (without .toml) or paths")
    parser.add_argument("--out", type=Path, default=REPO.parent / "results")
    args = parser.parse_args()

    points = probe_points()
    report = {"time": dt.datetime.now().isoformat(timespec="seconds"), "host": socket.gethostname(),
              "platform": platform.platform(), "machine": platform.machine(), "python": sys.version,
              "cpu_count": os.cpu_count(), "numpy": np.__version__,
              "probe_sha256": hashlib.sha256(points.tobytes()).hexdigest(), "results": []}
    for name in args.configs:
        path = Path(name) if name.endswith(".toml") else REPO / "experiments" / f"{name}.toml"
        item = {"config": path.stem}
        try:
            config = load_config(path)
            dataset = load_dataset(config.dataset)
            item["model"] = config.model.name
            start = time.perf_counter()
            model = build_model(config.model.name, config.model.params).fit(dataset.X, dataset.y)
            item["fit_seconds"] = round(time.perf_counter() - start, 2)
            start = time.perf_counter()
            mean, std = model.predict(points)
            item["predict_seconds"] = round(time.perf_counter() - start, 2)
            mean, std = np.asarray(mean, dtype=np.float64), np.asarray(std, dtype=np.float64)
            item["mean_sha256"] = hashlib.sha256(mean.tobytes()).hexdigest()
            item["std_sha256"] = hashlib.sha256(std.tobytes()).hexdigest()
            item["mean_min"] = float(mean.min())
            item["mean_head"] = [float(v) for v in mean[:3]]
        except Exception as error:  # keep timing the others
            item["error"] = repr(error)
            item["trace"] = traceback.format_exc()[-1500:]
        report["results"].append(item)
        print(f"{item['config']:<22} fit {item.get('fit_seconds', '-'):>8}s  predict {item.get('predict_seconds', '-'):>6}s"
              f"  {item.get('mean_sha256', item.get('error', ''))[:16]}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    target = args.out / f"bench-{report['host']}-{stamp}.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
