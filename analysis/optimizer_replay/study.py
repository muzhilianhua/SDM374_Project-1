"""Compare online-batch settings on continuous, training-table-only replay oracles."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil

from sdm.config import load_config
from sdm.data import load_dataset
from sdm.paths import REPO_ROOT
from sdm.registry import build_model, build_optimizer
from sdm.replay import ReplayConfig, run_replay
from sdm.runs import PACKAGES, git_state


TRUTHS = {
    "extratrees": {"n_estimators": 128, "seed": 2026},
    "histgb": {"max_iter": 150, "calib_folds": 2, "seed": 2026},
}
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "online-batch.toml"
RESULTS = Path(__file__).resolve().parent / "results"


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--out", type=Path, help="New output directory")
    args = parser.parse_args()
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be distinct non-negative integers")
    replay = ReplayConfig(args.calls, args.batch_size)
    replay.validate()
    config = load_config(args.config)
    if config.optimizer is None or config.optimizer.name != "online-batch":
        parser.error("the experiment must configure the online-batch optimizer")
    if config.model.name in TRUTHS:
        parser.error("choose a policy model distinct from the two replay truth models")
    dataset = load_dataset(config.dataset)

    def make_model():
        return build_model(config.model.name, config.model.params)

    optimizer_name = config.optimizer.name
    base_params = asdict(build_optimizer(optimizer_name, config.optimizer.params).params)
    radius = base_params["radius_ratio"]
    arms = {
        "configured": {},
        "exploit-only": {"exploit_fraction": 1.0},
        "fixed-radius": {"min_radius_ratio": radius, "max_radius_ratio": radius},
    }
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-replay-{args.config.stem}"
    out = args.out or RESULTS / run_id
    out.mkdir(parents=True)
    shutil.copyfile(args.config, out / "config.toml")
    write_json(out / "provenance.json", {
        "run_id": run_id,
        "git": git_state(),
        "dataset": {"version": dataset.version, "sha256": dataset.sha256},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: metadata.version(name) for name in PACKAGES},
        "thread_env": {name: os.environ.get(name) for name in
                       ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "budget": asdict(replay),
        "optimizer_seeds": args.seeds,
        "model": {"name": config.model.name, "params": asdict(make_model().params)},
        "optimizer": {"name": optimizer_name, "base_params": base_params, "arm_overrides": arms},
        "truth_models": TRUTHS,
        "initial_labels": "oracle predictions on all training weights",
        "solver_calls": 0,
    })
    summaries = []
    for truth_name, params in TRUTHS.items():
        oracle = build_model(truth_name, params).fit(dataset.X, dataset.y)

        def evaluate(points):
            return oracle.predict(points)[0]

        initial_labels = evaluate(dataset.X)
        for seed in args.seeds:
            for arm, overrides in arms.items():
                optimizer = build_optimizer(optimizer_name, {**base_params, **overrides, "seed": seed})
                result = run_replay(make_model, optimizer, dataset.X, initial_labels, evaluate, replay)
                filename = f"{truth_name}-{seed}-{arm}.json"
                write_json(out / filename, result.to_dict())
                summaries.append({
                    "truth": truth_name, "seed": seed, "arm": arm, "trajectory": filename,
                    "initial_best": result.initial_best,
                    "first_batch_best": result.rounds[0].batch_best,
                    "best_evaluated": result.best_evaluated,
                    "final_incumbent": result.final_best,
                    "final_radius_ratio": result.rounds[-1].optimizer_state["radius_ratio"],
                })
                print(f"{truth_name} {seed} {arm}: {result.best_evaluated:.4f}", flush=True)
    write_json(out / "summary.json", summaries)
    print(f"results: {out}")


if __name__ == "__main__":
    main()
