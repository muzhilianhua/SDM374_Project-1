"""Command line entry point (`uv run sdm ...`). Anything a GUI button triggers should go through `execute`."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from . import catalog, frozen_v32
from .config import load_config
from .data import load_dataset
from .evaluate import median_spacing, nearest_distances, out_of_fold, score
from .registry import build_model, build_optimizer, describe, models, optimizers
from .runs import Run, create_run, list_runs, resolve_run
from .simplex import N_BUILDINGS, N_METRICS, check_weights

KINDS = ("train", "evaluate", "optimize")
# Reproduction tolerance: same code, data and seeds should agree to floating-point noise.
RTOL, ATOL = 1e-9, 1e-12


def execute(kind: str, config_path, *, runs_root=None, name: str | None = None,
            n: int = 1, model_run: str | None = None) -> Run:
    """Run one train / evaluate / optimize experiment from a config and record it under runs_root."""
    config_path = Path(config_path)
    config = load_config(config_path)
    dataset = load_dataset(config.dataset)
    # Build plugins before creating the run directory so config mistakes leave nothing behind.
    model = build_model(config.model.name, config.model.params) if model_run is None else None
    optimizer = None
    if kind == "optimize":
        if config.optimizer is None:
            raise ValueError(f"{config_path}: optimize needs an [optimizer] section")
        optimizer = build_optimizer(config.optimizer.name, config.optimizer.params)
    options = {"n": n, "model_run": model_run} if kind == "optimize" else {}
    run = create_run(kind, config_path, name=name or config_path.stem, options=options,
                     dataset=dataset, root=runs_root)
    X, y = dataset.X, dataset.y

    if kind == "train":
        model.fit(X, y)
        run.save_model(model)
        run.write_json("metrics.json", model.metrics)
    elif kind == "evaluate" and config.evaluate.protocol == frozen_v32.PROTOCOL:
        splits = frozen_v32.load_outer_splits(dataset)
        predictions = frozen_v32.out_of_fold(lambda: build_model(config.model.name, config.model.params),
                                             dataset, splits)
        metrics, folds = frozen_v32.score_predictions(predictions, dataset, splits, top_k=config.evaluate.top_k)
        predictions.to_csv(run.path / "predictions.csv", index=False)
        folds.to_csv(run.path / "fold_scores.csv", index=False)
        run.write_json("metrics.json", metrics)
    elif kind == "evaluate":
        settings = config.evaluate
        mean, std = out_of_fold(lambda: build_model(config.model.name, config.model.params), X, y,
                                folds=settings.folds, seed=settings.seed)
        pd.DataFrame({"candidate_id": dataset.ids, "actual": y, "predicted_mean": mean,
                      "predicted_std": std}).to_csv(run.path / "predictions.csv", index=False)
        run.write_json("metrics.json", score(y, mean, std, top_k=settings.top_k))
    elif kind == "optimize":
        if model_run is not None:
            source = resolve_run(model_run, runs_root)
            if source.read_json("meta.json")["dataset"]["sha256"] != dataset.sha256:
                raise ValueError(f"model run {source.id} was trained on a different dataset")
            model = source.load_model()
        else:
            model.fit(X, y)
            run.save_model(model)
        weights = check_weights(optimizer.propose(model, X, y, n))
        mean, std = model.predict(weights)
        distance = nearest_distances(weights, X)
        run.write_json("weights.json", _as_matrices(weights)[0] if n == 1 else _as_matrices(weights))
        run.write_json("proposals.json", [
            {"weights": matrix, "predicted_mean": float(m), "predicted_std": float(s), "nearest_train_distance": float(d)}
            for matrix, m, s, d in zip(_as_matrices(weights), mean, std, distance)
        ])
        run.write_json("metrics.json", {
            "predicted_mean": float(mean[0]), "predicted_std": float(std[0]),
            "nearest_train_distance": float(distance[0]),
            # Typical spacing inside the table, so the distance above has a scale to compare against.
            "train_median_nn_distance": median_spacing(X),
            "best_observed": float(y.min()),
            **{f"optimizer_{key}": value for key, value in optimizer.diagnostics.items()},
            **{f"model_{key}": value for key, value in model.metrics.items()},
        })
    else:
        raise ValueError(f"unknown run kind '{kind}'; expected one of {KINDS}")
    return run


def _as_matrices(weights: np.ndarray) -> list:
    return weights.reshape(-1, N_BUILDINGS, N_METRICS).tolist()


def _compare(old, new, path: str = "") -> list[str]:
    """Paths where two JSON values differ beyond reproduction tolerance."""
    if isinstance(old, dict) and isinstance(new, dict):
        keys = sorted(set(old) | set(new))
        return [m for key in keys for m in _compare(old.get(key), new.get(key), f"{path}.{key}")]
    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        return [m for i, (a, b) in enumerate(zip(old, new)) for m in _compare(a, b, f"{path}[{i}]")]
    if isinstance(old, (int, float)) and isinstance(new, (int, float)) and not isinstance(old, bool):
        return [] if np.isclose(old, new, rtol=RTOL, atol=ATOL) else [f"{path}: {old} -> {new}"]
    return [] if old == new else [f"{path}: {old!r} -> {new!r}"]


def _print_json(data) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def _cmd_list(args) -> int:
    if args.json:
        _print_json({title: {name: {"class": f"{cls.__module__}.{cls.__qualname__}",
                                    "doc": (cls.__doc__ or "").strip(), "params": describe(cls)}
                             for name, cls in sorted(table.items())}
                     for title, table in (("models", models()), ("optimizers", optimizers()))})
        return 0
    for title, table in (("models", models()), ("optimizers", optimizers())):
        print(f"{title}:")
        for name, cls in sorted(table.items()):
            summary = (cls.__doc__ or "").strip().splitlines()[0] if cls.__doc__ else ""
            print(f"  {name:<12} {summary}")
            for field in describe(cls):
                print(f"      {field['name']}: {field['type']} = {field['default']!r}")
    return 0


def _cmd_check_data(args) -> int:
    dataset = load_dataset(args.dataset or load_config(args.config).dataset)
    best = int(np.argmin(dataset.y))
    summary = {"path": str(dataset.path), "version": dataset.version, "sha256": dataset.sha256,
               "rows": len(dataset.y), "ramping_min": float(dataset.y.min()),
               "ramping_median": float(np.median(dataset.y)), "ramping_max": float(dataset.y.max()),
               "best_candidate": dataset.ids[best]}
    if args.json:
        _print_json({**summary, "ok": True})
        return 0
    _print_json(summary)
    print("ok: checksum and weight constraints verified")
    return 0


def _cmd_run(args) -> int:
    options = {"n": args.n, "model_run": args.model_run} if args.command == "optimize" else {}
    run = execute(args.command, args.config, runs_root=args.runs_dir, name=args.name, **options)
    print(f"run: {run.path}")
    _print_json(run.read_json("metrics.json"))
    if args.command == "optimize":
        _print_json(run.read_json("weights.json"))
    return 0


def _cmd_reproduce(args) -> int:
    source = resolve_run(args.run, args.runs_dir)
    meta = source.read_json("meta.json")
    rerun = execute(meta["kind"], source.config_path, runs_root=args.runs_dir or source.path.parent,
                    name=meta["name"], **meta["options"])
    new_meta = rerun.read_json("meta.json")
    for label, old, new in (("git commit", meta["git"], new_meta["git"]),
                            ("packages", meta["packages"], new_meta["packages"])):
        if old != new:
            print(f"note: {label} differs: {old} -> {new}")
    mismatches = [f"{name}{m}" for name in ("metrics.json", "weights.json") if source.has(name)
                  for m in _compare(source.read_json(name), rerun.read_json(name))]
    print(f"original: {source.path}\nrerun:    {rerun.path}")
    if mismatches:
        print("NOT reproduced:\n  " + "\n  ".join(mismatches))
        return 1
    print(f"reproduced: outputs match within rtol={RTOL:g}")
    return 0


def _cmd_runs(args) -> int:
    if args.json:
        records = catalog.run_records(args.runs_dir)
        records["created"] = records["created"].astype(str)
        # NaN is not valid JSON; missing values become null.
        records = records.astype(object).where(records.notna(), None)
        _print_json(records.to_dict(orient="records"))
        return 0
    for run in list_runs(args.runs_dir):
        meta = run.read_json("meta.json")
        if not run.has("metrics.json"):
            summary = "incomplete"
        else:
            metrics = run.read_json("metrics.json")
            keys = [k for k in ("v32_score", "cv_mae", "cv_r2", "mae", "r2", "spearman", "predicted_mean",
                                "nearest_train_distance") if k in metrics]
            summary = "  ".join(f"{k}={metrics[k]:.4g}" for k in keys)
        dirty = "*" if meta["git"].get("dirty") else ""
        print(f"{run.id:<48} {meta['kind']:<9} {(meta['git'].get('commit') or '?')[:7]}{dirty:<2} {summary}")
    return 0


def _cmd_predict(args) -> int:
    model = resolve_run(args.model_run, args.runs_dir).load_model()
    text = sys.stdin.read() if args.weights == "-" else Path(args.weights).read_text(encoding="utf-8-sig")
    weights = check_weights(json.loads(text))
    mean, std = model.predict(weights)
    _print_json({"predicted_mean": mean.tolist(), "predicted_std": std.tolist()})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sdm", description="SDM374 Project 1 experiment runner")
    parser.add_argument("--runs-dir", type=Path, help="where run folders go (default: <repo>/runs)")
    commands = parser.add_subparsers(dest="command", required=True)

    listing = commands.add_parser("list", help="show registered models and optimizers with their parameters")
    listing.add_argument("--json", action="store_true", help="print one JSON document")
    listing.set_defaults(handler=_cmd_list)

    check = commands.add_parser("check-data", help="verify the dataset checksum and constraints")
    source = check.add_mutually_exclusive_group()
    source.add_argument("--dataset", type=Path)
    source.add_argument("-c", "--config", type=Path, default=Path("experiments/baseline.toml"))
    check.add_argument("--json", action="store_true", help="print one JSON document")
    check.set_defaults(handler=_cmd_check_data)

    helps = {"train": "fit the model on the full table and save it",
             "evaluate": "score the model with the config's [evaluate] protocol (kfold or frozen-v32)",
             "optimize": "propose weights by optimizing over a model"}
    for kind in KINDS:
        sub = commands.add_parser(kind, help=helps[kind])
        sub.add_argument("-c", "--config", type=Path, required=True)
        sub.add_argument("--name", help="run label (default: config file name)")
        if kind == "optimize":
            sub.add_argument("-n", type=int, default=1, help="number of proposals")
            sub.add_argument("--model-run", help="reuse the model from a train run instead of refitting")
        sub.set_defaults(handler=_cmd_run)

    again = commands.add_parser("reproduce", help="re-execute a run and compare its outputs")
    again.add_argument("run", help="run directory or run id")
    again.set_defaults(handler=_cmd_reproduce)

    listing = commands.add_parser("runs", help="list recorded runs")
    listing.add_argument("--json", action="store_true", help="print run records as a JSON array")
    listing.set_defaults(handler=_cmd_runs)

    predict = commands.add_parser("predict", help="predict ramping for weights with a trained model")
    predict.add_argument("--model-run", required=True)
    predict.add_argument("--weights", required=True,
                         help="JSON file with one 3x7 matrix or a list of them; '-' reads stdin")
    predict.set_defaults(handler=_cmd_predict)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (ValueError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
