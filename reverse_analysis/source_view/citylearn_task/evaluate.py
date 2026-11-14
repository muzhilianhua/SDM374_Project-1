"""Readable reconstruction of ``citylearn_task.evaluate`` validation helpers.

The public wrapper in ``evaluate.py`` uses these functions before constructing
CityLearn.  The larger convenience ``evaluate`` entry point is intentionally
left out of this release; evaluate_probe provides the implemented public path.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from copy import deepcopy
from collections.abc import Mapping
import json
import numpy as np

EVALUATOR_VERSION = "2026.10.13.v2"
TIMING_PROTOCOL = "executed_hours_v2"


@dataclass(frozen=True)
class Scenario:
    id: str
    dataset_dir: Path
    buildings: tuple[str, str, str]
    start: int
    end: int
    seed: int


def _integer(value, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{key} must be an integer")
    return int(value)


def normalize_weights(weights):
    """Accept exactly 21 finite, nonnegative values, flat or 3x7."""
    objects = np.asarray(weights, dtype=object)
    if any(isinstance(v, (bool, np.bool_)) for v in objects.flat):
        raise ValueError("boolean values are not valid weights")
    raw = np.asarray(weights)
    if raw.shape not in ((21,), (3, 7)) and raw.dtype.kind not in "iuf":
        raise ValueError("weights must be 21 numbers or a 3 x 7 numeric matrix")
    if raw.shape not in ((21,), (3, 7)):
        raise ValueError("weights must be 21 numbers or a 3 x 7 numeric matrix")
    w = np.asarray(raw, dtype=np.float64).reshape(3, 7).copy()
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("weights must be finite and nonnegative")
    if (w.sum(axis=1) <= 0).any():
        raise ValueError("each building row must contain a positive weight")
    with np.errstate(over="ignore"):
        sums = w.sum(axis=1)
    for i in range(3):
        if not np.isfinite(sums[i]):
            w[i] /= w[i].max()
            sums[i] = w[i].sum()
    return w / sums[:, None]


def load_scenario(scenario) -> Scenario:
    """Resolve a Scenario, mapping, or JSON path and validate its six fields."""
    if isinstance(scenario, Scenario):
        data = dict(id=scenario.id, dataset_dir=scenario.dataset_dir,
                    buildings=scenario.buildings, start=scenario.start,
                    end=scenario.end, seed=scenario.seed)
        base = Path.cwd()
    elif isinstance(scenario, (str, Path)):
        path = Path(scenario).resolve()
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        base = path.parent
    elif isinstance(scenario, Mapping):
        data = dict(scenario)
        base = Path.cwd()
    else:
        raise ValueError("scenario must be a Scenario, mapping, or JSON file path")
    if not isinstance(data, dict):
        raise ValueError("scenario JSON must contain an object")
    required = {"id", "dataset_dir", "buildings", "start", "end", "seed"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"scenario is missing fields: {sorted(missing)}")
    if not isinstance(data["id"], str) or not data["id"].strip():
        raise ValueError("scenario id must be a nonempty string")
    names = data["buildings"]
    if (not isinstance(names, (list, tuple)) or len(names) != 3
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != 3):
        raise ValueError("buildings must contain exactly three distinct building names")
    start, end, seed = (_integer(data[k], k) for k in ("start", "end", "seed"))
    if not (0 <= start < end):
        raise ValueError("scenario requires 0 <= start < end (inclusive endpoints)")
    if not (0 <= seed < 2**32):
        raise ValueError("seed must be in [0, 2**32)")
    dataset_dir = (base / Path(data["dataset_dir"])).resolve()
    if not (dataset_dir / "schema.json").is_file():
        raise FileNotFoundError(f"dataset schema not found: {dataset_dir / 'schema.json'}")
    return Scenario(str(data["id"]), dataset_dir, tuple(names), start, end, seed)


@lru_cache(maxsize=32)
def _read_schema(path: str, mtime_ns: int, size: int):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _scenario_schema(scenario: Scenario) -> dict:
    path = scenario.dataset_dir / "schema.json"
    schema = deepcopy(_read_schema(str(path), path.stat().st_mtime_ns, path.stat().st_size))
    source_start = int(schema.get("simulation_start_time_step", 0))
    source_end = int(schema["simulation_end_time_step"])
    missing = set(scenario.buildings) - set(schema["buildings"])
    if missing:
        raise ValueError(f"unknown buildings in {path}: {sorted(missing)}")
    if scenario.start < source_start or scenario.end > source_end:
        raise ValueError(f"scenario window is outside dataset range {source_start}..{source_end}")
    if float(schema.get("seconds_per_time_step", 3600)) != 3600:
        raise ValueError("this task requires hourly data (seconds_per_time_step=3600)")
    schema["root_directory"] = str(scenario.dataset_dir)
    schema["random_seed"] = scenario.seed
    schema["buildings"] = {name: schema["buildings"][name] for name in scenario.buildings}
    return schema
