"""Readable reconstruction of the per-week worker and runtime audit.

``read``, ``dump``, ``sha``, ``valid`` and ``rawkey`` are direct semantic
translations.  The audit loop is retained in compact form so the returned
object can be compared with the bytecode-level worker contract.
"""

import hashlib
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEAMS = HERE

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
                    + "\n", encoding="utf-8")
    temp.replace(path)


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def valid(w):
    a = np.asarray(w, dtype=float)
    assert a.shape == (3, 7) and np.isfinite(a).all() and (a >= 0).all()
    assert (a.sum(axis=1) > 0).all()
    return a.tolist()


def rawkey(w):
    return hashlib.sha256(np.asarray(valid(w), dtype="<f8").tobytes()).hexdigest()


def initializer(_lock=None) -> None:
    """The original pins Torch to one thread and disables CityLearn autosizing."""
    import torch
    import warnings
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    warnings.filterwarnings("ignore", message="You are using `torch.load`", category=FutureWarning)

    # The packed controller replaces PV/Battery autosizing with fixed equipment.
    from citylearn.data import DataSet
    from citylearn.energy_model import Battery, PV
    def forbidden_autosize(*args, **kwargs):
        raise RuntimeError("autosize is forbidden in this task")

    PV.autosize = forbidden_autosize
    Battery.autosize = forbidden_autosize
    DataSet.get_pv_sizing_data = lambda *args, **kwargs: None
    DataSet.get_battery_sizing_data = lambda *args, **kwargs: None


def worker(job: dict) -> dict:
    started = time.perf_counter()
    try:
        scenario_path = Path(job["scenario_path"])
        spec = read(scenario_path)
        data = (scenario_path.parent / spec["dataset_dir"]).resolve()
        schema = read(data / "schema.json")
        for name in spec["buildings"]:
            equipment = [
                value for value in schema["buildings"][name].values()
                if isinstance(value, dict) and "autosize" in value
            ]
            assert equipment and all(
                value.get("autosize") is False for value in equipment
            )

        # The packed program intentionally imports the local evaluate.py.
        from evaluate import evaluate_probe
        result = evaluate_probe(
            job["weights"],
            scenario_path,
            family="load_pv",
            strength=0,
            disturbance_seed=0,
            include_building_traces=True,
        )

        net = np.asarray(result["net"], dtype=float)
        assert result["steps"] == len(net) == 168
        assert np.isfinite(net).all()
        expected_ramping = float(np.abs(np.diff(net)).sum())
        assert np.isclose(result["ramping"], expected_ramping, rtol=1e-12)
        runtime_audit = []
        for trace in result["building_traces"]:
            attrs = schema["buildings"][trace["building"]]["dynamics"]["attributes"]
            position = attrs["input_observation_names"].index("indoor_dry_bulb_temperature")
            lo = attrs["input_normalization_minimum"][position]
            hi = attrs["input_normalization_maximum"][position]
            temp = np.asarray(trace["temp"])
            soc = np.asarray(trace["soc"])
            ref = np.asarray(trace["reference_cooling"])
            actual = np.asarray(trace["delivered_cooling"])
            assert all(len(trace[k]) == 168 for k in
                       ("temp", "setpoint", "reference_cooling", "delivered_cooling",
                        "cooling_kwh", "soc", "net"))
            assert np.min(soc) >= -1e-7 and np.max(soc) <= 1.0000001
            assert np.max(ref - actual) < 0.0002 or np.max(actual - 1.1 * ref) < 0.0002
            runtime_audit.append({
                "building": trace["building"],
                "temperature_below_model_range_hours": int(np.count_nonzero(temp < lo)),
                "temperature_above_model_range_hours": int(np.count_nonzero(temp > hi)),
                "model_indoor_range": [lo, hi],
                "min_soc": float(soc.min()),
                "max_soc": float(soc.max()),
                "max_cooling_shortfall": float(np.maximum(ref - actual, 0).max()),
                "max_cooling_upper_excess": float(np.maximum(actual - 1.1 * ref, 0).max()),
            })

        return {
            **job,
            "ramping": result["ramping"],
            "peak": result["peak"],
            "net": net.tolist(),
            "diagnostics": result["diagnostics"],
            "steps": result["steps"],
            "building_traces": result["building_traces"],
            "runtime_audit": runtime_audit,
            "runtime_seconds": time.perf_counter() - started,
            "completed_at": stamp(),
            "controller_version": "2026.10.15.bounded_single_accounting_stable_spans.v2",
        }
    except Exception:
        return {
            **job,
            "error": traceback.format_exc(),
            "runtime_seconds": time.perf_counter() - started,
        }
