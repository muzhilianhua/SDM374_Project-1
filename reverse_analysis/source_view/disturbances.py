"""Synthetic physical-input perturbations applied only to padded CSV copies."""

from __future__ import annotations

import csv
import hashlib
import io
import math
import tempfile
from pathlib import Path

import numpy as np

DISTURBANCE_VERSION = "2026.10.13.paired_physical_inputs.v1"
HOURS = 168
RHO = 0.8
SHARED_VARIANCE_FRACTION = 0.5


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stream(seed: int, component: int, building: str = ""):
    name_code = int.from_bytes(hashlib.sha256(building.encode("utf-8")).digest()[:8], "little")
    return np.random.default_rng(np.random.SeedSequence([seed, component, name_code]))


def _ar1(rng) -> np.ndarray:
    innovation = rng.standard_normal(HOURS)
    value = np.empty(HOURS, dtype=np.float64)
    value[0] = innovation[0]
    for t in range(1, HOURS):
        value[t] = RHO * value[t - 1] + math.sqrt(1.0 - RHO * RHO) * innovation[t]
    return value


def _outage(seed: int, strength: int):
    anchors = _stream(seed, 11).choice(np.arange(24, 163, 6), size=6, replace=False)
    count, duration = [(0, 0), (1, 2), (3, 3), (6, 6)][strength]
    signal = np.zeros(HOURS, dtype=np.int8)
    events = []
    for anchor in anchors[:count]:
        start = int(anchor)
        signal[start:start + duration] = 1
        events.append({"start_inclusive": start, "end_exclusive": start + duration})
    return signal, [int(v) for v in anchors], events


def _temporary_root(schema: dict) -> Path:
    root = Path(schema["root_directory"]).resolve(strict=True)
    temp = Path(tempfile.gettempdir()).resolve(strict=True)
    if root.parent != temp or not root.name.startswith("citylearn_timing_v2_"):
        raise ValueError("Only padded_schema citylearn_timing_v2_ directories under the OS temp directory may be changed")
    if schema.get("_teacher_disturbance_applied"):
        raise ValueError("Perturbation already applied; create a fresh padded schema to avoid compounding")
    allocated = int(schema.get("simulation_end_time_step", -1)) + 1
    if int(schema.get("simulation_start_time_step", -1)) != 0:
        raise ValueError("Expected a zero-based padded schema")
    if allocated < HOURS + 1 or int(schema.get("episode_time_steps", -1)) != allocated:
        raise ValueError("Expected 168 real hours plus unexecuted buffer rows")
    if float(schema.get("seconds_per_time_step", 3600)) != 3600.0:
        raise ValueError("Only hourly inputs are supported")
    return root


def perturb_schema(schema: dict, family: str, strength: int, seed: int) -> dict:
    if family not in ("outage", "load_pv"):
        raise ValueError("family must be outage or load_pv")
    if isinstance(strength, bool) or not isinstance(strength, (int, np.integer)) or strength not in (0, 1, 2, 3):
        raise ValueError("strength must be integer 0, 1, 2, or 3")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    strength, seed = int(strength), int(seed)
    root = _temporary_root(schema)
    buildings = schema["buildings"]
    if len(buildings) != 3:
        raise ValueError("Expected the three-building task schema")

    signal, anchors, events = _outage(seed, strength) if family == "outage" else (
        np.zeros(HOURS, dtype=np.int8), [], []
    )
    amplitude = [0.0, 0.1, 0.25, 0.5][strength] if family == "load_pv" else 0.0
    shared_nsl = _ar1(_stream(seed, 101))
    shared_pv = _ar1(_stream(seed, 201))
    allocated = int(schema["simulation_end_time_step"]) + 1
    prepared, entries, paths = [], {}, set()

    for name, building in buildings.items():
        path = Path(building["energy_simulation"])
        path = (root / path).resolve(strict=True) if not path.is_absolute() else path.resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError(f"{name}: energy input is outside temporary padded schema")
        if path in paths:
            raise ValueError("Each building must have its own temporary energy CSV for independent factors")
        paths.add(path)
        original = path.read_bytes()
        reader = csv.DictReader(io.StringIO(original.decode("utf-8-sig"), newline=""))
        fields = list(reader.fieldnames or [])
        if len(set(fields)) != len(fields):
            raise ValueError("Duplicate CSV field names")
        if not {"non_shiftable_load", "solar_generation"}.issubset(fields):
            raise ValueError(f"{name}: missing actual NSL/PV input columns")
        rows = list(reader)
        if len(rows) != allocated:
            raise ValueError(f"{name}: CSV length differs from padded allocation")
        if any(rows[t] != rows[HOURS - 1] for t in range(HOURS, allocated)):
            raise ValueError("Expected original buffer to contain only copies of the last real row")
        if "power_outage" not in fields:
            fields.append("power_outage")

        local_nsl = _ar1(_stream(seed, 102, name))
        local_pv = _ar1(_stream(seed, 202, name))
        z_nsl = math.sqrt(SHARED_VARIANCE_FRACTION) * shared_nsl + math.sqrt(1 - SHARED_VARIANCE_FRACTION) * local_nsl
        z_pv = math.sqrt(SHARED_VARIANCE_FRACTION) * shared_pv + math.sqrt(1 - SHARED_VARIANCE_FRACTION) * local_pv
        factor_nsl = 1.0 + amplitude * np.tanh(z_nsl)
        factor_pv = 1.0 - 0.5 * amplitude * (1.0 + np.tanh(z_pv))
        real = [dict(row) for row in rows[:HOURS]]
        for t, row in enumerate(real):
            for column, factor in (("non_shiftable_load", factor_nsl[t]), ("solar_generation", factor_pv[t])):
                value = float(row[column])
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"{name}: {column} must be finite and nonnegative")
                if amplitude:
                    row[column] = repr(value * float(factor))
            row["power_outage"] = str(int(signal[t]))
        new_rows = real + [dict(real[-1]) for _ in range(allocated - HOURS)]
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(new_rows)
        payload = stream.getvalue().encode("utf-8")
        entries[name] = {
            "input_file": path.name,
            "input_sha256_before": _sha(original),
            "input_sha256_after": _sha(payload),
            "nsl_factor_168": factor_nsl.tolist(),
            "pv_factor_168": factor_pv.tolist(),
            "nsl_latent_168": z_nsl.tolist(),
            "pv_latent_168": z_pv.tolist(),
            "columns_changed": ["power_outage"] + (["non_shiftable_load", "solar_generation"] if amplitude else []),
        }
        prepared.append((path, payload, building))

    for path, payload, building in prepared:
        path.write_bytes(payload)
        building["power_outage"] = {"simulate_power_outage": True, "stochastic_power_outage": False}
    schema["_teacher_disturbance_applied"] = DISTURBANCE_VERSION
    return {
        "version": DISTURBANCE_VERSION,
        "family": family,
        "strength": strength,
        "seed": seed,
        "synthetic_stress_test_not_empirical_reliability": True,
        "real_hours": HOURS,
        "allocated_hours": allocated,
        "buffer_policy": "copy transformed last real row; do not execute/score buffer",
        "outage_scope": "shared district schedule; physically each building islands independently",
        "outage_anchor_starts_6": anchors,
        "outage_events": events,
        "outage_168": signal.tolist(),
        "outage_hours": int(signal.sum()),
        "load_pv_amplitude": amplitude,
        "ar1_rho": RHO,
        "shared_latent_variance_fraction": SHARED_VARIANCE_FRACTION,
        "numpy_version": np.__version__,
        "perturbation_policy": "keep environment/equipment seed fixed across strengths and weights",
        "buildings": entries,
    }
