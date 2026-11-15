"""Independent additive-ARD-GP online validation using only the original training table."""
from __future__ import annotations

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Hyperparameter, Kernel, Matern, WhiteKernel

ROOT = Path(__file__).resolve().parent
DIM = 21
BLOCK = 7
WEIGHT_COLUMNS = [f"b{b}_{name}" for b in range(1, 4) for name in
                  ("cost", "carbon", "peak", "discomfort", "degradation", "ramping", "resilience")]
EVALUATION_COLUMNS = ["call", "point", *WEIGHT_COLUMNS, "actual_R", "predicted_mean",
                      "predicted_std_raw", "predicted_std", "slot", "dominant_b1", "dominant_b2",
                      "dominant_b3", "nearest_observed_distance", "timestamp"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl(path: Path, value) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def normalise(weights) -> np.ndarray:
    array = np.asarray(weights, dtype=np.float64).reshape(-1, DIM)
    blocks = array.reshape(-1, 3, BLOCK)
    return (blocks / blocks.sum(axis=2, keepdims=True)).reshape(-1, DIM)


def valid_weight_mask(weights) -> np.ndarray:
    array = np.asarray(weights, dtype=np.float64).reshape(-1, DIM)
    sums = array.reshape(-1, 3, BLOCK).sum(axis=2)
    return (np.isfinite(array).all(axis=1) & (array >= 0).all(axis=1)
            & np.isclose(sums, 1.0, rtol=0, atol=1e-9).all(axis=1))


def validate_weights(weights, batch_limit: int = 50) -> np.ndarray:
    array = np.asarray(weights, dtype=np.float64).reshape(-1, DIM)
    if not 1 <= len(array) <= batch_limit:
        raise ValueError(f"expected 1 to {batch_limit} rows, got {len(array)}")
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("weights must be finite and non-negative")
    array = normalise(array)
    if not valid_weight_mask(array).all():
        raise ValueError("normalised building blocks must each sum to one")
    return array


def dominant_structures(weights) -> np.ndarray:
    array = np.asarray(weights).reshape(-1, 3, BLOCK)
    return array.argmax(axis=2)


def duplicate_mask(candidates: np.ndarray, excluded: np.ndarray, atol: float = 1e-9) -> np.ndarray:
    """Exact L-infinity duplicate test; the tree also catches values across decimal-rounding cells."""
    if len(excluded) == 0:
        return np.zeros(len(candidates), dtype=bool)
    distances, _ = cKDTree(excluded).query(candidates, k=1, p=np.inf, workers=1)
    return distances <= atol


def load_training(config: dict) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    path = ROOT / config["data"]["path"]
    if sha256_file(path) != config["data"]["sha256"]:
        raise RuntimeError("training table SHA-256 differs from config.json")
    table = pd.read_csv(path)
    if len(table) != config["data"]["expected_rows"] or list(table.columns) != ["candidate_id", *WEIGHT_COLUMNS, "ramping"]:
        raise RuntimeError("training table schema or row count differs from the original 2048-row table")
    weights = table[WEIGHT_COLUMNS].to_numpy(dtype=np.float64)
    actual = table["ramping"].to_numpy(dtype=np.float64)
    if not np.isfinite(actual).all() or not valid_weight_mask(weights).all():
        raise RuntimeError("training table contains invalid weights or targets")
    return weights, actual, table


class BuildingARDKernel(Kernel):
    """Three additive seven-dimensional Matern-2.5 kernels plus one global interaction."""

    def __init__(self, length_scale=(0.5,) * 7, global_weight=0.1):
        self.length_scale = length_scale
        self.global_weight = global_weight

    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter("length_scale", "numeric", (0.03, 3.0), len(self.length_scale))

    def __call__(self, X, Y=None, eval_gradient=False):
        kernel = Matern(np.asarray(self.length_scale), length_scale_bounds=(0.03, 3.0), nu=2.5)
        parts = [kernel(X[:, 7 * b:7 * b + 7], None if Y is None else Y[:, 7 * b:7 * b + 7],
                        eval_gradient=eval_gradient) for b in range(3)]
        full_kernel = Matern(np.tile(self.length_scale, 3), length_scale_bounds=(0.03, 3.0), nu=2.5)
        full = full_kernel(X, Y, eval_gradient=eval_gradient)
        weight = self.global_weight
        if eval_gradient:
            matrix = sum(part[0] for part in parts) * (1 - weight) / 3 + weight * full[0]
            gradient = sum(part[1] for part in parts) * (1 - weight) / 3
            gradient += weight * full[1].reshape(len(X), len(X), 3, 7).sum(axis=2)
            return matrix, gradient
        return sum(parts) * (1 - weight) / 3 + weight * full

    def diag(self, X):
        return np.ones(len(X))

    def is_stationary(self):
        return True


class AdditiveARDGP:
    def __init__(self, config: dict, maxiter_override: int | None = None):
        self.config = config
        self.maxiter_override = maxiter_override
        self.diagnostics: dict = {}

    def fit(self, weights: np.ndarray, actual: np.ndarray) -> "AdditiveARDGP":
        settings = self.config["model"]
        self.location_ = float(actual.mean())
        self.scale_ = max(float(actual.std()), 1e-9)
        target = (actual - self.location_) / self.scale_
        signal = BuildingARDKernel((settings["initial_length_scale"],) * 7, settings["global_weight"])
        kernel = signal + WhiteKernel(settings["initial_noise"], noise_level_bounds=tuple(settings["noise_bounds"]))
        optimizer_diagnostics: dict = {}

        def optimizer(objective, theta, bounds):
            result = minimize(objective, theta, method="L-BFGS-B", jac=True, bounds=bounds,
                              options={"maxiter": self.maxiter_override or settings["maxiter"],
                                       "ftol": settings["ftol"], "maxls": settings["maxls"]})
            optimizer_diagnostics.update({
                "optimizer_nit": int(result.nit), "optimizer_success": bool(result.success),
                "optimizer_status": int(result.status), "optimizer_nfev": int(result.nfev),
                "optimizer_gradient_inf_norm": float(np.max(np.abs(result.jac))),
                "optimizer_message": str(result.message),
            })
            return result.x, result.fun

        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.model = GaussianProcessRegressor(
                kernel=kernel, optimizer=optimizer, normalize_y=False, alpha=settings["alpha"],
                n_restarts_optimizer=0).fit(weights, target)
        self.diagnostics = {
            "train_rows": int(len(actual)), "target_location": self.location_, "target_scale": self.scale_,
            "length_scales": np.atleast_1d(self.model.kernel_.k1.length_scale).tolist(),
            "noise": float(self.model.kernel_.k2.noise_level),
            "global_weight": float(self.model.kernel_.k1.global_weight),
            "log_marginal_likelihood": float(self.model.log_marginal_likelihood_value_),
            "fit_seconds": time.perf_counter() - started,
            "warnings": [str(item.message) for item in caught], **optimizer_diagnostics,
        }
        return self

    def predict(self, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        means, standard_deviations = [], []
        chunk = self.config["model"]["prediction_chunk"]
        for start in range(0, len(weights), chunk):
            mean, std = self.model.predict(weights[start:start + chunk], return_std=True)
            means.append(mean * self.scale_ + self.location_)
            standard_deviations.append(std * self.scale_)
        return np.concatenate(means), np.concatenate(standard_deviations)


def build_elites(weights: np.ndarray, actual: np.ndarray, config: dict) -> dict:
    """Derive all structure centres and role donors exclusively from observed rows."""
    structures = dominant_structures(weights)
    order = np.argsort(actual, kind="mergesort")
    best_by_structure = {}
    for index in order:
        key = tuple(int(v) for v in structures[index])
        if key not in best_by_structure:
            best_by_structure[key] = int(index)
    ranked = sorted(best_by_structure.items(), key=lambda item: (actual[item[1]], item[0]))
    count = config["candidates"]["elite_structures"]
    chosen = ranked[:count]
    donor_limit = config["candidates"]["donors_per_building_role"]
    donors = {(building, role): [] for building in range(3) for role in range(7)}
    for index in order:
        for building in range(3):
            key = (building, int(structures[index, building]))
            if len(donors[key]) < donor_limit:
                donors[key].append(weights[index, building * 7:(building + 1) * 7].copy())
    return {
        "weights": np.array([weights[index] for _, index in chosen]),
        "structures": np.array([key for key, _ in chosen], dtype=int),
        "donors": donors,
        "observed_structure_count": len(ranked),
    }


def generate_local(elites: dict, count: int, concentrations: list[float], rng: np.random.Generator) -> np.ndarray:
    centres = elites["weights"]
    output = np.empty((count, DIM), dtype=np.float64)
    for row in range(count):
        centre = centres[rng.integers(len(centres))]
        concentration = float(concentrations[rng.integers(len(concentrations))])
        for building in range(3):
            block = centre[building * 7:(building + 1) * 7]
            alpha = np.maximum(block, 1e-12) * concentration
            output[row, building * 7:(building + 1) * 7] = rng.dirichlet(alpha)
    return output


def generate_structure(elites: dict, count: int, config: dict, rng: np.random.Generator,
                       return_details: bool = False):
    centres = elites["weights"]
    centre_structures = elites["structures"]
    donors = elites["donors"]
    settings = config["candidates"]
    output = np.empty((count, DIM), dtype=np.float64)
    base_indices = np.empty(count, dtype=int)
    changed_rows = []
    for row in range(count):
        base_index = int(rng.integers(len(centres)))
        base_indices[row] = base_index
        base = centres[base_index]
        n_changed = 2 if rng.random() < settings["two_building_probability"] else 1
        changed = rng.choice(3, size=n_changed, replace=False)
        changed_rows.append(changed.copy())
        candidate = base.copy()
        for building_value in changed:
            building = int(building_value)
            old_role = int(centre_structures[base_index, building])
            available = [block for role in range(7) if role != old_role
                         for block in donors[(building, role)]]
            if not available:
                raise RuntimeError("no donor with a different dominant role is available")
            donor = available[int(rng.integers(len(available)))]
            fraction = float(settings["replacement_fractions"][
                rng.integers(len(settings["replacement_fractions"]))])
            mixed = (1 - fraction) * base[building * 7:(building + 1) * 7] + fraction * donor
            concentration = float(settings["structure_concentrations"][
                rng.integers(len(settings["structure_concentrations"]))])
            candidate[building * 7:(building + 1) * 7] = rng.dirichlet(
                np.maximum(mixed, 1e-12) * concentration)
        output[row] = candidate
    if return_details:
        return output, {"base_indices": base_indices, "changed": changed_rows}
    return output


def make_candidates(weights: np.ndarray, actual: np.ndarray, config: dict, call: int,
                    local_stream: int, structure_stream: int,
                    local_count: int | None = None, structure_count: int | None = None) -> dict:
    elites = build_elites(weights, actual, config)
    local_n = config["candidates"]["local"] if local_count is None else local_count
    structure_n = config["candidates"]["structure"] if structure_count is None else structure_count
    local_rng = np.random.default_rng([config["master_seed"], call, local_stream])
    structure_rng = np.random.default_rng([config["master_seed"], call, structure_stream])
    local = generate_local(elites, local_n, config["candidates"]["local_concentrations"], local_rng)
    structural = generate_structure(elites, structure_n, config, structure_rng)
    return {
        "weights": np.vstack([local, structural]),
        "source": np.array(["exploit"] * local_n + ["structure"] * structure_n),
        "structures": dominant_structures(np.vstack([local, structural])),
        "elite_count": len(elites["weights"]),
        "observed_structure_count": elites["observed_structure_count"],
    }


def quotas_for_call(config: dict, call: int, previous_calls: list[dict], batch_size: int) -> dict:
    selection = config["selection"]
    if call == 1:
        quotas = dict(selection["first_round_quotas"])
    else:
        successful = [record for record in previous_calls if record.get("status") == "ok"]
        recent = successful[-selection["improvement_lookback"]:]
        key = "improving_quotas" if any(record.get("improved_global", False) for record in recent) else "stalled_quotas"
        quotas = dict(selection[key])
    if batch_size == config["batch_size"]:
        return quotas
    # Only the synthetic interface smoke test uses another batch size.
    result = {key: int(math.floor(value * batch_size / config["batch_size"])) for key, value in quotas.items()}
    for key in ("structure", "exploit", "explore"):
        if sum(result.values()) >= batch_size:
            break
        result[key] += 1
    return result


class SelectionState:
    def __init__(self, pool: dict, atol: float):
        self.pool = pool
        self.atol = atol
        self.used = np.zeros(len(pool["weights"]), dtype=bool)
        self.points: list[np.ndarray] = []
        self.labels: list[str] = []
        self.indices: list[int] = []
        self.counts = {label: Counter() for label in ("exploit", "structure", "explore")}

    def add(self, index: int, label: str) -> None:
        self.used[index] = True
        self.points.append(self.pool["weights"][index].copy())
        self.labels.append(label)
        self.indices.append(int(index))
        self.counts[label][tuple(int(v) for v in self.pool["structures"][index])] += 1


def greedy_select(state: SelectionState, label: str, eligible: np.ndarray, scores: np.ndarray,
                  requested: int, cap: int, base_separation: float) -> int:
    if requested <= 0 or len(eligible) == 0:
        return 0
    ordered = eligible[np.argsort(scores[eligible], kind="mergesort")]
    picked = 0
    separation = base_separation
    # Relax only after the preceding pass could not fill the requested quota.
    while picked < requested and separation > 1e-12:
        for index_value in ordered:
            if picked >= requested:
                break
            index = int(index_value)
            if state.used[index]:
                continue
            structure = tuple(int(v) for v in state.pool["structures"][index])
            if state.counts[label][structure] >= cap:
                continue
            point = state.pool["weights"][index]
            if state.points:
                selected = np.asarray(state.points)
                if (np.max(np.abs(selected - point), axis=1) <= state.atol).any():
                    continue
                if (np.linalg.norm(selected - point, axis=1) < separation).any():
                    continue
            state.add(index, label)
            picked += 1
        separation *= 0.5
    return picked


def select_batch(pool: dict, mean: np.ndarray, std: np.ndarray, nearest: np.ndarray,
                 spacing: float, quotas: dict, config: dict) -> SelectionState:
    selection = config["selection"]
    state = SelectionState(pool, selection["duplicate_atol"])
    source = pool["source"]
    exploit_eligible = np.flatnonzero(source == "exploit")
    structure_eligible = np.flatnonzero(source == "structure")
    explore_eligible = np.flatnonzero(nearest >= selection["explore_distance_factor"] * spacing)
    scores = {
        "exploit": mean + selection["exploit_std_weight"] * std,
        "structure": mean - selection["structure_std_weight"] * std,
        "explore": mean - selection["explore_std_weight"] * std,
    }
    eligible = {"exploit": exploit_eligible, "structure": structure_eligible, "explore": explore_eligible}
    caps = {"exploit": selection["exploit_structure_cap"],
            "structure": selection["structure_target_cap"],
            "explore": selection["explore_structure_cap"]}
    separation = selection["selected_distance_factor"] * spacing

    def take(label: str, amount: int) -> int:
        return greedy_select(state, label, eligible[label], scores[label], amount, caps[label], separation)

    requested = dict(quotas)
    got = take("exploit", requested["exploit"])
    requested["structure"] += requested["exploit"] - got
    got = take("structure", requested["structure"])
    deficit = requested["structure"] - got
    if deficit:
        take("exploit", deficit)
    got = take("explore", requested["explore"])
    deficit = requested["explore"] - got
    if deficit:
        transferred = take("structure", deficit)
        if transferred < deficit:
            take("exploit", deficit - transferred)
    return state


def calibration_coefficient(evaluations: pd.DataFrame, config: dict) -> float:
    if evaluations.empty:
        return 1.0
    recent = evaluations.tail(config["calibration"]["history"])
    raw = np.maximum(recent["predicted_std_raw"].to_numpy(dtype=float), 1e-12)
    residual = recent["actual_R"].to_numpy(dtype=float) - recent["predicted_mean"].to_numpy(dtype=float)
    value = float(np.sqrt(np.mean((residual / raw) ** 2)))
    return float(np.clip(value, config["calibration"]["minimum"], config["calibration"]["maximum"]))


def read_attempts(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare_resume(run_dir: Path, accept_lost: bool) -> tuple[list[dict], np.ndarray, int]:
    path = run_dir / "attempts.jsonl"
    attempts = read_attempts(path)
    latest = {}
    for event in attempts:
        latest[int(event["call"])] = event
    unresolved = sorted(call for call, event in latest.items() if event["status"] == "started")
    if unresolved:
        if not accept_lost:
            raise RuntimeError(f"calls {unresolved} have 'started' but no outcome; use --accept-lost-attempt explicitly")
        for call in unresolved:
            append_jsonl(path, {"status": "lost", "call": call, "timestamp": utc_now()})
        attempts = read_attempts(path)
    sent = [row for event in attempts if event["status"] == "started" for row in event["weights"]]
    excluded = np.asarray(sent, dtype=float).reshape(-1, DIM)
    next_call = max((int(event["call"]) for event in attempts), default=0) + 1
    return attempts, excluded, next_call


def load_previous_calls(run_dir: Path) -> list[dict]:
    records = []
    for path in sorted((run_dir / "calls").glob("call_*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return sorted(records, key=lambda record: record["call"])


def load_evaluations(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "evaluations.csv"
    if not path.exists():
        return pd.DataFrame(columns=EVALUATION_COLUMNS)
    return pd.read_csv(path)


def append_evaluations(run_dir: Path, rows: list[dict]) -> None:
    path = run_dir / "evaluations.csv"
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVALUATION_COLUMNS)
        if path.stat().st_size == 0:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def prediction_metrics(predicted: np.ndarray, std: np.ndarray, actual: np.ndarray) -> dict:
    error = predicted - actual
    correlation = spearmanr(predicted, actual).statistic if len(actual) > 1 else float("nan")
    return {
        "mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bias_predicted_minus_actual": float(np.mean(error)),
        "spearman": None if not np.isfinite(correlation) else float(correlation),
        "std_coverage_95": float(np.mean(np.abs(error) <= 1.96 * std)),
    }


def write_metrics(run_dir: Path, calls: list[dict]) -> None:
    columns = ["call", "status", "train_rows", "calibration", "fit_seconds", "requested_exploit",
               "requested_structure", "requested_explore", "actual_exploit", "actual_structure",
               "actual_explore", "candidate_structures", "batch_min", "running_best_online",
               "mae", "rmse", "bias_predicted_minus_actual", "spearman", "std_coverage_95"]
    with (run_dir / "metrics_by_call.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in calls:
            metrics = record.get("metrics") or {}
            row = {key: record.get(key) for key in ("call", "status", "train_rows", "calibration",
                                                    "batch_min", "running_best_online")}
            row["fit_seconds"] = record["model"]["fit_seconds"]
            row["candidate_structures"] = len({tuple(value) for value in record["dominant_structures"]})
            for label in ("exploit", "structure", "explore"):
                row[f"requested_{label}"] = record["requested_quotas"][label]
                row[f"actual_{label}"] = record["slot_counts"].get(label, 0)
            row.update(metrics)
            writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


class SolverError(RuntimeError):
    def __init__(self, message: str, stderr: str = "", stdout: str = "", returncode: int | None = None):
        super().__init__(message)
        self.stderr = stderr
        self.stdout = stdout
        self.returncode = returncode


def parse_solver_output(stdout: str, stderr: str = "") -> list[float]:
    """Parse ordered JSON-lines responses, tolerating non-JSON serve log lines."""
    values = []
    for line in stdout.strip().splitlines():
        if not line.strip():
            continue
        try:
            response = json.loads(line)
        except ValueError:
            continue
        if not isinstance(response, dict):
            continue
        if "R" in response:
            raw = response["R"]
            candidates = raw if isinstance(raw, list) else [raw]
            if not candidates or not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                                         and math.isfinite(value) for value in candidates):
                raise SolverError("solver returned a missing or non-finite R value", stderr, stdout)
            values.extend(float(value) for value in candidates)
        elif isinstance(response.get("error"), str):
            raise SolverError(f"solver reported an error: {response['error']}", stderr, stdout)
    if not values:
        raise SolverError(f"no R response in solver output: {stdout[-500:]!r}", stderr, stdout)
    return values


class SolverEvaluator:
    kind = "solver"

    def __init__(self, config: dict, command: list[str] | None = None, verify_hashes: bool = True):
        settings = config["solver"]
        self.outer_executable = Path(settings["path"]).resolve()
        self.executable = Path(settings["inner_path"]).resolve()
        self.timeout = float(settings["timeout_seconds"])
        self.workers = int(settings["workers"])
        if verify_hashes:
            outer_hash = sha256_file(self.outer_executable)
            inner_hash = sha256_file(self.executable)
            if outer_hash != settings["sha256"]:
                raise RuntimeError(f"outer solver SHA-256 {outer_hash} does not match the pinned value")
            if inner_hash != settings["inner_sha256"]:
                raise RuntimeError(f"embedded solver SHA-256 {inner_hash} does not match the outer resource")
        self.command = list(command) if command else [
            str(self.executable), "--serve", "--workers", str(self.workers)]

    def evaluate(self, weights: np.ndarray) -> tuple[np.ndarray, dict]:
        # The outer batch wrapper loses stdin on this machine. The identical embedded solver accepts
        # one flat 21-weight JSON array per line and returns one {"R": number} line per input.
        payload = "".join(json.dumps(row, allow_nan=False) + "\n" for row in weights.tolist())
        started = time.perf_counter()
        group = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if sys.platform == "win32" else {}
        process = subprocess.Popen(self.command, cwd=str(self.executable.parent),
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1, **group)
        try:
            stdout, stderr = process.communicate(payload, timeout=self.timeout)
        except subprocess.TimeoutExpired as error:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True)
            stdout, stderr = process.communicate()
            raise SolverError(f"solver timed out after {self.timeout} seconds", stderr, stdout,
                              process.returncode) from error
        seconds = time.perf_counter() - started
        if process.returncode != 0:
            detail = stderr.strip() or stdout.strip()[-500:] or "(no output)"
            raise SolverError(f"solver exited with code {process.returncode}: {detail}",
                              stderr, stdout, process.returncode)
        try:
            values = parse_solver_output(stdout, stderr)
        except SolverError as error:
            error.returncode = process.returncode
            raise
        actual = np.asarray(values, dtype=float)
        if actual.shape != (len(weights),):
            raise SolverError(f"expected {len(weights)} R values, got {len(actual)}",
                              stderr, stdout, process.returncode)
        return actual, {"returncode": process.returncode, "stderr": stderr, "seconds": seconds,
                        "response_keys": ["R"], "transport": "embedded --serve JSON-lines",
                        "workers": self.workers, "response_rows": len(actual)}


class SyntheticEvaluator:
    """Deterministic interface-only truth; never used for model selection or real-performance claims."""
    kind = "synthetic"

    def __init__(self, config: dict):
        del config
        target = np.linspace(0.03, 0.4, 21).reshape(3, 7)
        self.target = (target / target.sum(axis=1, keepdims=True)).reshape(-1)

    def evaluate(self, weights: np.ndarray) -> tuple[np.ndarray, dict]:
        started = time.perf_counter()
        difference = (weights - self.target).reshape(-1, 3, 7)
        actual = 120 + 35 * np.sum(difference ** 2, axis=(1, 2)) + 2 * np.prod(
            np.abs(difference).sum(axis=2), axis=1)
        return actual, {"returncode": 0, "stderr": "", "seconds": time.perf_counter() - started,
                        "response_keys": ["R"]}


def environment_record(config: dict, command: list[str], evaluator_kind: str, started: str) -> dict:
    packages = {name: importlib.metadata.version(name) for name in
                ("numpy", "scipy", "scikit-learn", "pandas", "matplotlib")}
    solver_path = Path(config["solver"]["path"])
    inner_path = Path(config["solver"]["inner_path"])
    return {
        "command": command, "python": sys.version, "python_executable": sys.executable,
        "platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
        "dependencies": packages, "thread_environment": {name: os.environ.get(name) for name in
            ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS")},
        "script_sha256": {path.name: sha256_file(path) for path in
                          (ROOT / "run_online.py", ROOT / "plot_results.py", ROOT / "config.json",
                           ROOT / "requirements.txt")},
        "training_sha256": sha256_file(ROOT / config["data"]["path"]),
        "solver_path": str(solver_path) if evaluator_kind == "solver" else None,
        "solver_sha256": sha256_file(solver_path) if evaluator_kind == "solver" else None,
        "solver_transport": config["solver"]["transport"] if evaluator_kind == "solver" else None,
        "inner_solver_path": str(inner_path) if evaluator_kind == "solver" else None,
        "inner_solver_sha256": sha256_file(inner_path) if evaluator_kind == "solver" else None,
        "master_seed": config["master_seed"], "evaluator": evaluator_kind, "started_utc": started,
        "ended_utc": None, "status": "running",
    }


def run(args: argparse.Namespace) -> Path:
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_id):
        raise ValueError("run-id may contain only letters, digits, dot, underscore and hyphen")
    train_weights, train_actual, _ = load_training(config)
    run_dir = ROOT / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "calls").mkdir(exist_ok=True)
    synthetic = args.synthetic
    dry_run = args.dry_run
    if synthetic and dry_run:
        raise ValueError("--synthetic and --dry-run are mutually exclusive")
    evaluator = SyntheticEvaluator(config) if synthetic else SolverEvaluator(config)
    rounds = 2 if synthetic else config["rounds"]
    batch_size = 8 if synthetic else config["batch_size"]
    local_count = 600 if synthetic else None
    structure_count = 600 if synthetic else None
    maxiter_override = 3 if synthetic else None
    attempts_path = run_dir / "attempts.jsonl"
    if dry_run:
        attempts, sent_weights, next_call = [], np.empty((0, DIM)), 1
        previous_calls = []
        evaluations = pd.DataFrame(columns=EVALUATION_COLUMNS)
    else:
        attempts, sent_weights, next_call = prepare_resume(run_dir, args.accept_lost_attempt)
        previous_calls = load_previous_calls(run_dir)
        evaluations = load_evaluations(run_dir)
    started_utc = utc_now()
    if next_call > rounds:
        print(f"Run already has {next_call - 1} attempts; nothing to do.", flush=True)
        return run_dir
    if not dry_run and next_call == 1:
        write_json(run_dir / "environment.json", environment_record(config, sys.argv, evaluator.kind, started_utc))
    elif not dry_run:
        environment_path = run_dir / "environment.json"
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
        environment.setdefault("resumptions", []).append(
            environment_record(config, sys.argv, evaluator.kind, started_utc))
        environment.update({"ended_utc": None, "status": "running"})
        write_json(environment_path, environment)

    for call in range(next_call, rounds + 1):
        online_weights = evaluations[WEIGHT_COLUMNS].to_numpy(dtype=float) if len(evaluations) else np.empty((0, DIM))
        online_actual = evaluations["actual_R"].to_numpy(dtype=float) if len(evaluations) else np.empty(0)
        observed_weights = np.vstack([train_weights, online_weights])
        observed_actual = np.concatenate([train_actual, online_actual])
        excluded = np.vstack([train_weights, sent_weights]) if len(sent_weights) else train_weights
        calibration = calibration_coefficient(evaluations, config)
        model = AdditiveARDGP(config, maxiter_override=maxiter_override).fit(observed_weights, observed_actual)
        quotas = quotas_for_call(config, call, previous_calls, batch_size)
        observation_tree = cKDTree(observed_weights)
        neighbour_distances, _ = observation_tree.query(observed_weights, k=2, p=2, workers=1)
        spacing = float(np.median(neighbour_distances[:, 1]))
        pool = None
        selection = None
        pool_stats = []
        for replenishment in range(config["candidates"]["max_replenishments"] + 1):
            new_pool = make_candidates(observed_weights, observed_actual, config, call,
                                       2 * replenishment, 2 * replenishment + 1,
                                       local_count, structure_count)
            valid = valid_weight_mask(new_pool["weights"])
            unique = ~duplicate_mask(new_pool["weights"], excluded, config["selection"]["duplicate_atol"])
            keep = valid & unique
            new_pool = {key: (value[keep] if isinstance(value, np.ndarray) and len(value) == len(keep) else value)
                        for key, value in new_pool.items()}
            mean, raw_std = model.predict(new_pool["weights"])
            new_pool["mean"] = mean
            new_pool["std_raw"] = raw_std
            new_pool["std"] = np.maximum(raw_std, 1e-6) * calibration
            new_pool["nearest"] = observation_tree.query(new_pool["weights"], k=1, p=2, workers=1)[0]
            if pool is None:
                pool = new_pool
            else:
                pool = {key: (np.concatenate([pool[key], new_pool[key]]) if isinstance(value, np.ndarray)
                              else value for key, value in new_pool.items())}
            pool_stats.append({"replenishment": replenishment, "added": len(new_pool["weights"]),
                               "local": int(np.sum(new_pool["source"] == "exploit")),
                               "structure": int(np.sum(new_pool["source"] == "structure"))})
            selection = select_batch(pool, pool["mean"], pool["std"], pool["nearest"], spacing, quotas, config)
            if len(selection.points) == batch_size:
                break
        if selection is None or len(selection.points) != batch_size:
            raise RuntimeError(f"candidate replenishment could not fill call {call}: "
                               f"got {0 if selection is None else len(selection.points)} of {batch_size}; solver not called")
        weights = normalise(np.asarray(selection.points))
        # Recheck every system-boundary condition immediately before logging/sending.
        validate_weights(weights, batch_limit=batch_size)
        if duplicate_mask(weights, excluded, config["selection"]["duplicate_atol"]).any():
            raise RuntimeError("selected batch duplicates a training or previously sent point")
        if duplicate_mask(weights, weights, 0).sum() != len(weights):
            raise RuntimeError("internal duplicate self-check failed")
        for left in range(batch_size):
            if (np.max(np.abs(weights[left + 1:] - weights[left]), axis=1) <=
                    config["selection"]["duplicate_atol"]).any():
                raise RuntimeError("selected batch contains an intra-batch duplicate")
        selected_indices = np.asarray(selection.indices, dtype=int)
        structures = dominant_structures(weights)
        nearest = pool["nearest"][selected_indices]
        record = {
            "call": call, "status": "prepared", "timestamp_utc": utc_now(),
            "seed_specification": {"master_seed": config["master_seed"],
                                   "streams": [2 * len(pool_stats) - 2, 2 * len(pool_stats) - 1]},
            "train_rows": len(observed_weights), "original_train_rows": len(train_weights),
            "online_train_rows": len(online_weights), "model": model.diagnostics,
            "calibration": calibration, "spacing": spacing, "candidate_pool": pool_stats,
            "candidate_structures_before_selection": int(len({tuple(v) for v in pool["structures"]})),
            "requested_quotas": quotas, "slot_counts": dict(Counter(selection.labels)),
            "weights": weights.tolist(), "predicted_mean": pool["mean"][selected_indices].tolist(),
            "predicted_std_raw": pool["std_raw"][selected_indices].tolist(),
            "predicted_std": pool["std"][selected_indices].tolist(), "slots": selection.labels,
            "dominant_structures": structures.tolist(), "nearest_observed_distance": nearest.tolist(),
            "actual_R": None, "metrics": None, "solver": None,
        }
        if dry_run:
            record["status"] = "dry_run"
            write_json(run_dir / "calls" / "call_01.json", record)
            write_json(run_dir / "dry_run.json", record)
            print(f"Dry run: call 1 prepared {batch_size} points; solver was not started.", flush=True)
            return run_dir

        append_jsonl(attempts_path, {"status": "started", "call": call, "timestamp": utc_now(),
                                     "weights": weights.tolist()})
        sent_weights = np.vstack([sent_weights, weights]) if len(sent_weights) else weights.copy()
        try:
            actual, solver_info = evaluator.evaluate(weights)
        except Exception as error:
            append_jsonl(attempts_path, {"status": "failed", "call": call, "timestamp": utc_now(),
                                         "error": str(error), "stderr": getattr(error, "stderr", ""),
                                         "stdout": getattr(error, "stdout", "")[-4000:],
                                         "returncode": getattr(error, "returncode", None)})
            record.update({"status": "failed", "error": str(error)})
            write_json(run_dir / "calls" / f"call_{call:02d}.json", record)
            previous_calls.append(record)
            write_metrics(run_dir, previous_calls)
            environment_path = run_dir / "environment.json"
            environment = json.loads(environment_path.read_text(encoding="utf-8"))
            environment.update({"ended_utc": utc_now(), "status": "failed"})
            write_json(environment_path, environment)
            raise
        append_jsonl(attempts_path, {"status": "ok", "call": call, "timestamp": utc_now(),
                                     "actual_R": actual.tolist(), "solver": solver_info})
        previous_online_best = float(online_actual.min()) if len(online_actual) else None
        best_before = float(observed_actual.min())
        running_best = min(float(actual.min()), previous_online_best) if previous_online_best is not None else float(actual.min())
        record.update({
            "status": "ok", "actual_R": actual.tolist(), "solver": solver_info,
            "metrics": prediction_metrics(pool["mean"][selected_indices], pool["std"][selected_indices], actual),
            "batch_min": float(actual.min()), "running_best_online": running_best,
            "improved_global": bool(float(actual.min()) < best_before),
        })
        write_json(run_dir / "calls" / f"call_{call:02d}.json", record)
        timestamp = utc_now()
        rows = []
        for point in range(batch_size):
            row = {"call": call, "point": point, "actual_R": float(actual[point]),
                   "predicted_mean": float(pool["mean"][selected_indices][point]),
                   "predicted_std_raw": float(pool["std_raw"][selected_indices][point]),
                   "predicted_std": float(pool["std"][selected_indices][point]),
                   "slot": selection.labels[point], "nearest_observed_distance": float(nearest[point]),
                   "timestamp": timestamp}
            row.update({name: float(weights[point, index]) for index, name in enumerate(WEIGHT_COLUMNS)})
            row.update({f"dominant_b{building + 1}": int(structures[point, building]) for building in range(3)})
            rows.append(row)
        append_evaluations(run_dir, rows)
        evaluations = load_evaluations(run_dir)
        previous_calls.append(record)
        write_metrics(run_dir, previous_calls)
        print(f"call {call:02d}/{rounds}: min R={actual.min():.12f}; "
              f"online best={running_best:.12f}; rows={len(evaluations)}", flush=True)

    environment_path = run_dir / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment.update({"ended_utc": utc_now(), "status": "completed"})
    write_json(environment_path, environment)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="independent_gp_2026")
    parser.add_argument("--dry-run", action="store_true", help="prepare only call 1; never start the solver")
    parser.add_argument("--synthetic", action="store_true", help="run the deterministic two-call interface smoke test")
    parser.add_argument("--accept-lost-attempt", action="store_true",
                        help="mark unresolved started calls lost and continue without resending them")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(f"Run directory: {result}", flush=True)
