"""Reconstructed configuration, helpers, base initialization and reset.

The unused base prediction branch is omitted. The public evaluation path uses
the fully implemented controller.ProbeAgent override.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from citylearn.agents.base import Agent
from citylearn_task.env import (
    build_observation_index,
    get_cooling_device_nominal_power,
    get_electrical_storage_nominal_power,
)


@dataclass(frozen=True)
class LowerLevelConfig:
    action_candidates: tuple[float, ...] = (
        -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    )
    cooling_action_candidates: tuple[float, ...] = (
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    )
    cooling_sensitivity: float = 1.0
    max_battery_action_delta: float = 0.4
    max_cooling_action_delta: float = 0.5
    min_soc_reserve_normal: float = 0.05
    min_soc_reserve_outage: float = 0.15
    use_price_lookahead: bool = True
    lookahead_gamma: float = 1.0
    normalize_candidate_objectives: bool = True
    soc_min: float = 0.01
    soc_max: float = 0.99
    soc_action_gain: float = 0.2
    local_refine_step: float = 0.05
    local_refine_radius: float = 0.25
    local_refine_points: int = 9


def normalize_building_weights(w):
    values = np.asarray(w, dtype=float)
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    if total <= 0.0:
        return np.ones_like(values) / len(values)
    return values / total


def canonicalize_weights(w, n_objectives=7):
    values = np.asarray(w, dtype=float)
    if values.ndim != 1:
        values = values.reshape(-1)
    if len(values) == n_objectives:
        return normalize_building_weights(values)
    if len(values) == 4 and n_objectives == 7:
        return normalize_building_weights(np.concatenate([values, np.zeros(3, dtype=float)]))
    raise ValueError(f"weights length must be 4 or {n_objectives}, got {len(values)}")


def _unique_sorted(values, lo, hi):
    clipped = [float(np.clip(v, lo, hi)) for v in values]
    rounded = sorted({round(v, 6) for v in clipped})
    return [float(v) for v in rounded]


def _build_local_refine_points(center, step, radius, points):
    if step <= 0.0 or radius <= 0.0 or points < 2:
        return [center]
    half_span = min(radius, step * max(1, points // 2))
    return [float(v) for v in np.linspace(center - half_span, center + half_span, points)]


class WeightedMyopicBatteryAgent(Agent):
    """Base class used by ProbeAgent; the public evaluator uses the subclass."""

    def __init__(self, env, weights, config=None):
        super().__init__(env)
        self.config = config or LowerLevelConfig()
        self.weights = [canonicalize_weights(row) for row in weights]
        self.obs_index = build_observation_index(env)
        self.nominal_power = [get_electrical_storage_nominal_power(env, i)
                              for i in range(len(env.buildings))]
        self.cooling_nominal_power = [get_cooling_device_nominal_power(env, i)
                                      for i in range(len(env.buildings))]
        self.prev_battery_action = [0.0] * len(env.buildings)
        self.prev_cooling_action = [0.0] * len(env.buildings)
        self.prev_net = [None] * len(env.buildings)

    def reset(self):
        super().reset()
        self.prev_battery_action = [0.0] * len(self.env.buildings)
        self.prev_cooling_action = [0.0] * len(self.env.buildings)
        self.prev_net = [None] * len(self.env.buildings)

    def predict(self, observations, deterministic=None):
        # Keep the omitted branch explicit rather than silently running a stub.
        raise NotImplementedError("base predict is omitted; public path uses controller.ProbeAgent")
