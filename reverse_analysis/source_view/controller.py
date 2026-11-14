"""Readable per-function reconstruction of the original bounded controller.

Comments and formatting were added during analysis. See ../docs/VALIDATION.md.
The release uses vendored CityLearn and reconstructed citylearn_task helpers.
"""

from dataclasses import dataclass
import math
import numpy as np

from citylearn_task.lower import WeightedMyopicBatteryAgent, _build_local_refine_points

PROBE_CONTROLLER_VERSION = "2026.10.15.bounded_single_accounting_stable_spans.v2"
REFERENCE_COOLING_MAX_RATIO = 1.1


@dataclass(frozen=True)
class BatteryProjection:
    grid_energy: float
    soc_after: float
    requested_energy: float
    clipped_energy: float


def project_battery(storage, action: float) -> BatteryProjection:
    """Read-only replica of the hourly battery transition used for candidates."""
    if not math.isclose(float(storage.time_step_ratio), 1.0):
        raise ValueError("probe requires hourly storage time_step_ratio=1")
    capacity = float(storage.capacity)
    if capacity <= 0:
        return BatteryProjection(0.0, 0.0, 0.0, 0.0)

    t = int(storage.time_step)
    soc = float(storage.soc[t - 1 if t else 0])
    initial = float(storage.energy_init)
    requested = float(action) * float(storage.nominal_power)
    maximum = max(float(storage.get_max_input_power()), 0.0)

    if requested >= 0:
        energy = min(
            maximum,
            max(float(storage.available_nominal_power), 0.0),
            max(float(storage.degraded_capacity) - initial, 0.0),
            requested,
        )
        efficiency = float(storage.get_current_efficiency(min(requested, maximum)))
    else:
        dod_limit = -max(
            (soc - (1.0 - float(storage.depth_of_discharge)))
            * capacity * float(storage.round_trip_efficiency), 0.0
        )
        energy = max(-maximum, dod_limit, requested)
        efficiency = float(storage.get_current_efficiency(min(abs(requested), maximum)))

    eta = math.sqrt(efficiency)
    if energy >= 0:
        final = min(initial + energy * eta, capacity)
    else:
        final = max(0.0, initial + energy / eta)
    delta = final - initial
    grid = delta / eta if delta >= 0 else delta * eta
    return BatteryProjection(float(grid), float(final / capacity), requested, float(energy))


def reference_cooling(building) -> float:
    t = int(building.time_step)
    return max(float(building.energy_simulation.cooling_demand_without_control[t]), 0.0)


def cooling_service_floor(reference, cop, nominal, enabled=True) -> float:
    if not enabled or nominal <= 0 or cop <= 0:
        return 0.0
    return float(np.clip(reference / (cop * nominal), 0.0, 1.0))


def choose_candidate(values, weights, tie_keys, normalize=True) -> int:
    values = np.asarray(values, dtype=float)
    if normalize and len(values) > 1:
        spans = np.ptp(values, axis=0)
        active = spans > 1e-10 * np.maximum(1.0, np.max(np.abs(values), axis=0))
        normalized = np.zeros_like(values)
        normalized[:, active] = (
            values[:, active] - values[:, active].min(axis=0)
        ) / spans[active]
        values = normalized

    objectives = values @ np.asarray(weights, dtype=float)
    best = float(objectives.min())
    tied = np.flatnonzero(objectives <= best + 1e-12 * max(1.0, abs(best)))
    return min(tied.tolist(), key=lambda j: tie_keys[j])


def _candidates(values, low, high):
    if low > high + 1e-12:
        raise ValueError(f"empty candidate interval {low}..{high}")
    return sorted({float(np.clip(v, low, high)) for v in [*values, low, high]})


def _thermal_electricity(device, demand, outdoor, heating=True) -> float:
    if hasattr(device, "get_cop"):
        needed = float(device.get_input_power(demand, outdoor, heating=heating))
    else:
        needed = float(device.get_input_power(demand))
    return min(max(needed, 0.0), max(float(device.available_nominal_power), 0.0))


class ProbeAgent(WeightedMyopicBatteryAgent):
    """Bounded one-step controller selected by the public evaluator."""

    def __init__(self, env, weights, mode="bounded", config=None):
        if mode != "bounded":
            raise ValueError("this isolated controller requires mode='bounded'")
        self.mode = mode
        super().__init__(env, weights, config=config)
        self.diagnostics = []
        self.last_diagnostics = []

    def reset(self):
        super().reset()
        self.diagnostics = []
        self.last_diagnostics = []

    def predict(self, observations, deterministic=None):
        """Enumerate battery/cooling candidates and choose the lowest weighted score."""
        actions = []
        self.last_diagnostics = []

        for i, (obs, building) in enumerate(zip(observations, self.env.buildings)):
            t = int(building.time_step)
            if not math.isclose(float(building.seconds_per_time_step), 3600.0):
                raise ValueError("probe supports one-hour steps only")
            names = self.env.action_names[i]
            get = lambda name, default=0.0: float(obs[self.obs_index[i].get(name, -1)]) \
                if name in self.obs_index[i] else default

            outage = bool(building.power_outage)
            outdoor = float(building.weather.outdoor_dry_bulb_temperature[t])
            price = get("price")
            carbon = get("carbon")
            future_price = max([price] + [
                get(k) for k in ("price_pred_1", "price_pred_2", "price_pred_3")
                if k in self.obs_index[i]
            ])
            if not self.config.use_price_lookahead:
                future_price = price

            nsl = max(float(building.energy_simulation.non_shiftable_load[t]), 0.0)
            pv = abs(float(building.solar_generation[t]))
            indoor = get("indoor_temp")
            setpoint = get("setpoint", indoor)
            occupancy = max(get("occupant_count"), 0.0)

            battery = building.electrical_storage
            soc = float(battery.soc[t - 1 if t else 0])
            physical_floor = 1.0 - float(battery.depth_of_discharge)
            reserve = (self.config.min_soc_reserve_outage if outage
                       else self.config.min_soc_reserve_normal)
            soc_floor = max(physical_floor, self.config.soc_min, reserve)
            previous = self.prev_battery_action[i]
            nominal = float(battery.nominal_power)
            previous_effective = (
                float(battery.energy_balance[t - 1]) / nominal
                if t and nominal else 0.0
            )

            max_rate = float(battery.get_max_input_power())
            lo = max(-1.0, -max_rate / nominal) if nominal > 0 else 0.0
            hi = min(1.0, max_rate / nominal) if nominal > 0 else 0.0
            if nominal > 0:
                lo = max(lo, -(max(soc - soc_floor, 0.0) * battery.capacity
                              * battery.round_trip_efficiency) / nominal)
                hi = min(hi, max(float(battery.degraded_capacity) - float(battery.energy_init), 0.0) / nominal)
            if outage:
                hi = min(hi, 0.0)
            physical_lo, physical_hi = lo, hi
            lo = max(lo, previous - self.config.max_battery_action_delta)
            hi = min(hi, previous + self.config.max_battery_action_delta)
            slew_relaxed = lo > hi
            if slew_relaxed:
                lo, hi = physical_lo, physical_hi

            local = _build_local_refine_points(previous, self.config.local_refine_step,
                                         self.config.local_refine_radius,
                                         self.config.local_refine_points)
            xs = _candidates([*self.config.action_candidates, *local, 0.0], lo, hi)
            projected = []
            zero_soc = project_battery(battery, 0.0).soc_after
            for x in xs:
                p = project_battery(battery, x)
                if x < 0 and p.soc_after < soc_floor - 1e-7:
                    continue
                if x > 0 and p.soc_after > max(self.config.soc_max, zero_soc) + 1e-7:
                    continue
                projected.append((x, p))
            if not projected:
                projected = [(0.0, project_battery(battery, 0.0))]
                slew_relaxed = True

            cooler = building.cooling_device
            cop = float(cooler.get_cop(outdoor, heating=False))
            nominal_cool = float(cooler.nominal_power)
            q_ref = reference_cooling(building)
            hvac = int(building.energy_simulation.hvac_mode[t])
            dynamic = bool(building.simulate_dynamics)
            enabled = hvac in (1, 3)
            floor = cooling_service_floor(q_ref, cop, nominal_cool, enabled)
            ceiling = cooling_service_floor(REFERENCE_COOLING_MAX_RATIO * q_ref,
                                             cop, nominal_cool, enabled)
            prev_cool = self.prev_cooling_action[i]
            c_lo = max(floor, prev_cool - self.config.max_cooling_action_delta)
            c_hi = min(ceiling, prev_cool + self.config.max_cooling_action_delta)
            cooling_slew_relaxed = c_lo > c_hi
            if cooling_slew_relaxed:
                c_lo, c_hi = floor, ceiling
            c_local = _build_local_refine_points(prev_cool, self.config.local_refine_step,
                                           self.config.local_refine_radius,
                                           self.config.local_refine_points)
            cs = _candidates([*self.config.cooling_action_candidates, *c_local, floor], c_lo, c_hi)
            if not dynamic:
                cs = [floor]

            old_net = float(building.net_electricity_consumption[t - 1]) if t else None
            completed_peak = max(
                np.asarray(building.net_electricity_consumption[:t], dtype=float),
                default=0.0,
            )
            heat_e = _thermal_electricity(building.heating_device,
                                          max(float(building.energy_simulation.heating_demand[t]), 0.0), outdoor)
            dhw_e = _thermal_electricity(building.dhw_device,
                                          max(float(building.energy_simulation.dhw_demand[t]), 0.0), outdoor)
            # Already-accounted device electricity cannot be replaced by heat_e + dhw_e.
            existing_e = sum(float(d.electricity_consumption[t]) for d in (
                cooler, building.heating_device, building.dhw_device,
                building.non_shiftable_load_device, battery,
            ))

            metrics = []
            choices = []
            ties = []
            for x, bp in projected:
                for c in cs:
                    desired_q = q_ref if not dynamic else (c * nominal_cool * cop if enabled else 0.0)
                    cooling_e = min(max(desired_q / cop, 0.0), max(float(cooler.available_nominal_power), 0.0))
                    local_heat, local_dhw, served_nsl = heat_e, dhw_e, nsl
                    if outage:
                        available = max(pv - existing_e - bp.grid_energy, 0.0)
                        cooling_e = min(cooling_e, available)
                        available -= cooling_e
                        local_heat = min(local_heat, available)
                        available -= local_heat
                        local_dhw = min(local_dhw, available)
                        available -= local_dhw
                        served_nsl = min(nsl, available)
                    supplied_q = cooling_e * cop
                    flow = served_nsl + local_heat + local_dhw + cooling_e + bp.grid_energy
                    predicted = 0.0 if outage else existing_e + flow - pv
                    cost = price * predicted - self.config.lookahead_gamma * future_price * bp.grid_energy
                    emissions = max(carbon * predicted, 0.0)
                    peak = max(completed_peak, predicted)
                    discomfort = max(indoor - setpoint - self.config.cooling_sensitivity * c, 0.0)
                    discomfort += max(setpoint - indoor, 0.0)
                    effective_x = bp.grid_energy / nominal if nominal else 0.0
                    degradation = abs(effective_x) + 0.5 * abs(effective_x - previous_effective)
                    ramping = abs(predicted - old_net) if old_net is not None else 0.0
                    target = float(np.clip(0.2 + 0.03 * occupancy, 0.2, 0.5))
                    resilience = max(nsl - served_nsl, 0.0) if outage else 0.0
                    resilience += max(target - bp.soc_after, 0.0)
                    metrics.append([cost, emissions, peak, discomfort, degradation, ramping, resilience])
                    choices.append((x, c, bp, predicted, supplied_q))
                    ties.append((abs(x), max(supplied_q - q_ref, 0.0), c, x))

            selected = choose_candidate(metrics, self.weights[i], ties,
                                        normalize=self.config.normalize_candidate_objectives)
            x, c, bp, predicted, supplied_q = choices[selected]
            action = [0.0] * len(names)
            action[names.index("electrical_storage")] = float(x)
            action[names.index("cooling_device")] = float(c)
            actions.append(action)
            self.prev_battery_action[i] = float(x)
            self.prev_cooling_action[i] = float(c)
            self.prev_net[i] = float(predicted)

            diagnostic = {
                "building": building.name,
                "time_step": t,
                "mode": self.mode,
                "warmup": not dynamic,
                "hvac_mode": hvac,
                "outage": outage,
                "battery_action": float(x),
                "cooling_action": float(c),
                "predicted_net": float(predicted),
                "predicted_soc": float(bp.soc_after),
                "controller_version": PROBE_CONTROLLER_VERSION,
                "reference_cooling": q_ref,
                "cooling_floor": floor,
                "cooling_ceiling": ceiling,
                "cooling_candidate_min": c_lo,
                "cooling_candidate_max": c_hi,
                "cooling_reference_max_ratio": REFERENCE_COOLING_MAX_RATIO,
                "predicted_cooling": supplied_q,
                "predicted_unserved_cooling": max(q_ref - supplied_q, 0.0),
                "predicted_battery_energy": bp.grid_energy,
                "soc_before": soc,
                "soc_floor": soc_floor,
                "battery_slew_relaxed": slew_relaxed,
                "cooling_slew_relaxed": cooling_slew_relaxed,
            }
            self.last_diagnostics.append(diagnostic)
            self.diagnostics.append(diagnostic)
        return actions
