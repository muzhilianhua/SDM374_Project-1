"""Readable reconstruction of the public 168-hour evaluator."""

import importlib.metadata
import time
from pathlib import Path

import numpy as np

from citylearn_task.evaluate import load_scenario, normalize_weights, _scenario_schema
from citylearn_task.temporal import padded_schema, decision_observations


def metrics(net) -> dict:
    net = np.asarray(net, dtype=float)
    assert net.ndim == 1 and len(net) == 168 and np.isfinite(net).all()
    peak = float(np.max(net))
    mean = float(np.mean(net))
    return {
        "ramping": float(np.abs(np.diff(net)).sum()),
        "load_factor": float(1 - mean / peak) if peak > 0 else 0.0,
        "peak": peak,
        "mean_net": mean,
        "import_kwh": float(np.maximum(net, 0).sum()),
        "export_kwh": float(np.maximum(-net, 0).sum()),
    }


def evaluate_probe(weights, scenario, family="load_pv", strength=0,
                   disturbance_seed=0, include_building_traces=False):
    """Run one deterministic scenario and return the audit object used by engine.worker."""
    mode = "perturbed"
    from disturbances import perturb_schema

    if importlib.metadata.version("CityLearn") != "2.5.0":
        raise RuntimeError("This task requires CityLearn==2.5.0")
    from citylearn.citylearn import CityLearnEnv
    from citylearn_task.lower import WeightedMyopicBatteryAgent

    spec = load_scenario(scenario)
    w = normalize_weights(weights)
    start = time.perf_counter()
    expected_steps = spec.end - spec.start + 1
    assert expected_steps == 168

    with padded_schema(_scenario_schema(spec), spec) as schema:
        perturbation = perturb_schema(schema, family, strength, disturbance_seed)
        env = CityLearnEnv(
            schema,
            buildings=list(range(3)),
            simulation_start_time_step=0,
            simulation_end_time_step=schema["simulation_end_time_step"],
            random_seed=spec.seed,
            central_agent=False,
            active_actions=["electrical_storage", "cooling_device"],
        )
        assert [b.name for b in env.buildings] == list(spec.buildings)
        for building in env.buildings:
            outage_model = getattr(building, "stochastic_power_outage_model", None)
            if outage_model is not None:
                outage_model.random_seed = spec.seed

        # The public path uses the isolated bounded ProbeAgent branch.
        if mode == "v2":
            agent = WeightedMyopicBatteryAgent(env, w)
        else:
            from controller import ProbeAgent
            agent = ProbeAgent(env, w, mode="bounded")

        observations, _ = env.reset()
        from initialization import single_accounting, check_completed_hour
        initial_accounting = single_accounting(env)
        requested_actions = []
        hourly_checks = []

        for step in range(expected_steps):
            actions = agent.predict(
                decision_observations(env, observations), deterministic=True
            )
            requested_actions.append(actions)
            observations, _, terminated, truncated, _ = env.step(actions)
            hourly_checks.append(check_completed_hour(env, agent, step))
            if (terminated or truncated) and step < expected_steps - 1:
                raise RuntimeError("early end")

        # CityLearn stores one net trace per building; the task scores the district sum.
        net = np.sum(
            [np.asarray(b.net_electricity_consumption, dtype=float)[:168]
             for b in env.buildings],
            axis=0,
        )
        diagnostics = []
        building_traces = []
        for building in env.buildings:
            energy = building.energy_simulation
            temp = np.asarray(energy.indoor_dry_bulb_temperature, dtype=float)[:168]
            setpoint = np.asarray(
                energy.indoor_dry_bulb_temperature_cooling_set_point, dtype=float
            )[:168]
            actual = np.asarray(building.energy_from_cooling_device, dtype=float)[:168]
            reference = np.asarray(energy.cooling_demand_without_control, dtype=float)[:168]
            cooling = np.asarray(building.cooling_electricity_consumption, dtype=float)[:168]
            occupied = np.asarray(energy.occupant_count, dtype=float)[:168] > 0
            outage = (np.asarray(building.power_outage_signal, dtype=float)[:168]
                      if building.simulate_power_outage else np.zeros(168))
            unmet_nsl = np.maximum(
                np.asarray(energy.non_shiftable_load, dtype=float)[:168]
                - np.asarray(building.energy_to_non_shiftable_load, dtype=float)[:168], 0
            )
            diagnostics.append({
                "building": building.name,
                "unserved_non_shiftable_kwh": float(unmet_nsl.sum()),
                "max_temp": float(temp.max()),
                "min_temp": float(temp.min()),
                "occupied_degree_hours_below_setpoint2": float(
                    np.maximum(setpoint - 2 - temp, 0)[occupied].sum()),
                "excess_reference_cooling_kwh": float(np.maximum(actual - reference, 0).sum()),
                "battery_throughput_kwh": float(np.abs(np.asarray(
                    building.electrical_storage.energy_balance, dtype=float
                )[:168]).sum()),
                "occupied_degree_hours_above_setpoint2": float(
                    np.maximum(temp - setpoint - 2, 0)[occupied].sum()),
                "cooling_electricity_kwh": float(cooling.sum()),
                "reference_cooling_kwh": float(reference.sum()),
                "delivered_cooling_kwh": float(actual.sum()),
                "unmet_reference_kwh": float(np.maximum(reference - actual, 0).sum()),
                "unmet_after_warmup_kwh": float(np.maximum(reference[13:] - actual[13:], 0).sum()),
                "cooling_active_hours_after_warmup": int((cooling[13:] > 1e-6).sum()),
                "soc_final": float(building.electrical_storage.soc[167]),
                "outage_hours": int((outage > 0).sum()),
            })
            if include_building_traces:
                building_traces.append({
                    "building": building.name,
                    "temp": temp.tolist(),
                    "setpoint": setpoint.tolist(),
                    "reference_cooling": reference.tolist(),
                    "delivered_cooling": actual.tolist(),
                    "cooling_kwh": cooling.tolist(),
                    "soc": np.asarray(building.electrical_storage.soc, dtype=float)[:168].tolist(),
                    "net": np.asarray(building.net_electricity_consumption, dtype=float)[:168].tolist(),
                })
        score = metrics(net)
        result = {
            "mode": mode,
            "scenario_id": spec.id,
            **score,
            "weights": np.asarray(weights, dtype=float).reshape(3, 7).tolist(),
            "steps": expected_steps,
            "net": net.tolist(),
            "diagnostics": diagnostics,
            "runtime_seconds": time.perf_counter() - start,
            "protocol": "student_teams_20261014_single_accounting_168h",
            "diagnostic_version": "service_quality_v3",
            "seed": spec.seed,
            "building_traces": building_traces,
            "initial_accounting": initial_accounting,
            "hourly_accounting": hourly_checks,
            "requested_actions": requested_actions,
            "perturbation": perturbation,
        }
        result["actual_outage_signals"] = [
            np.asarray(b.power_outage_signal, dtype=float)[:168].tolist()
            if b.simulate_power_outage else [0.0] * 168
            for b in env.buildings
        ]
        return result
