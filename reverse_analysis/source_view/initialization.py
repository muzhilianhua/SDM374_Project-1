"""Single-accounting correction and per-hour consistency checks."""

from __future__ import annotations

from types import MethodType
import numpy as np

DEVICE_NAMES = (
    "cooling_device", "heating_device", "dhw_device",
    "non_shiftable_load_device", "electrical_storage",
)


def single_accounting(env):
    """Patch hour-zero update_variables so reset accumulation is applied once."""
    assert env.time_step == 0
    records = []
    for building in env.buildings:
        assert not getattr(building, "_teacher_single_accounting", False)
        devices = [getattr(building, name) for name in DEVICE_NAMES]
        records.append({
            "building": building.name,
            "reset_device_electricity": [float(d.electricity_consumption[0]) for d in devices],
        })
        for device in devices:
            assert float(device.time_step_ratio) == 1.0
            device.update_electricity_consumption(
                -float(device.electricity_consumption[0]), enforce_polarity=False
            )
            assert abs(float(device.electricity_consumption[0])) < 1e-8

        original = building.update_variables

        def update(this, original=original, devices=devices):
            if this.time_step != 0:
                return original()
            saved = [d.update_electricity_consumption for d in devices]
            try:
                for device in devices:
                    device.update_electricity_consumption = lambda *args, **kwargs: None
                return original()
            finally:
                for device, method in zip(devices, saved):
                    device.update_electricity_consumption = method

        building.update_variables = MethodType(update, building)
        building._teacher_single_accounting = True
    return records


def check_completed_hour(env, agent, hour):
    errors = []
    for building, predicted in zip(env.buildings, agent.last_diagnostics):
        assert int(building.time_step) == hour + 1
        devices = [getattr(building, name) for name in DEVICE_NAMES]
        raw = sum(float(device.electricity_consumption[hour]) for device in devices)
        raw += float(building.solar_generation[hour])
        expected = 0.0 if building.power_outage_signal[hour] else raw
        actual = float(building.net_electricity_consumption[hour])
        net_error = abs(expected - actual)
        projection_error = abs(float(predicted["predicted_net"]) - actual)
        nsl_error = abs(
            float(building.non_shiftable_load_device.electricity_consumption[hour])
            - float(building.energy_to_non_shiftable_load[hour])
        )
        assert net_error < 2e-5, (building.name, hour, net_error)
        assert projection_error < 2e-4, (building.name, hour, projection_error)
        assert nsl_error < 2e-5, (building.name, hour, nsl_error)
        ref = float(building.energy_simulation.cooling_demand_without_control[hour])
        actual_cooling = float(building.energy_from_cooling_device[hour])
        assert actual_cooling >= ref - 2e-4
        assert actual_cooling <= 1.1 * ref + 2e-4
        errors.append([net_error, projection_error, nsl_error])
    return np.max(errors, axis=0).tolist()
