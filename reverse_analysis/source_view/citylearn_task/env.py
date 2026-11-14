"""Readable reconstruction of the observation and equipment helpers."""

from __future__ import annotations

from typing import Sequence


def _find_obs_index(observation_names: Sequence[str], candidates: Sequence[str]) -> int | None:
    lower_names = [name.lower() for name in observation_names]
    for candidate in candidates:
        c_lower = candidate.lower()
        if c_lower in lower_names:
            return lower_names.index(c_lower)
    for candidate in candidates:
        c_lower = candidate.lower()
        for i, name in enumerate(lower_names):
            if c_lower in name:
                return i
    return None


def build_observation_index(env) -> list[dict[str, int]]:
    indices = []
    for i in range(len(env.observation_names)):
        names = env.observation_names[i]
        idx: dict[str, int] = {}

        def put(key: str, candidates: Sequence[str]) -> None:
            j = _find_obs_index(names, candidates)
            if j is not None:
                idx[key] = j

        put("price", ["electricity_pricing", "electricity_price", "price"])
        put("price_pred_1", ["electricity_pricing_predicted_1",
                             "electricity_price_predicted_1", "price_predicted_1"])
        put("price_pred_2", ["electricity_pricing_predicted_2",
                             "electricity_price_predicted_2", "price_predicted_2"])
        put("price_pred_3", ["electricity_pricing_predicted_3",
                             "electricity_price_predicted_3", "price_predicted_3"])
        put("carbon", ["carbon_intensity", "carbon_emission_factor",
                        "carbon_factor", "emissions"])
        put("net_electricity", ["net_electricity_consumption",
                                "net_electricity_demand", "net_electricity"])
        put("non_shiftable_load", ["non_shiftable_load",
                                    "non_shiftable_electric_load"])
        put("solar_generation", ["solar_generation", "pv_generation", "solar_power"])
        put("indoor_temp", ["indoor_dry_bulb_temperature",
                             "indoor_temperature", "indoor_air_temperature"])
        put("setpoint", ["indoor_dry_bulb_temperature_set_point",
                          "indoor_dry_bulb_temperature_cooling_set_point",
                          "indoor_dry_bulb_temperature_heating_set_point",
                          "indoor_temperature_set_point", "set_point"])
        put("cooling_demand", ["cooling_demand"])
        put("occupant_count", ["occupant_count"])
        put("power_outage", ["power_outage"])
        put("outdoor_temp", ["outdoor_dry_bulb_temperature",
                              "outdoor_temperature"])
        put("soc", ["electrical_storage_soc", "battery_soc", "soc"])
        indices.append(idx)
    return indices


def _nominal_power(env, building_index: int, device_name: str) -> float | None:
    device = getattr(env.buildings[building_index], device_name, None)
    if device is None:
        return None
    for attr in ("nominal_power", "max_power", "power", "rated_power"):
        value = getattr(device, attr, None)
        if value is not None:
            try:
                return float(value)
            except Exception:
                pass
    return None


def get_electrical_storage_nominal_power(env, building_index: int) -> float | None:
    return _nominal_power(env, building_index, "electrical_storage")


def get_cooling_device_nominal_power(env, building_index: int) -> float | None:
    return _nominal_power(env, building_index, "cooling_device")
