"""Align decisions and completed hourly states for CityLearn 2.5.0."""

from __future__ import annotations

import csv
from contextlib import contextmanager
from copy import deepcopy
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory


@contextmanager
def padded_schema(schema: dict, spec):
    schema = deepcopy(schema)
    count = spec.end - spec.start + 1
    allocated = ((count // 24) + 1) * 24
    buffer_rows = allocated - count

    with TemporaryDirectory(prefix="citylearn_timing_v2_") as temp:
        cache = {}
        for building in schema["buildings"].values():
            for key in ("energy_simulation", "weather", "carbon_intensity", "pricing"):
                if not building.get(key):
                    continue
                source = (spec.dataset_dir / building[key]).resolve()
                if source not in cache:
                    with source.open(encoding="utf-8-sig", newline="") as stream:
                        reader = csv.reader(stream)
                        header = next(reader)
                        rows = list(islice(reader, spec.start, spec.end + 1))
                    if len(rows) != count:
                        raise ValueError(f"{source} does not cover the declared window")
                    destination = Path(temp) / f"series_{len(cache)}.csv"
                    with destination.open("w", encoding="utf-8", newline="") as stream:
                        writer = csv.writer(stream)
                        writer.writerow(header)
                        writer.writerows(rows)
                        writer.writerows([rows[-1]] * buffer_rows)
                    cache[source] = str(destination)
                building[key] = cache[source]

            dynamics = building.get("dynamics")
            if dynamics:
                attributes = dynamics["attributes"]
                attributes["filename"] = str((spec.dataset_dir / attributes["filename"]).resolve())

        schema["root_directory"] = temp
        schema["simulation_start_time_step"] = 0
        schema["simulation_end_time_step"] = allocated - 1
        schema["episode_time_steps"] = allocated
        schema["rolling_episode_split"] = False
        schema["random_episode_split"] = False
        yield schema


def decision_observations(env, raw_observations):
    """Use the most recently completed physical state for three observations."""
    if env.time_step == 0:
        return [list(row) for row in raw_observations]

    completed = env.time_step - 1
    fixed = [list(row) for row in raw_observations]
    for i, (building, names) in enumerate(zip(env.buildings, env.observation_names)):
        values = {
            "electrical_storage_soc": building.electrical_storage.soc[completed],
            "net_electricity_consumption": building.net_electricity_consumption[completed],
            "indoor_dry_bulb_temperature": (
                building.energy_simulation.indoor_dry_bulb_temperature[completed]
            ),
        }
        for name, value in values.items():
            if name not in names:
                raise RuntimeError(f"required controller observation missing: {name}")
            fixed[i][names.index(name)] = float(value)
    return fixed
