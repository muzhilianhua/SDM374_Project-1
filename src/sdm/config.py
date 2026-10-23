"""Experiment configs (TOML): the dataset, model, optimizer and evaluation protocol a run uses."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
import tomllib
from typing import Any

from .paths import resolve


@dataclass(frozen=True)
class Component:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluateSettings:
    folds: int = 5
    seed: int = 2026
    top_k: int = 50


@dataclass(frozen=True)
class Config:
    dataset: Path
    model: Component
    optimizer: Component | None
    evaluate: EvaluateSettings


def _component(raw: dict, section: str) -> Component | None:
    table = raw.get(section)
    if table is None:
        return None
    unknown = sorted(set(table) - {"name", "params"})
    if unknown or "name" not in table:
        raise ValueError(f"[{section}] needs `name` and optional [{section}.params]; unexpected keys {unknown}")
    return Component(table["name"], dict(table.get("params", {})))


def load_config(path) -> Config:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    unknown = sorted(set(raw) - {"data", "model", "optimizer", "evaluate"})
    if unknown:
        raise ValueError(f"{path}: unknown section(s) {unknown}")
    model = _component(raw, "model")
    if model is None:
        raise ValueError(f"{path}: a [model] section is required")
    evaluate = raw.get("evaluate", {})
    valid = [item.name for item in fields(EvaluateSettings)]
    if set(evaluate) - set(valid):
        raise ValueError(f"[evaluate] accepts {valid}; got {sorted(evaluate)}")
    return Config(
        dataset=resolve(raw.get("data", {}).get("dataset", "dataset_same8")),
        model=model,
        optimizer=_component(raw, "optimizer"),
        evaluate=EvaluateSettings(**evaluate),
    )
