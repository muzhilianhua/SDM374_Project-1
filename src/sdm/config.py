"""Experiment configs (TOML): the dataset, model, optimizer and evaluation protocol a run uses; online
configs (`sdm online`) add [policy], [evaluator] and [online]."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import datetime
import json
from pathlib import Path
import re
import tomllib
from typing import Any

from .paths import REPO_ROOT, resolve
from .registry import build_model, build_optimizer


@dataclass(frozen=True)
class Component:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


PROTOCOLS = ("kfold", "frozen-v32")


@dataclass(frozen=True)
class EvaluateSettings:
    # "kfold": outer K-fold CV (folds, seed). "frozen-v32": the frozen 2x5 splits scored by v3.2,
    # where folds and seed are fixed by the release. top_k feeds the k-fold metrics under both.
    protocol: str = "kfold"
    folds: int = 5
    seed: int = 2026
    top_k: int = 50


@dataclass(frozen=True)
class EvaluatorSpec:
    # kind: "solver", "table" or "synthetic"; params go to sdm.online.build_evaluator.
    kind: str = "solver"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OnlineSettings:
    calls: int = 20                 # call budget, counting every solver attempt (decision 19)
    batch_size: int = 50            # points per call
    seed: int = 2026                # passed to the policy through OnlineState
    candidate_margin: int = 25      # ask the policy for batch_size + margin candidates, then screen
    max_rounds: int = 3             # if too few pass, ask again with twice as many, up to this many times
    max_retries: int = 1            # resend a failed batch at most this often per call; each try costs a call
    dedupe_atol: float = 1e-9       # renormalised points this close (max abs) count as the same point

    def __post_init__(self):
        if not 1 <= self.batch_size <= 50 or self.calls < 1:
            raise ValueError("[online] needs calls >= 1 and 1 <= batch_size <= 50")
        if self.candidate_margin < 0 or self.max_rounds < 1 or self.max_retries < 0 or self.dedupe_atol < 0:
            raise ValueError("[online] candidate_margin, max_retries and dedupe_atol must be >= 0, max_rounds >= 1")


@dataclass(frozen=True)
class Config:
    dataset: Path
    model: Component
    optimizer: Component | None
    evaluate: EvaluateSettings
    policy: Component | None = None
    evaluator: EvaluatorSpec | None = None
    online: OnlineSettings = field(default_factory=OnlineSettings)


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
    unknown = sorted(set(raw) - {"data", "model", "optimizer", "evaluate", "policy", "evaluator", "online"})
    if unknown:
        raise ValueError(f"{path}: unknown section(s) {unknown}")
    model = _component(raw, "model")
    if model is None:
        raise ValueError(f"{path}: a [model] section is required")
    evaluate = raw.get("evaluate", {})
    valid = [item.name for item in fields(EvaluateSettings)]
    if set(evaluate) - set(valid):
        raise ValueError(f"[evaluate] accepts {valid}; got {sorted(evaluate)}")
    protocol = evaluate.get("protocol", "kfold")
    if protocol not in PROTOCOLS:
        raise ValueError(f"[evaluate] protocol must be one of {list(PROTOCOLS)}; got {protocol!r}")
    if protocol == "frozen-v32" and {"folds", "seed"} & set(evaluate):
        raise ValueError("[evaluate] protocol 'frozen-v32' uses the frozen splits; remove folds and seed")
    evaluator = raw.get("evaluator")
    if evaluator is not None:
        unknown = sorted(set(evaluator) - {"kind", "params"})
        if unknown or "kind" not in evaluator:
            raise ValueError(f"[evaluator] needs `kind` and optional [evaluator.params]; unexpected keys {unknown}")
        evaluator = EvaluatorSpec(evaluator["kind"], dict(evaluator.get("params", {})))
    online = raw.get("online", {})
    valid = [item.name for item in fields(OnlineSettings)]
    if set(online) - set(valid):
        raise ValueError(f"[online] accepts {valid}; got {sorted(online)}")
    return Config(
        dataset=resolve(raw.get("data", {}).get("dataset", "dataset_same8")),
        model=model,
        optimizer=_component(raw, "optimizer"),
        evaluate=EvaluateSettings(**evaluate),
        policy=_component(raw, "policy"),
        evaluator=evaluator,
        online=OnlineSettings(**online),
    )


CONFIG_NAME = re.compile(r"^[a-z0-9][a-z0-9-]*$")
EXPERIMENTS_DIR = REPO_ROOT / "experiments"


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)  # JSON string escapes are valid TOML basic strings
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"cannot write {type(value).__name__} to TOML")


def dump_config(raw: dict[str, Any]) -> str:
    """Serialize a config dict ({data, model, optimizer, evaluate}) to TOML; load_config reads it back."""
    lines = []
    for section in ("data", "model", "optimizer", "evaluate"):
        table = raw.get(section)
        if table is None:
            continue
        lines.append(f"[{section}]")
        lines += [f"{key} = {_toml_value(value)}" for key, value in table.items() if key != "params"]
        if "params" in table:
            lines += ["", f"[{section}.params]"]
            lines += [f"{key} = {_toml_value(value)}" for key, value in table["params"].items()]
        lines.append("")
    return "\n".join(lines)


def write_new_config(name: str, raw: dict[str, Any], directory: Path = EXPERIMENTS_DIR) -> Path:
    """Validate and save experiments/<name>.toml; never overwrites an existing config."""
    if not CONFIG_NAME.match(name):
        raise ValueError("config name may only use lowercase letters, digits and '-', starting with a letter or digit")
    path = Path(directory) / f"{name}.toml"
    if path.exists():
        raise ValueError(f"{path.name} already exists; pick another name")
    build_model(raw["model"]["name"], raw["model"].get("params"))
    if raw.get("optimizer"):
        build_optimizer(raw["optimizer"]["name"], raw["optimizer"].get("params"))
    text = f"# Created from the workbench on {datetime.now():%Y-%m-%d %H:%M}.\n\n" + dump_config(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as file:
        file.write(text)
    load_config(path)
    return path
