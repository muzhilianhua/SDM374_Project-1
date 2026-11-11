"""Name -> class registries. Every non-underscore module in sdm.models / sdm.optimizers / sdm.policies is
imported on first lookup, so a new plugin only needs a new file with a @register_model / @register_optimizer /
@register_policy class."""
from __future__ import annotations

import dataclasses
import importlib
import pkgutil
from typing import Any

from .interfaces import BatchPolicy, Optimizer, Surrogate

_MODELS: dict[str, type[Surrogate]] = {}
_OPTIMIZERS: dict[str, type[Optimizer]] = {}
_POLICIES: dict[str, type[BatchPolicy]] = {}
_PLUGIN_PACKAGES = ("sdm.models", "sdm.optimizers", "sdm.policies")
_discovered = False


def _register(table: dict, base: type, kind: str, name: str):
    def decorator(cls):
        if not issubclass(cls, base):
            raise TypeError(f"{cls.__name__} must subclass {base.__name__}")
        if not dataclasses.is_dataclass(cls.Params):
            raise TypeError(f"{cls.__name__}.Params must be a dataclass")
        if name in table and table[name] is not cls:
            raise ValueError(f"{kind} '{name}' is already registered by {table[name].__module__}")
        cls.name = name
        table[name] = cls
        return cls
    return decorator


def register_model(name: str):
    return _register(_MODELS, Surrogate, "model", name)


def register_optimizer(name: str):
    return _register(_OPTIMIZERS, Optimizer, "optimizer", name)


def register_policy(name: str):
    return _register(_POLICIES, BatchPolicy, "policy", name)


def _discover() -> None:
    global _discovered
    if _discovered:
        return
    for package_name in _PLUGIN_PACKAGES:
        package = importlib.import_module(package_name)
        for module in pkgutil.iter_modules(package.__path__):
            if not module.name.startswith("_"):
                importlib.import_module(f"{package_name}.{module.name}")
    _discovered = True


def models() -> dict[str, type[Surrogate]]:
    _discover()
    return dict(_MODELS)


def optimizers() -> dict[str, type[Optimizer]]:
    _discover()
    return dict(_OPTIMIZERS)


def policies() -> dict[str, type[BatchPolicy]]:
    _discover()
    return dict(_POLICIES)


def build_params(cls: type, values: dict[str, Any] | None):
    """Instantiate cls.Params from a config table, rejecting unknown keys with the list of valid ones."""
    values = dict(values or {})
    known = [field.name for field in dataclasses.fields(cls.Params)]
    unknown = sorted(set(values) - set(known))
    if unknown:
        raise ValueError(f"{cls.name}: unknown parameter(s) {unknown}; valid: {known}")
    # TOML arrays arrive as lists; tuples keep the frozen Params hashable.
    values = {key: tuple(value) if isinstance(value, list) else value for key, value in values.items()}
    return cls.Params(**values)


def _lookup(table: dict, kind: str, name: str):
    if name not in table:
        raise ValueError(f"unknown {kind} '{name}'; registered: {sorted(table)}")
    return table[name]


def build_model(name: str, params: dict[str, Any] | None = None) -> Surrogate:
    cls = _lookup(models(), "model", name)
    return cls(build_params(cls, params))


def build_optimizer(name: str, params: dict[str, Any] | None = None) -> Optimizer:
    cls = _lookup(optimizers(), "optimizer", name)
    return cls(build_params(cls, params))


def build_policy(name: str, params: dict[str, Any] | None = None, *, model=None, optimizer=None) -> BatchPolicy:
    """`model` / `optimizer` are the config's components (objects with `name` and `params`), or None."""
    cls = _lookup(policies(), "policy", name)
    return cls(build_params(cls, params), model=model, optimizer=optimizer)


def describe(cls: type) -> list[dict[str, Any]]:
    """Parameter schema of a plugin: name, annotated type and default of each Params field."""
    return [
        {"name": field.name, "type": str(field.type),
         "default": None if field.default is dataclasses.MISSING else field.default}
        for field in dataclasses.fields(cls.Params)
    ]
