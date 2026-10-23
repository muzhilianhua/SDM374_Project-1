"""Simplex constraint, dataset integrity, registry and config validation."""
import shutil

import numpy as np
import pytest

from sdm import registry
from sdm.config import load_config
from sdm.data import DataIntegrityError, load_dataset
from sdm.interfaces import Surrogate
from sdm.paths import REPO_ROOT
from sdm.simplex import check_weights, project, sample


def test_projection_lands_on_simplex_and_is_idempotent():
    rng = np.random.default_rng(0)
    projected = project(rng.normal(size=(50, 21)) * 3)
    check_weights(projected)
    np.testing.assert_allclose(project(projected), projected, atol=1e-12)
    feasible = sample(10, rng)
    np.testing.assert_allclose(project(feasible), feasible, atol=1e-12)


@pytest.mark.parametrize("weights", [
    np.full(21, 1 / 6),                               # rows sum to 7/6
    np.r_[-0.1, 1.1, np.zeros(5), np.full(14, 1 / 7)],  # negative entry
    np.full(20, 1 / 7),                               # wrong length
])
def test_check_weights_rejects_invalid(weights):
    with pytest.raises(ValueError):
        check_weights(weights)


def test_tampered_dataset_is_rejected(tmp_path):
    copy = tmp_path / "dataset"
    shutil.copytree(REPO_ROOT / "dataset_same8", copy)
    with open(copy / "train.csv", "ab") as file:
        file.write(b"\n")
    with pytest.raises(DataIntegrityError):
        load_dataset(copy)


def test_registry_discovers_baseline_plugins():
    assert "ridge2" in registry.models()
    assert "pgd" in registry.optimizers()


def test_unknown_names_and_params_are_reported():
    with pytest.raises(ValueError, match="registered"):
        registry.build_model("no-such-model")
    with pytest.raises(ValueError, match="valid"):
        registry.build_model("ridge2", {"alpha": 1.0})


def test_registering_a_plugin_and_duplicate_names(monkeypatch):
    monkeypatch.setattr(registry, "_MODELS", dict(registry.models()))

    @registry.register_model("constant")
    class Constant(Surrogate):
        def fit(self, X, y):
            self.value = float(np.mean(y))
            return self

        def predict(self, X):
            return np.full(len(X), self.value), np.zeros(len(X))

    model = registry.build_model("constant").fit(np.zeros((2, 21)), [1.0, 3.0])
    assert model.predict(np.zeros((1, 21)))[0][0] == 2.0
    with pytest.raises(ValueError, match="already registered"):
        registry.register_model("constant")(type("Other", (Constant,), {}))


def test_config_rejects_unknown_sections(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text('[model]\nname = "ridge2"\n[modle.params]\nfolds = 3\n', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown section"):
        load_config(path)
