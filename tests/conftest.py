import importlib.util

import pytest

from sdm.data import load_dataset
from sdm.paths import REPO_ROOT
from sdm.registry import build_model

BASELINE_CONFIG = REPO_ROOT / "experiments" / "baseline.toml"


@pytest.fixture(scope="session")
def dataset():
    return load_dataset(REPO_ROOT / "dataset_same8")


@pytest.fixture(scope="session")
def baseline():
    """The untouched course script, imported as a module for side-by-side comparison."""
    spec = importlib.util.spec_from_file_location("course_baseline", REPO_ROOT / "baseline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def ridge2(dataset):
    return build_model("ridge2").fit(dataset.X, dataset.y)
