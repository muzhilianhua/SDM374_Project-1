"""Submission replacement must pass a separate, predefined resampling gate."""
import importlib.util

import numpy as np

from sdm.paths import REPO_ROOT
from sdm.simplex import check_weights, sample

spec = importlib.util.spec_from_file_location("robust_study", REPO_ROOT / "analysis/robust_search/study.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_audit_gate_keeps_incumbent_on_one_regression_or_negligible_improvement():
    gate = study.passes_replacement_gate
    assert gate(108, 112, 110, 114, [-2, -1.5, -1.0, -1.2, -1.1])
    assert not gate(108, 112, 110, 114, [-3, -3, -3, -3, 0.1])
    assert not gate(108, 112, 110, 114, [-0.1] * 5)
    assert not gate(113, 112, 110, 114, [-2] * 5)
    assert not gate(108, 112, 110, 114, [-2, -2, -2, -2, float("nan")])


def test_integer_export_is_feasible_and_does_not_drift():
    points = sample(100, np.random.default_rng(2026))
    rounded = study.export_weights(points)
    check_weights(rounded)
    np.testing.assert_array_equal(study.export_weights(rounded), rounded)
    assert np.max(abs(points - rounded)) < 4e-8


def test_screen_and_audit_use_disjoint_fixed_seeds():
    assert len(study.SEEDS["screen"]) == len(study.SEEDS["audit"]) == 5
    assert not set(study.SEEDS["screen"]) & set(study.SEEDS["audit"])
