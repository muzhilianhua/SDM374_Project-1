"""The submission-2 study steps along simplex directions, so every step must keep each building's sum at 1."""
import importlib.util

import numpy as np

from sdm.paths import REPO_ROOT

spec = importlib.util.spec_from_file_location("offline_submission2_study",
                                              REPO_ROOT / "analysis/offline_submission2/study.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_directions_are_unit_and_keep_building_sums(dataset):
    u = study.simplex_directions(500, np.random.default_rng(0))
    np.testing.assert_allclose(np.linalg.norm(u, axis=1), 1.0)
    np.testing.assert_allclose(u.reshape(-1, 3, 7).sum(axis=2), 0.0, atol=1e-12)
    step = dataset.X[0] + 0.01 * u
    np.testing.assert_allclose(step.reshape(-1, 3, 7).sum(axis=2), 1.0, atol=1e-12)
