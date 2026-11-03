"""Guard actual-radius and cross-model-reference semantics in the candidate audit."""
import importlib.util
from pathlib import Path

import numpy as np

from sdm.simplex import check_weights

spec = importlib.util.spec_from_file_location('perturbation_audit',
    Path(__file__).parents[1] / 'analysis/perturbation_audit/audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_exact_radius_is_preserved_at_simplex_corners():
    center = np.tile([1., 0, 0, 0, 0, 0, 0], 3)
    unit = audit.directions(2027, 128)
    for radius in (.02, .05):
        nearby = audit.perturb(center, unit, radius, 'project-exact')
        check_weights(nearby)
        np.testing.assert_allclose(np.linalg.norm(nearby - center, axis=1), radius,
                                   atol=1e-10, rtol=0)
        nominal = audit.perturb(center, unit, radius, 'project-nominal')
        assert np.median(np.linalg.norm(nominal - center, axis=1)) < .9 * radius


def test_directions_pair_across_candidates_and_have_zero_building_mass():
    first = audit.directions(2028, 40)
    np.testing.assert_array_equal(first, audit.directions(2028, 40))
    np.testing.assert_allclose(first.reshape(-1, 3, 7).sum(axis=2), 0, atol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1)


def test_worst_delta_uses_each_judges_own_reference_and_requires_all_to_improve():
    # One judge improves by10 but another by only2: this point must fail the3-unit rule.
    predictions = np.array([[90., 118., 127.], [96., 116., 126.]])
    reference = np.array([100., 120., 130.])
    delta = audit.worst_difference(predictions, reference)
    np.testing.assert_array_equal(delta, [-2., -4.])
    np.testing.assert_array_equal(delta < -3, [False, True])
