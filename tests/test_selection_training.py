"""Checks for three-way selection tuning, nested raw predictions and export semantics."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sdm.models.selection_stack import SelectionStack, measured_pools, trial_rows
from sdm.registry import build_model
from sdm.simplex import check_weights

spec = importlib.util.spec_from_file_location('selection_precision',
    Path(__file__).resolve().parents[1]/'analysis/selection_training/precision.py')
precision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(precision)


def test_trial_roles_disjoint_and_validation_labels_excluded(dataset):
    X, y = dataset.X[:60], dataset.y[:60].copy()
    model = SelectionStack(SelectionStack.Params(iterations=8, selection_repeats=1))
    rows = next(trial_rows(len(y), 2026, 1))
    assert sorted(np.concatenate(rows)) == list(range(len(y)))
    first = model._trial(X, y, rows)
    y[rows[2]] += 1000
    second = model._trial(X, y, rows)
    np.testing.assert_array_equal(first['raw'], second['raw'])
    np.testing.assert_array_equal(first['delta'], second['delta'])
    assert not np.array_equal(first['actual'], second['actual'])


def test_selection_tunes_measured_pool_regret_and_prefers_smaller_tie():
    model = SelectionStack(SelectionStack.Params(pool_size=50))
    trial = {'raw': np.array([0., 1., 2.]), 'delta': np.array([3., 0., 0.]),
             'actual': np.array([5., 1., 2.])}
    strength, records = model._choose_strength([trial])
    assert strength == .5
    assert records[0]['pool_regret'] == 4.
    assert next(r for r in records if r['strength'] == strength)['pool_regret'] == 0.
    trial['delta'] *= -1
    assert model._choose_strength([trial])[0] == 0.


def test_nested_raw_prediction_excludes_own_label_and_scheduling_is_deterministic(dataset):
    X, y = dataset.X[:60], dataset.y[:60].copy()
    params = {'iterations': 8, 'selection_repeats': 1}
    serial = build_model('selection-stack', {**params, 'fit_workers': 1}).fit(X, y)
    parallel = build_model('selection-stack', {**params, 'fit_workers': 3}).fit(X, y)
    for a,b in zip(serial.predict(X), parallel.predict(X)):
        np.testing.assert_array_equal(a, b)
    y[7] += 1000
    changed = build_model('selection-stack', params).fit(X, y)
    assert serial.calibration_prediction_[7] == changed.calibration_prediction_[7]


def test_zero_strength_recovers_original_mean_exactly(dataset):
    X, y = dataset.X[:60], dataset.y[:60]
    corrected = build_model('selection-stack', {'iterations': 8, 'strengths': [0.],
                                               'selection_repeats': 1}).fit(X, y)
    original = build_model('method-study', {'family': 'ensemble', 'ensemble_mode': 'stack',
                                           'iterations': 8}).fit(X, y)
    np.testing.assert_array_equal(corrected.predict(X)[0], original.predict(X)[0])


def test_measured_pools_cover_rows_once_and_do_not_depend_on_labels():
    pools = measured_pools(411, 2042)
    np.testing.assert_array_equal(np.sort(np.concatenate(pools)), np.arange(411))
    assert min(map(len, pools)) >= 50


def test_decimal_exports_sum_exactly_in_integer_units_and_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    raw = rng.dirichlet(np.full(7, .01), size=(12, 3)).reshape(-1, 21)
    for decimals in (4, 6, 8):
        canonical = precision.canonical_weights(raw, decimals)
        check_weights(canonical)
        units = np.rint(canonical.reshape(-1, 7)*10**decimals).astype(int)
        np.testing.assert_array_equal(units.sum(axis=1), 10**decimals)
        path = tmp_path/f'weights-{decimals}.json'
        precision.export_weights(path, canonical, decimals)
        assert 'e-' not in path.read_text()
    np.testing.assert_array_equal(precision.canonical_weights(raw,8),
                                 precision.canonical_weights(precision.canonical_weights(raw,8),8))
    cleaned = precision.canonical_weights(raw, 8, 1e-6)
    assert np.all(cleaned[raw < 1e-6] == 0)


def test_duplicate_batches_preserve_fixed_anchor_without_substituting_ineligible_point():
    first = np.full(21,1/7)
    other = first.copy()
    other[:2] += [.1,-.1]
    points = np.array([first,other,first])
    indices = precision.prioritized_unique(points,np.arange(3),2)
    assert indices.tolist() == [2,1]
    with pytest.raises(ValueError,match='eligibility'):
        precision.prioritized_unique(points,np.array([0,1]),2)


def test_nonzero_correction_warns_even_for_an_existing_fitted_model(dataset):
    model = build_model('selection-stack', {'iterations': 3, 'strengths': [0.],
        'selection_repeats': 1}).fit(dataset.X[:40], dataset.y[:40])
    model.strength_ = .25
    with pytest.warns(RuntimeWarning, match="full-data features"):
        mean, scale = model.predict(dataset.X[:2])
    assert np.isfinite(mean).all() and np.isfinite(scale).all()


def test_zero_strength_does_not_evaluate_correction(dataset, monkeypatch):
    model = build_model('selection-stack', {'iterations': 3, 'strengths': [0.],
        'selection_repeats': 1}).fit(dataset.X[:40], dataset.y[:40])

    def unexpected(features):
        raise AssertionError("disabled correction must not run")

    monkeypatch.setattr(model.correction_, 'predict', unexpected)
    np.testing.assert_array_equal(model.predict(dataset.X[:2])[0],
                                  model.predictor_.predict(dataset.X[:2])[0])
