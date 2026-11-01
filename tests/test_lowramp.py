"""Check training-only weighting and guarded inner selection, including failure cases."""
import numpy as np
import pytest

from sdm.models.lowramp import training_weights, choose_trial
from sdm.registry import build_model


def test_soft_weights_preserve_support_and_are_monotone_and_affine_invariant():
    y = np.arange(100.0)
    mask, w = training_weights(y, "soft", 3, .2, .3)
    assert mask.all() and np.all((w >= 1) & (w <= 4)) and np.all(np.diff(w) <= 0)
    np.testing.assert_allclose(w, training_weights(y * 5 + 100, "soft", 3, .2, .3)[1])
    np.testing.assert_array_equal(training_weights(y, "soft", 0, .2, .3)[1], np.ones(100))


def test_trimmed_comparator_drops_exact_upper_decile_but_step_keeps_every_row():
    y = np.arange(101.0)[::-1]
    mask, w = training_weights(y, "trimmed-step", 2, .2, .3)
    assert mask.sum() == 90 and (w == 3).sum() == 20
    assert y[mask].max() < y[~mask].min()
    assert training_weights(y, "step", 2, .2, .3)[0].all()


def test_guard_rejects_high_score_with_global_error_regression_and_prefers_simple_ties():
    rows = [dict(alpha=0, temperature=.3, score=70, global_mae=5, local_mae=6),
            dict(alpha=3, temperature=.1, score=80, global_mae=6, local_mae=5),
            dict(alpha=1, temperature=.3, score=70.2, global_mae=5, local_mae=6)]
    assert choose_trial(rows)["alpha"] == 0
    rows[2]["score"] = 71
    assert choose_trial(rows)["alpha"] == 1


@pytest.mark.parametrize("family", ["histgb", "xgboost"])
def test_zero_alpha_is_identical_to_unweighted_and_training_rows_are_preserved(dataset, family):
    params = dict(family=family, iterations=20, calib_folds=2)
    soft = build_model("lowramp-boosted", dict(**params, strategy="soft", alpha=0))
    plain = build_model("lowramp-boosted", dict(**params, strategy="unweighted"))
    for model in (soft, plain):
        model.fit(dataset.X[:400], dataset.y[:400])
    np.testing.assert_array_equal(soft.predict(dataset.X[400:450])[0], plain.predict(dataset.X[400:450])[0])
    assert soft.metrics["training_rows"] == 400 and soft.metrics["dropped_rows"] == 0


def test_calibration_recomputes_weights_without_validation_targets(dataset, monkeypatch):
    import sdm.models.lowramp as module
    calls, original = [], module.training_weights
    def record(y, *args):
        calls.append(np.array(y))
        return original(y, *args)
    monkeypatch.setattr(module, "training_weights", record)
    model = build_model("lowramp-boosted", dict(iterations=10, calib_folds=2))
    model.fit(dataset.X[:400], dataset.y[:400])
    assert [len(y) for y in calls] == [200, 200, 400, 400]
    assert not np.array_equal(calls[0], calls[1])


def test_nested_selection_keeps_trials_and_is_deterministic(dataset):
    params = dict(family="histgb", tune=True, iterations=10, calib_folds=2,
                  alpha_grid=(1.0,), temperature_grid=(.3,))
    first = build_model("lowramp-boosted", params).fit(dataset.X, dataset.y)
    again = build_model("lowramp-boosted", params).fit(dataset.X, dataset.y)
    assert first.trials_ == again.trials_ and len(first.trials_) == 2
    assert first.selected_ == again.selected_


def test_invalid_weighting_settings_fail():
    with pytest.raises(ValueError):
        training_weights(np.arange(10.), "soft", 2, .2, 0)
