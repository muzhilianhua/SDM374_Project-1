"""Frozen v3.3.1 score contracts, P1 isolation and original-lineage trimming."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'analysis/v33_protocol/release_v1'
spec = importlib.util.spec_from_file_location('v331_protocol_test', BASE / 'protocol.py')
v33 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v33)


class ScoreTests(unittest.TestCase):
    def test_perfect_predictions_score_100(self):
        result = v33.score_task([120.], {'a': 115., 'b': 125.}, {'a': 115., 'b': 125.}, v33.P1_ACTUAL)
        self.assertAlmostEqual(result['score'], 100.)

    def test_good_ranking_with_bad_numeric_prediction_loses_accuracy_points(self):
        result = v33.score_task([120.], {'a': 115., 'b': 125.}, {'a': 90., 'b': 120.}, v33.P1_ACTUAL)
        self.assertEqual(result['points_breakthrough'], 35.)
        self.assertEqual(result['true_low_mae'], 15.)
        self.assertAlmostEqual(result['points_true_low_accuracy'], 7.5)
        self.assertLess(result['score'], 100.)

    def test_no_breakthrough_has_zero_recognition_points(self):
        result = v33.score_task([120.], {'a': 115., 'b': 125.}, {'a': 120., 'b': 110.}, v33.P1_ACTUAL)
        self.assertEqual(result['points_breakthrough'], 0.)
        self.assertLess(result['raw_breakthrough_score'], 0.)

    def test_mae_five_is_half_accuracy_points(self):
        result = v33.score_task([120.], {'a': 115.}, {'a': 120.}, v33.P1_ACTUAL + 5.)
        self.assertEqual(result['points_true_low_accuracy'], 15.)
        self.assertEqual(result['points_predicted_low_accuracy'], 15.)
        self.assertEqual(result['points_p1_accuracy'], 2.5)

    def test_empty_predicted_low_does_not_earn_points(self):
        result = v33.score_task([120.], {'a': 115.}, {'a': 140.}, v33.P1_ACTUAL)
        self.assertEqual(result['points_predicted_low_accuracy'], 0.)
        self.assertEqual(result['predicted_low_n'], 0)
        self.assertIsNone(result['predicted_low_mae'])

    def test_false_low_ordinary_point_is_penalized(self):
        result = v33.score_task([120.], {'a': 115., 'b': 140.}, {'a': 118., 'b': 110.}, v33.P1_ACTUAL)
        self.assertEqual(result['points_breakthrough'], 0.)
        self.assertAlmostEqual(result['predicted_low_mae'], 16.5)
        self.assertTrue(result['false_breakthrough'])

    def test_feedback_never_scores_p1(self):
        result = v33.score_task([120.], {'a': 115.}, {'a': 115.}, v33.P1_ACTUAL, feedback=True)
        self.assertAlmostEqual(result['score'], 95.)
        self.assertEqual(result['points_p1_accuracy'], 0.)
        self.assertEqual(result['train_best'], v33.P1_ACTUAL)
        self.assertTrue(result['p1_seen_in_fit'])

    def test_empty_true_low_fails(self):
        with self.assertRaises(ValueError):
            v33.score_task([150.], {'a': 140.}, {'a': 135.}, v33.P1_ACTUAL)

    def test_nonfinite_p1_fails(self):
        with self.assertRaises(ValueError):
            v33.score_task([120.], {'a': 115.}, {'a': 115.}, float('nan'))


class FrozenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.manifest, cls.p1 = v33.load_release()

    def test_p1_excluded_from_all_primary_and_inner_fits(self):
        for task in self.manifest['tasks']:
            train, test = set(task['train_ids']), set(task['test_ids'])
            self.assertNotIn('P1', train | test)
            self.assertFalse(train & test)
            self.assertEqual(len(train), 1638)
            self.assertEqual(len(test), 410)
            validations = []
            for inner in task['inner_splits']:
                fit, validation = set(inner['train_ids']), set(inner['validation_ids'])
                self.assertEqual(fit | validation, train)
                self.assertFalse(fit & validation)
                self.assertFalse((fit | validation) & test)
                validations.extend(validation)
            self.assertEqual(len(validations), len(set(validations)))
            self.assertEqual(set(validations), train)

    def rows(self):
        return [{'split_id': task['split_id'], 'candidate_id': cid,
                 'predicted_mean': self.data[cid]['y'] if cid != 'P1' else v33.P1_ACTUAL}
                for task in self.manifest['tasks'] for cid in task['test_ids'] + ['P1']]

    def test_all_components_and_families_have_frozen_weights(self):
        metrics, scores = v33.evaluate_rows(self.rows(), self.manifest, self.data)
        self.assertEqual(metrics['score'], 100.)
        for component, maximum in v33.POLICY['points'].items():
            self.assertEqual(metrics['points_' + component], maximum)
        self.assertEqual(len(scores), 9)

    def test_missing_or_duplicate_p1_rejected(self):
        rows = self.rows()
        for invalid in [rows[:-1], rows + rows[-1:]]:
            with self.assertRaises(ValueError):
                v33.evaluate_rows(invalid, self.manifest, self.data)

    def test_families_equal_weight_despite_different_task_counts(self):
        rows = self.rows()
        for row in rows:
            if row['split_id'].startswith('region') and row['candidate_id'] != 'P1':
                row['predicted_mean'] = -row['predicted_mean']
        metrics, _ = v33.evaluate_rows(rows, self.manifest, self.data)
        families = metrics['families']
        self.assertAlmostEqual(metrics['score'], .5 * families['hidden_elite']['score'] + .5 * families['local_region']['score'])


class RecipeTests(unittest.TestCase):
    def test_trim_and_low_weight_use_training_labels_only(self):
        import numpy as np
        spec = importlib.util.spec_from_file_location('v331_recipes_test', BASE / 'recipes.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.RunzeRecipe(module.RunzeRecipe.Params(iterations=2, trim_fraction=.8, low_weight=3.))
        X, y = np.full((40, 21), 1./7), np.arange(40., 0., -1)
        model.fit(X, y)
        self.assertEqual(len(model.retained_indices_), 32)
        self.assertLessEqual(max(y[model.retained_indices_]), 32.)
        self.assertEqual(model.metrics['weighted_rows'], 8)
        self.assertFalse(model.metrics['uncertainty_available'])


if __name__ == '__main__':
    unittest.main()
