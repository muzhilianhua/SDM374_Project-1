"""Decision-relevant tests for the isolated experimental v3.3 protocol."""
import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = importlib.util.spec_from_file_location('v33_protocol', os.path.join(ROOT, 'analysis/v33_protocol/protocol.py'))
if hasattr(importlib.util, 'module_from_spec'):
    v33 = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(v33)
else:
    # The pure scoring tests can run on the host's legacy Python without ML dependencies.
    v33 = SPEC.loader.load_module()


class ScoreTests(unittest.TestCase):
    def score(self, predictions):
        return v33.score_task([120.0, 130.0], {'a': 115.0, 'b': 120.0, 'c': 125.0}, predictions)

    def test_perfect_identification_is_100(self):
        result = self.score({'a': 100.0, 'b': 101.0, 'c': 102.0})
        self.assertEqual(result['score'], 100.0)
        self.assertTrue(result['actual_breakthrough'])

    def test_known_best_performance_is_zero(self):
        self.assertEqual(self.score({'a': 121.0, 'b': 119.0, 'c': 122.0})['score'], 0.0)

    def test_failure_is_negative_not_clipped(self):
        result = self.score({'a': 130.0, 'b': 129.0, 'c': 110.0})
        self.assertEqual(result['score'], -100.0)
        self.assertTrue(result['false_breakthrough'])
        self.assertEqual(result['regret'], 10.0)

    def test_calibration_does_not_replace_actual_utility(self):
        first = self.score({'a': 118.0, 'b': 119.0, 'c': 120.0})
        second = self.score({'a': 138.0, 'b': 139.0, 'c': 140.0})
        self.assertEqual(first['score'], second['score'])
        self.assertNotEqual(first['predicted_breakthrough'], second['predicted_breakthrough'])

    def test_exact_ties_use_id_not_input_order(self):
        result = v33.score_task([120.0], {'b': 125.0, 'a': 115.0}, {'b': 110.0, 'a': 110.0})
        self.assertEqual(result['selected_id'], 'a')

    def test_opportunity_normalization(self):
        result = v33.score_task([125.0], {'a': 115.0, 'b': 120.0}, {'a': 122.0, 'b': 121.0})
        self.assertEqual(result['score'], 50.0)

    def test_no_opportunity_fails_instead_of_silent_skip(self):
        with self.assertRaises(ValueError):
            v33.score_task([110.0], {'a': 115.0}, {'a': 100.0})

    def test_nonfinite_predictions_fail(self):
        for invalid in [float('nan'), float('inf'), -float('inf')]:
            with self.assertRaises(ValueError):
                self.score({'a': invalid, 'b': 120.0, 'c': 125.0})

    def test_missing_or_extra_candidate_fails(self):
        for predictions in [{'a': 110.0}, {'a': 110.0, 'b': 120.0, 'c': 125.0, 'd': 130.0}]:
            with self.assertRaises(ValueError):
                self.score(predictions)


class FrozenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.manifest = v33.load_release()

    def test_complete_disjoint_outer_and_inner_membership(self):
        canonical = set(self.data)
        self.assertEqual(len(self.manifest['tasks']), 9)
        for task in self.manifest['tasks']:
            train, test = set(task['train_ids']), set(task['test_ids'])
            self.assertFalse(train & test)
            self.assertEqual(train | test, canonical)
            self.assertEqual(len(test), 410)
            validations = []
            for inner in task['inner_splits']:
                fit, validation = set(inner['train_ids']), set(inner['validation_ids'])
                self.assertFalse(fit & validation)
                self.assertEqual(fit | validation, train)
                self.assertFalse((fit | validation) & test)
                validations.extend(validation)
            self.assertEqual(len(validations), len(set(validations)))
            self.assertEqual(set(validations), train)

    def test_every_task_has_real_breakthrough_opportunity(self):
        for task in self.manifest['tasks']:
            self.assertGreater(task['train_best'], task['test_best'])

    def test_tampered_membership_is_rejected(self):
        tampered = copy.deepcopy(self.manifest)
        task = tampered['tasks'][0]
        task['test_ids'][0] = task['train_ids'][0]
        with self.assertRaises(ValueError):
            v33.validate_manifest(tampered, self.data)

    def truth_predictions(self):
        return [{'split_id': task['split_id'], 'candidate_id': cid,
                 'predicted_mean': self.data[cid]['y']}
                for task in self.manifest['tasks'] for cid in task['test_ids']]

    def test_oracle_sanity_not_a_real_model_result(self):
        metrics, _ = v33.evaluate_rows(self.truth_predictions(), self.manifest, self.data)
        self.assertEqual(metrics['score'], 100.0)

    def test_duplicates_and_incomplete_tasks_rejected(self):
        rows = self.truth_predictions()
        for invalid in [rows[:-1], rows + rows[:1], rows + [dict(rows[0], split_id='v32_fold')]]:
            with self.assertRaises(ValueError):
                v33.evaluate_rows(invalid, self.manifest, self.data)

    def test_family_weights_not_task_counts(self):
        rows = self.truth_predictions()
        for row in rows:
            if row['split_id'].startswith('region'):
                row['predicted_mean'] = -row['predicted_mean']
        metrics, _ = v33.evaluate_rows(rows, self.manifest, self.data)
        families = metrics['families']
        self.assertAlmostEqual(metrics['score'],
                               0.5 * families['hidden_elite']['score'] + 0.5 * families['local_region']['score'])
        self.assertLess(metrics['score'], 100.0)

    def test_external_prediction_cli_records_provenance(self):
        # Oracle rows test the transport/scorer only, never a fitted model's quality.
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, 'oracle_sanity_only.csv')
            output = os.path.join(temporary, 'output')
            v33.write_csv(path, self.truth_predictions())
            subprocess.check_output([sys.executable,
                os.path.join(ROOT, 'analysis/v33_protocol/evaluate.py'),
                '--predictions', path, '--output', output], cwd=ROOT)
            with open(os.path.join(output, 'metrics.json')) as stream:
                self.assertEqual(json.load(stream)['score'], 100.0)
            with open(os.path.join(output, 'provenance.json')) as stream:
                record = json.load(stream)
            self.assertEqual(record['status'], 'complete')
            self.assertEqual(record['mode'], 'external_predictions')
            self.assertEqual(record['simulator_calls'], 0)
            self.assertEqual(record['fitting_exclusion'], 'provider-declared, not independently proven')
            self.assertEqual(set(record['outputs']), {'predictions.csv', 'task_scores.csv', 'metrics.json'})


if __name__ == '__main__':
    unittest.main()
