"""V3.3.2 must rotate elite training access and handle non-breakthrough tests honestly."""
import copy
import importlib.util
from pathlib import Path
import unittest

BASE = Path(__file__).resolve().parents[1] / 'analysis/v33_protocol/release_v2'
spec = importlib.util.spec_from_file_location('v332_protocol_test', BASE / 'protocol.py')
v33 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v33)


class ScoreTests(unittest.TestCase):
    def test_supported_test_accepts_training_global_best(self):
        score = v33.score_task([110.], {'a': 115., 'b': 120.}, {'a': 115., 'b': 120.},
                               v33.P1_ACTUAL, mode='within_test')
        self.assertEqual(score['score'], 100.)
        self.assertFalse(score['actual_breakthrough'])
        self.assertEqual(score['points_best_identification'], 35.)

    def test_supported_regret_five_is_half_recognition(self):
        score = v33.score_task([110.], {'a': 115., 'b': 120.}, {'a': 122., 'b': 119.},
                               v33.P1_ACTUAL, mode='within_test')
        self.assertEqual(score['points_best_identification'], 17.5)

    def test_stress_still_requires_opportunity(self):
        with self.assertRaises(ValueError):
            v33.score_task([110.], {'a': 115.}, {'a': 115.}, v33.P1_ACTUAL)

    def test_feedback_does_not_score_memorized_p1(self):
        score = v33.score_task([110.], {'a': 115.}, {'a': 115.}, v33.P1_ACTUAL,
                               feedback=True, mode='within_test')
        self.assertEqual(score['score'], 95.)
        self.assertEqual(score['points_p1_accuracy'], 0.)

    def test_bad_numeric_prediction_loses_points_even_with_correct_rank(self):
        score = v33.score_task([110.], {'a': 115.}, {'a': 90.}, v33.P1_ACTUAL, mode='within_test')
        self.assertEqual(score['points_best_identification'], 35.)
        self.assertLess(score['points_true_low_accuracy'], 30.)

    def test_invalid_mode_or_nonfinite_input_fails(self):
        for kwargs in [dict(mode='unknown'), dict(p1_prediction=float('inf'))]:
            args = dict(train_values=[120.], actual={'a': 115.}, predictions={'a': 115.}, p1_prediction=v33.P1_ACTUAL)
            args.update(kwargs)
            with self.assertRaises(ValueError):
                v33.score_task(**args)


class FrozenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.manifest, _ = v33.load_release()

    def test_elite_rotation_and_reserved_neighbor_support(self):
        tasks = [t for t in self.manifest['tasks'] if t['family'] == 'supported_low']
        elite = sorted(self.data, key=lambda cid: (self.data[cid]['y'], cid))[:20]
        self.assertEqual(len(tasks), 5)
        for cid in elite:
            self.assertEqual(sum(cid in t['test_ids'] for t in tasks), 1)
            self.assertEqual(sum(cid in t['train_ids'] for t in tasks), 4)
        for task in tasks:
            self.assertEqual(len(set(elite[:5]) & set(task['train_ids'])), 4)
            for cid, neighbor in task['design']['reserved_neighbors'].items():
                self.assertIn(cid, task['test_ids'])
                self.assertIn(neighbor, task['train_ids'])

    def test_all_outer_and_inner_boundaries(self):
        self.assertEqual(len(self.manifest['tasks']), 14)
        for task in self.manifest['tasks']:
            train, test = set(task['train_ids']), set(task['test_ids'])
            self.assertFalse(train & test)
            self.assertNotIn('P1', train | test)
            self.assertEqual(train | test, set(self.data))
            self.assertEqual(len(train), 1638)
            self.assertEqual(len(test), 410)
            seen = []
            for inner in task['inner_splits']:
                fit, validation = set(inner['train_ids']), set(inner['validation_ids'])
                self.assertFalse(fit & validation)
                self.assertEqual(fit | validation, train)
                self.assertFalse(test & (fit | validation))
                seen.extend(validation)
            self.assertEqual(len(seen), len(set(seen)))
            self.assertEqual(set(seen), train)

    def test_pressure_tasks_keep_identical_v331_membership(self):
        _, old_manifest, _ = v33.old.load_release()
        for old_task, new_task in zip(old_manifest['tasks'], self.manifest['tasks'][:9]):
            self.assertEqual(old_task['train_ids'], new_task['train_ids'])
            self.assertEqual(old_task['test_ids'], new_task['test_ids'])

    def rows(self):
        return [{'split_id': t['split_id'], 'candidate_id': cid,
                 'predicted_mean': self.data[cid]['y'] if cid != 'P1' else v33.P1_ACTUAL}
                for t in self.manifest['tasks'] for cid in t['test_ids'] + ['P1']]

    def test_perfect_predictions_and_25_25_50_aggregation(self):
        rows = self.rows()
        metrics, _ = v33.evaluate_rows(rows, self.manifest, self.data)
        self.assertEqual(metrics['score'], 100.)
        for row in rows:
            if row['split_id'].startswith('supported') and row['candidate_id'] != 'P1':
                row['predicted_mean'] = -row['predicted_mean']
        metrics, _ = v33.evaluate_rows(rows, self.manifest, self.data)
        f = metrics['families']
        self.assertAlmostEqual(metrics['score'], .25*f['hidden_elite']['score'] + .25*f['local_region']['score'] + .5*f['supported_low']['score'])

    def test_missing_or_duplicated_records_fail(self):
        rows = self.rows()
        for invalid in [rows[:-1], rows + rows[:1]]:
            with self.assertRaises(ValueError):
                v33.evaluate_rows(invalid, self.manifest, self.data)

    def test_tampered_rotation_is_rejected(self):
        manifest = copy.deepcopy(self.manifest)
        manifest['tasks'][-1]['train_ids'][0] = manifest['tasks'][-1]['test_ids'][0]
        with self.assertRaises(ValueError):
            v33.validate_manifest(manifest, self.data)


if __name__ == '__main__':
    unittest.main()
