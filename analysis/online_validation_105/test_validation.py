"""Offline tests for the independent validation pipeline; no real solver is invoked."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from run_online import (DIM, ROOT, SelectionState, append_evaluations, build_elites, calibration_coefficient,
                        dominant_structures, duplicate_mask, generate_structure, load_evaluations,
                        make_candidates, normalise, prepare_resume, quotas_for_call, select_batch,
                        valid_weight_mask, append_jsonl, utc_now, SolverEvaluator, SolverError,
                        parse_solver_output)


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        cls.rng = np.random.default_rng(712)
        cls.weights = normalise(cls.rng.gamma(2.0, 1.0, size=(160, DIM)))
        cls.actual = np.sum((cls.weights - 1 / 7) ** 2, axis=1) * 100 + cls.rng.normal(0, 0.01, 160)

    def test_simplex_and_dominant(self):
        self.assertTrue(valid_weight_mask(self.weights).all())
        structures = dominant_structures(self.weights)
        for row, structure in zip(self.weights.reshape(-1, 3, 7), structures):
            for building in range(3):
                self.assertEqual(int(np.argmax(row[building])), int(structure[building]))

    def test_linf_deduplication_including_boundary_cells(self):
        reference = np.full((1, DIM), 1 / 7)
        near = reference.copy()
        near[0, 0] += 0.9e-9
        near[0, 1] -= 0.9e-9
        far = reference.copy()
        far[0, 0] += 1.1e-9
        result = duplicate_mask(np.vstack([reference, near, far]), reference, 1e-9)
        self.assertEqual(result.tolist(), [True, True, False])

    def test_multiple_structure_elites(self):
        synthetic = self.weights[:30].copy()
        synthetic[0, :7] = [0.8, 0.1, 0.1, 0, 0, 0, 0]
        synthetic[1, :7] = [0.1, 0.8, 0.1, 0, 0, 0, 0]
        actual = self.actual[:30].copy()
        actual[0], actual[1] = -2, -1
        elites = build_elites(synthetic, actual, self.config)
        self.assertGreaterEqual(len(elites["weights"]), 2)
        self.assertEqual(elites["structures"][0].tolist(), dominant_structures(synthetic[[0]])[0].tolist())
        self.assertEqual(elites["structures"][1].tolist(), dominant_structures(synthetic[[1]])[0].tolist())

    def test_structure_mutation_freezes_unselected_buildings(self):
        elites = build_elites(self.weights, self.actual, self.config)
        candidates, details = generate_structure(elites, 30, self.config, np.random.default_rng(42), True)
        for row, (base_index, changed) in enumerate(zip(details["base_indices"], details["changed"])):
            base = elites["weights"][base_index]
            for building in range(3):
                if building not in changed:
                    np.testing.assert_array_equal(candidates[row, building * 7:(building + 1) * 7],
                                                  base[building * 7:(building + 1) * 7])
            for building in changed:
                self.assertNotEqual(int(dominant_structures(elites["weights"][[base_index]])[0, building]), -1)
        self.assertTrue(valid_weight_mask(candidates).all())

    def test_seed_reproducibility(self):
        first = make_candidates(self.weights, self.actual, self.config, 3, 0, 1, 80, 80)
        second = make_candidates(self.weights, self.actual, self.config, 3, 0, 1, 80, 80)
        np.testing.assert_array_equal(first["weights"], second["weights"])

    def test_real_quotas_and_improvement_lookback(self):
        self.assertEqual(quotas_for_call(self.config, 1, [], 50), {"exploit": 15, "structure": 25, "explore": 10})
        calls = [{"status": "ok", "improved_global": False}, {"status": "ok", "improved_global": True}]
        self.assertEqual(quotas_for_call(self.config, 2, calls, 50), {"exploit": 30, "structure": 10, "explore": 10})
        old = [{"status": "ok", "improved_global": True}] + [{"status": "ok", "improved_global": False}] * 3
        self.assertEqual(quotas_for_call(self.config, 5, old, 50), {"exploit": 20, "structure": 20, "explore": 10})

    def test_selection_caps_and_determinism(self):
        pool = make_candidates(self.weights, self.actual, self.config, 1, 0, 1, 700, 700)
        mean = np.sum(pool["weights"], axis=1) + np.sin(np.arange(len(pool["weights"])))
        std = np.full(len(mean), 0.3)
        nearest = np.full(len(mean), 2.0)
        quotas = {"exploit": 15, "structure": 25, "explore": 10}
        first = select_batch(pool, mean, std, nearest, 0.05, quotas, self.config)
        second = select_batch(pool, mean, std, nearest, 0.05, quotas, self.config)
        self.assertEqual(first.indices, second.indices)
        self.assertEqual(len(first.points), 50)
        self.assertEqual(first.labels, ["exploit"] * 15 + ["structure"] * 25 + ["explore"] * 10)
        for label, cap in (("exploit", 5), ("structure", 1), ("explore", 2)):
            self.assertLessEqual(max(first.counts[label].values()), cap)
        self.assertTrue(valid_weight_mask(np.asarray(first.points)).all())

    def test_calibration_uses_recent_online_errors(self):
        rows = pd.DataFrame({"actual_R": [2.0, 4.0], "predicted_mean": [0.0, 0.0],
                             "predicted_std_raw": [1.0, 2.0]})
        self.assertAlmostEqual(calibration_coefficient(rows, self.config), np.sqrt(4.0))
        self.assertEqual(calibration_coefficient(rows.iloc[:0], self.config), 1.0)

    def test_evaluation_log_roundtrip_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            rows = [{key: 0 for key in __import__("run_online").EVALUATION_COLUMNS}]
            rows[0].update({name: 1 / 7 for name in __import__("run_online").WEIGHT_COLUMNS})
            rows[0].update({"call": 1, "point": 0, "actual_R": 123.0, "predicted_mean": 124.0,
                            "predicted_std_raw": 1.0, "predicted_std": 2.0, "slot": "exploit"})
            append_evaluations(run_dir, rows)
            loaded = load_evaluations(run_dir)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded.loc[0, "actual_R"], 123.0)
            append_jsonl(run_dir / "attempts.jsonl", {"status": "started", "call": 1,
                         "timestamp": utc_now(), "weights": self.weights[:2].tolist()})
            append_jsonl(run_dir / "attempts.jsonl", {"status": "failed", "call": 1,
                         "timestamp": utc_now(), "error": "intentional test"})
            _, sent, next_call = prepare_resume(run_dir, False)
            self.assertEqual(next_call, 2)
            self.assertEqual(len(sent), 2)
            self.assertTrue(duplicate_mask(self.weights[:2], sent).all())

    def test_unresolved_attempt_requires_explicit_acceptance(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            append_jsonl(run_dir / "attempts.jsonl", {"status": "started", "call": 1,
                         "timestamp": utc_now(), "weights": self.weights[:1].tolist()})
            with self.assertRaises(RuntimeError):
                prepare_resume(run_dir, False)
            _, sent, next_call = prepare_resume(run_dir, True)
            self.assertEqual(next_call, 2)
            self.assertEqual(len(sent), 1)
            statuses = [json.loads(line)["status"] for line in (run_dir / "attempts.jsonl").read_text().splitlines()]
            self.assertEqual(statuses, ["started", "lost"])

    def test_parse_serve_json_lines_with_log_lines(self):
        stdout = 'serve ready\n{"R": 123.5}\nnot JSON\n{"R": 120.25}\n'
        self.assertEqual(parse_solver_output(stdout), [123.5, 120.25])
        with self.assertRaises(SolverError):
            parse_solver_output('serve ready\n{"R": null}\n')
        with self.assertRaises(SolverError):
            parse_solver_output('{"error": "worker failed"}\n')

    def test_serve_evaluator_sends_flat_json_lines_and_parses_replies(self):
        with tempfile.TemporaryDirectory() as temporary:
            script = Path(temporary) / "fake_solver.py"
            script.write_text(
                'import json, sys\n'
                'print("serve ready", flush=True)\n'
                'for line in sys.stdin:\n'
                '    weights = json.loads(line)\n'
                '    assert isinstance(weights, list) and len(weights) == 21\n'
                '    print(json.dumps({"R": 100 + sum(weights)}), flush=True)\n',
                encoding="utf-8")
            evaluator = SolverEvaluator(
                self.config, command=[sys.executable, "-u", str(script), "--serve"], verify_hashes=False)
            actual, info = evaluator.evaluate(self.weights[:3])
            np.testing.assert_allclose(actual, 100 + self.weights[:3].sum(axis=1))
            self.assertEqual(info["response_rows"], 3)
            self.assertEqual(info["transport"], "embedded --serve JSON-lines")


if __name__ == "__main__":
    unittest.main(verbosity=2)
