"""Input contracts and a regression against an independently saved hourly trace."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from run import parse_candidates
from runtime import initialize, trace_week, verify_assets


class InputContractTests(unittest.TestCase):
    def setUp(self):
        self.flat = [1.0] * 21
        self.matrix = [self.flat[i:i + 7] for i in range(0, 21, 7)]

    def test_single_and_batch_protocol(self):
        for value in (self.flat, self.matrix, {"weights": self.matrix}):
            self.assertEqual(parse_candidates(json.dumps(value)), [self.matrix])
        self.assertEqual(parse_candidates(json.dumps({"weights": [self.flat] * 50})), [self.matrix] * 50)

    def test_rejects_invalid_candidates(self):
        for value in ([], [self.flat] * 51, [0] * 21, [-1] + [1] * 20,
                      [True] + [1] * 20, [float("nan")] + [1] * 20,
                      {"weights": self.flat, "unexpected": 1}):
            with self.subTest(value=str(value)[:60]), self.assertRaises(ValueError):
                parse_candidates(json.dumps(value))

    def test_release_assets_match_manifest(self):
        self.assertEqual(verify_assets(), 124)


class OriginalTrajectoryTests(unittest.TestCase):
    def test_week_one_matches_saved_complete_evaluator(self):
        import numpy as np
        with np.load(HERE / "evidence/exact_local_best.npz") as reference:
            weights = reference["weights"].tolist()
            expected = reference["nets"][0].copy()
        initialize()
        actual = trace_week((weights, 1))
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
