# -*- coding: utf-8 -*-
"""`--probe-seed` 必须落在真实训练参数空间（sklearn 核的 log(theta)/log bounds）。

训练链路里 `make_optimizer` 消费的是 `build_kernel(...)` 返回核的 `theta` 与 `bounds`
（log 空间）。本测试对 seed 2026/2027/2028 逐个校验：
  - 第一起点就是核的初始 theta，全部起点都由 bounds 内均匀采样；
  - exp 还原后的长度尺度 ≤ 6、噪声 ∈ [0.001, 0.3]（真实边界，以 kernel.bounds 为准）；
  - 2027 的第二起点与 2026 不同（证明确实生效），且仍然合法。

运行：python -m unittest discover -s tests -v   （在包根目录）
"""
import sys
import unittest
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
import train_four_models as T  # noqa: E402

SEEDS = (2026, 2027, 2028)
CONFIG = T.CONFIGS["blend_xgb_w20x3"]


class ProbeSeedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernel, _ = T.build_kernel(CONFIG, n_features=21)
        cls.theta = np.asarray(cls.kernel.theta, float)          # log(length_scale ×7, noise)
        cls.bounds = np.asarray(cls.kernel.bounds, float)        # log 边界
        cls.probes = {seed: T.probe_seed(seed) for seed in SEEDS}

    def starts(self, seed):
        return np.asarray(self.probes[seed]["starts_log"], float)

    def test_first_start_is_kernel_theta(self):
        for seed in SEEDS:
            with self.subTest(seed=seed):
                self.assertEqual(len(self.starts(seed)[0]), len(self.theta))
                np.testing.assert_array_equal(self.starts(seed)[0], self.theta)

    def test_bounds_are_kernel_log_bounds(self):
        for seed in SEEDS:
            with self.subTest(seed=seed):
                np.testing.assert_array_equal(np.asarray(self.probes[seed]["bounds_log"], float),
                                              self.bounds)
                np.testing.assert_allclose(
                    np.exp(self.bounds[:, 0]), [0.03] * 7 + [0.001], rtol=1e-12, atol=0)
                np.testing.assert_allclose(
                    np.exp(self.bounds[:, 1]), [6.0] * 7 + [0.3], rtol=0, atol=1e-12)

    def test_all_starts_within_real_bounds(self):
        lo, hi = np.exp(self.bounds[:, 0]), np.exp(self.bounds[:, 1])
        for seed in SEEDS:
            with self.subTest(seed=seed):
                self.assertTrue(self.probes[seed]["all_starts_within_bounds"])
                starts_raw = np.exp(self.starts(seed))
                self.assertTrue(np.all(starts_raw >= lo))
                self.assertTrue(np.all(starts_raw <= hi))
                self.assertTrue(np.all(starts_raw[:, :-1] <= 6.0))
                self.assertTrue(np.all(starts_raw[:, -1] >= 0.001))
                self.assertTrue(np.all(starts_raw[:, -1] <= 0.3))

    def test_reported_raw_values_are_legal(self):
        for seed in SEEDS:
            for idx, start in enumerate(self.probes[seed]["starts_raw"]):
                with self.subTest(seed=seed, start=idx):
                    length_scales = np.asarray(start["length_scale"], float)
                    self.assertEqual(length_scales.shape, (7,))
                    self.assertTrue(np.all(length_scales >= 0.03 - 1e-9))
                    self.assertTrue(np.all(length_scales <= 6.0 + 1e-9))
                    self.assertGreaterEqual(start["noise"], 0.001 - 1e-9)
                    self.assertLessEqual(start["noise"], 0.3 + 1e-9)

    def test_seed_2027_second_start_differs_and_is_legal(self):
        a = self.starts(2026)[1]
        b = self.starts(2027)[1]
        self.assertFalse(np.array_equal(a, b))
        self.assertFalse(np.array_equal(self.starts(2027)[1], self.starts(2028)[1]))
        raw = np.exp(b)
        self.assertTrue(np.all(raw[:-1] <= 6.0 + 1e-9))
        self.assertTrue(0.001 - 1e-9 <= raw[-1] <= 0.3 + 1e-9)

    def test_seed_metadata_semantics_unchanged(self):
        for seed in SEEDS:
            with self.subTest(seed=seed):
                self.assertEqual(self.probes[seed]["seed_argument"], seed)
                self.assertEqual(self.probes[seed]["seed_effective"], seed)


if __name__ == "__main__":
    unittest.main()
