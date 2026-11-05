# -*- coding: utf-8 -*-
"""接口回归测试：标准差语义（blend 的可信 std 必须被拒绝，均值必须逐位不变）。

评审 P2-B 的四条：
  1. 四模型 `return_std` 默认行为（blend 抛错，三个 GP 模型正常返回后验 σ）；
  2. blend 的纯均值预测与修复前逐位相同（p01969 / 第一组离线提交，容差 1e-9）；
  3. `return_std_heuristic=True` 返回值等于 (1-α)·σ_gp，且语义标 heuristic；
  4. `return_std=True` 与 `return_std_heuristic=True` 同时给必须报错。

运行：python -m unittest discover -s tests -v   （在包根目录）
"""
import sys
import unittest
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
import modellib as M  # noqa: E402

# p01969 的 21 维权重（规范顺序 b1_cost … b3_resilience）
W21 = np.array([
    0.1882019242483394, 0.0636046642005978, 0.202188120195192,
    0.2143194240587854, 0.033248929385831, 0.2198126737971956,
    0.0786242641140584, 0.2358813109739701, 0.1410031076302182,
    0.0820577008032854, 0.0553050045816525, 0.1297430575559024,
    0.2875770324599347, 0.0684327859950362, 0.1625895187677247,
    0.0292558054701189, 0.0445871805653544, 0.0695964909613512,
    0.1134199167822772, 0.3471329386268938, 0.2334181488262795,
])
GP_MODELS = ("gp-ard", "tail5opt", "hp_wlearn")
BLEND = "blend_xgb_w20x3"

# C03（第一组离线提交）的 3×7 权重，内联进测试，避免干净检出时依赖仓库根文件。
W_SUB = np.array([
    [0.0669105903, 3.694524401e-59, 0.0628613854, 0.3267870193, 0.0173857488, 0.5178950113, 0.0081602449],
    [0.1584423481, 0.0015068326, 0.1819948397, 0.0000314801, 0.0, 0.5172659787, 0.1407585208],
    [0.0000000005, 0.0105767965, 0.1064174031, 0.2395106988, 0.0000000001, 0.3124662746, 0.3310288264],
])

# 修复前的纯均值预测（改动只动 std 语义，这两个数必须逐位不变）
P01969_MEAN = 115.972963704331
SUBMISSION_MEAN = 119.577258208507
TOL = 1e-9


class StdContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundles = {name: M.load_model(PKG / "models" / name)
                       for name in (*GP_MODELS, BLEND)}
        cls.x = W21.reshape(1, -1)
        cls.x_sub = W_SUB.reshape(1, -1)

    # 1) 默认行为：GP 模型正常，blend 拒绝
    def test_default_std_behavior(self):
        for name in GP_MODELS:
            with self.subTest(model=name):
                mean, sd = self.bundles[name].predict(self.x)
                self.assertEqual(mean.shape, (1,))
                self.assertEqual(sd.shape, (1,))
                self.assertTrue(np.all(sd >= 0.0))
                self.assertEqual(self.bundles[name].uncertainty_kind, "gp_posterior")
        with self.subTest(model=BLEND):
            self.assertEqual(self.bundles[BLEND].uncertainty_kind, "heuristic")
            with self.assertRaises(M.ContractError) as ctx:
                self.bundles[BLEND].predict(self.x)          # 默认 return_std=True
            self.assertIn("启发式近似", str(ctx.exception))
            self.assertIn("return_std_heuristic", str(ctx.exception))

    # 2) 纯均值预测与修复前逐位相同
    def test_blend_mean_unchanged(self):
        blend = self.bundles[BLEND]
        mean = blend.predict(self.x, return_std=False)
        self.assertAlmostEqual(float(mean[0]), P01969_MEAN, delta=TOL)
        self.assertEqual(mean.shape, (1,))
        mean_sub = blend.predict(self.x_sub, return_std=False)
        self.assertAlmostEqual(float(mean_sub[0]), SUBMISSION_MEAN, delta=TOL)
        # heuristic 入口返回的均值必须与纯均值逐位相同
        mean_h, _ = blend.predict(self.x, return_std_heuristic=True)
        np.testing.assert_array_equal(mean_h, mean)

    # 3) heuristic = (1-α)·σ_gp，且语义标 heuristic
    def test_heuristic_std_equals_scaled_gp(self):
        blend = self.bundles[BLEND]
        mean_h, sd_h = blend.predict(self.x, return_std_heuristic=True)
        g = blend.payload["gp"]
        _, Ks, mz = blend._gp_mean(g, self.x)
        sd_gp = blend._gp_std(g, self.x, Ks, mz)
        alpha = float(blend.payload["alpha_mix"])
        np.testing.assert_allclose(sd_h, (1.0 - alpha) * sd_gp, rtol=1e-12, atol=0.0)
        self.assertEqual(blend.uncertainty_kind, "heuristic")

    # 4) 两个参数同时给必须报错，不能静默取一个
    def test_both_std_flags_rejected(self):
        for name, bundle in self.bundles.items():
            with self.subTest(model=name):
                with self.assertRaises(M.ContractError) as ctx:
                    bundle.predict(self.x, return_std=True, return_std_heuristic=True)
                self.assertIn("不能同时为 True", str(ctx.exception))

    # 额外：GP 模型不接受 heuristic 入口（避免把后验 σ 误当启发式）
    def test_gp_rejects_heuristic_flag(self):
        for name in GP_MODELS:
            with self.subTest(model=name):
                with self.assertRaises(M.ContractError):
                    self.bundles[name].predict(self.x, return_std_heuristic=True)


if __name__ == "__main__":
    unittest.main()
