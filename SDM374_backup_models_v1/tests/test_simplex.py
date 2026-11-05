# -*- coding: utf-8 -*-
"""单纯形契约与 3×7 行号索引回归测试。

1. `predict` / `pack_weights` 入口必须拒绝负权重、每栋和≠1、NaN/inf（抛 ContractError）；
2. 合法候选 p01969 / C01 / C02 / C03 仍通过，且 p01969 四模型预测逐位不变（1e-9）；
3. 3×7 表用 pandas 默认 `RangeIndex`（列头 `Unnamed: 0`、值 0/1/2）应按行序映射楼1→楼3，
   与 21 维向量同一预测；真正不合法的表（3 列、20 列、重复楼栋）仍必须报错。

运行：python -m unittest discover -s tests -v   （在包根目录）
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[1]
PROJECT = PKG.parent
sys.path.insert(0, str(PKG))
import modellib as M  # noqa: E402
import load_and_predict as L  # noqa: E402

W21 = np.array([
    0.1882019242483394, 0.0636046642005978, 0.202188120195192,
    0.2143194240587854, 0.033248929385831, 0.2198126737971956,
    0.0786242641140584, 0.2358813109739701, 0.1410031076302182,
    0.0820577008032854, 0.0553050045816525, 0.1297430575559024,
    0.2875770324599347, 0.0684327859950362, 0.1625895187677247,
    0.0292558054701189, 0.0445871805653544, 0.0695964909613512,
    0.1134199167822772, 0.3471329386268938, 0.2334181488262795,
])
MODELS = ("gp-ard", "tail5opt", "hp_wlearn", "blend_xgb_w20x3")
P01969_MEAN = {"gp-ard": 115.7941456944, "tail5opt": 115.7742110860,
               "hp_wlearn": 115.9229831019, "blend_xgb_w20x3": 115.9729637043}
TOL = 1e-9


class SimplexContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundles = {name: M.load_model(PKG / "models" / name) for name in MODELS}
        cls.x = W21.reshape(1, -1)

    # ── 1) 非法输入必须拒绝 ──
    def test_negative_weights_rejected(self):
        bad = -np.ones((1, M.N_FEATURES))
        with self.assertRaises(M.ContractError) as ctx:
            self.bundles["gp-ard"].predict(bad)
        self.assertIn("非负", str(ctx.exception))
        with self.assertRaises(M.ContractError):
            M.pack_weights(-np.ones(21))

    def test_sum_not_one_rejected(self):
        bad = self.x * 1.001                          # 每栋和 = 1.001
        with self.assertRaises(M.ContractError) as ctx:
            self.bundles["gp-ard"].predict(bad)
        self.assertIn("之和必须为 1", str(ctx.exception))
        with self.assertRaises(M.ContractError):
            M.pack_weights(bad)

    def test_nan_inf_rejected(self):
        for value in (np.nan, np.inf):
            with self.subTest(value=value):
                bad = self.x.copy()
                bad[0, 0] = value
                with self.assertRaises(M.ContractError) as ctx:
                    self.bundles["gp-ard"].predict(bad)
                self.assertIn("NaN", str(ctx.exception))
                with self.assertRaises(M.ContractError):
                    M.pack_weights(bad)

    def test_within_tolerance_accepted(self):
        w = self.x.copy()
        w[0, 0] += 5e-7                               # 楼1 和 1.0000005，楼2 和 0.9999995
        w[0, 7] -= 5e-7
        self.assertEqual(M.pack_weights(w).shape, (1, M.N_FEATURES))

    # ── 2) 合法候选通过，预测逐位不变 ──
    def test_p01969_predictions_unchanged(self):
        for name in MODELS:
            with self.subTest(model=name):
                mean = self.bundles[name].predict(self.x, return_std=False)
                self.assertAlmostEqual(float(mean[0]), P01969_MEAN[name], delta=TOL)

    def test_legal_candidates_pass(self):
        files = [PROJECT / "最终提交候选.json", PROJECT / "最终提交候选_非树模型.json",
                 PROJECT / "第一组离线提交.csv"]
        loaded = [(p.name, L.read_weights(p)) for p in files if p.exists()]
        self.assertTrue(loaded, "未找到 C01–C03 的权重文件")
        for label, w in loaded:
            with self.subTest(candidate=label):
                self.assertEqual(w.shape, (1, M.N_FEATURES))
                for name in MODELS:
                    mean = self.bundles[name].predict(w, return_std=False)
                    self.assertTrue(np.all(np.isfinite(mean)))

    # ── 3) 3×7 + RangeIndex 行号索引 ──
    def _write_3x7_with_rangeindex(self) -> Path:
        df = pd.DataFrame(W21.reshape(M.N_BUILDINGS, M.N_METRICS), columns=M.METRICS)
        path = Path(tempfile.mkdtemp()) / "weights_3x7_rangeindex.csv"
        df.to_csv(path)                              # 默认 index=True → Unnamed: 0, 值 0/1/2
        return path

    def test_3x7_rangeindex_csv_accepted(self):
        path = self._write_3x7_with_rangeindex()
        w = L.read_weights(path)
        np.testing.assert_allclose(w.ravel(), W21, rtol=0.0, atol=0.0)
        mean = self.bundles["gp-ard"].predict(w, return_std=False)
        self.assertAlmostEqual(float(mean[0]), P01969_MEAN["gp-ard"], delta=TOL)

    def test_bad_tables_still_rejected(self):
        # 7 行 × 3 列（指标×建筑写反 / 列数不足）
        with self.assertRaises(M.ContractError):
            M.pack_weights(pd.DataFrame(np.full((7, 3), 1.0 / 3.0)))
        # 20 列（缺一个规范特征）
        with self.assertRaises(M.ContractError):
            M.pack_weights(pd.DataFrame(np.ones((1, 20))))
        # 重复楼栋标签
        df = pd.DataFrame(W21.reshape(3, 7), columns=list(M.METRICS))
        df.insert(0, "building", [1, 1, 2])
        with self.assertRaises(M.ContractError) as ctx:
            L._weights_from_csv(df)
        self.assertIn("重复", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
