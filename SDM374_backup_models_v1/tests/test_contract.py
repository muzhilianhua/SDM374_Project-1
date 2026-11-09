# -*- coding: utf-8 -*-
"""契约回归测试：带标签的 3×7 输入必须按标签重排，不能随行列顺序静默变化。

运行：python -m unittest discover -s tests -v   （在包根目录）
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

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
EXPECTED = 115.7941456944
CN = ["电费", "碳排", "峰值", "舒适", "损耗", "波动", "韧性"]
EN = list(M.METRICS)
ROWS_CN = ["第 1 栋", "第 2 栋", "第 3 栋"]
ROWS_EN = ["building_1", "building_2", "building_3"]


def frame(rows, cols):
    return pd.DataFrame(W21.reshape(3, 7), index=rows, columns=cols)


class LabelContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = M.load_model(PKG / "models" / "gp-ard")

    def pred(self, obj):
        return float(np.asarray(self.bundle.predict(M.pack_weights(obj))).reshape(-1)[0])

    def test_four_scrambles_same_prediction(self):
        base = frame(ROWS_EN, EN)
        cases = {
            "行逆序": base.iloc[::-1],
            "列逆序": base.iloc[:, ::-1],
            "行列同时逆序": base.iloc[::-1, ::-1],
            "列名打乱但标签正确": base[["ramping", "cost", "resilience", "peak",
                                       "degradation", "discomfort", "carbon"]],
            "中文表头+乱序": frame(ROWS_CN, CN)[["波动", "电费", "韧性", "峰值", "损耗", "舒适", "碳排"]],
            "英文同义词+大小写": frame(["Bldg_1", "Building 2", "b3"],
                                     ["Cost", "Carbon", "Peak", "Comfort",
                                      "Degradation", "Ramping", "Resilience"]),
        }
        for label, df in cases.items():
            with self.subTest(label=label):
                self.assertAlmostEqual(self.pred(df), EXPECTED, delta=1e-9)

    def test_positional_vector_unchanged(self):
        self.assertAlmostEqual(float(self.bundle.predict(W21)[0]), EXPECTED, delta=1e-9)

    def test_unknown_label(self):
        df = frame(ROWS_EN, EN).rename(columns={"cost": "costX"})
        with self.assertRaises(M.ContractError) as ctx:
            M.pack_weights(df)
        self.assertIn("未知列标签", str(ctx.exception))

    def test_duplicate_label(self):
        df = frame(ROWS_EN, ["cost", "cost", "peak", "discomfort",
                             "degradation", "ramping", "resilience"])
        with self.assertRaises(M.ContractError) as ctx:
            M.pack_weights(df)
        self.assertIn("重复", str(ctx.exception))

    def test_missing_column(self):
        df = frame(ROWS_EN, EN).drop(columns=["resilience"])
        with self.assertRaises(M.ContractError) as ctx:
            M.pack_weights(df)
        self.assertIn("得到 3×6", str(ctx.exception))

    def test_unlabeled_rows_are_rejected(self):
        df = pd.DataFrame(W21.reshape(3, 7), columns=EN)  # RangeIndex 0,1,2
        with self.assertRaises(M.ContractError) as ctx:
            M.pack_weights(df)
        self.assertIn("未知行标签", str(ctx.exception))

    def test_duplicate_building(self):
        df = pd.DataFrame(W21.reshape(3, 7), index=["building_1", "b1", "building_2"], columns=EN)
        with self.assertRaises(M.ContractError) as ctx:
            M.pack_weights(df)
        self.assertIn("建筑标签重复", str(ctx.exception))

    def test_wrong_row_count(self):
        df = pd.DataFrame(np.vstack([W21.reshape(3, 7), W21.reshape(3, 7)[:1]]),
                          index=["b1", "b2", "b3", "b1"], columns=EN)
        with self.assertRaises(M.ContractError) as ctx:
            M.pack_weights(df)
        self.assertIn("需要 3 行建筑", str(ctx.exception))

    def test_flat_21_columns_reordered_by_name(self):
        df = pd.DataFrame([W21], columns=M.FEATURES)
        self.assertAlmostEqual(self.pred(df[list(reversed(M.FEATURES))]), EXPECTED, delta=1e-9)

    def test_flat_21_columns_bad_names(self):
        df = pd.DataFrame([W21], columns=[f"x{i}" for i in range(21)])
        with self.assertRaises(M.ContractError) as ctx:
            M.pack_weights(df)
        self.assertIn("无法按名对齐", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
