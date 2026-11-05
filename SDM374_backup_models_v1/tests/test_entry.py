# -*- coding: utf-8 -*-
"""文件入口测试：真实写 CSV/JSON/NPY/TXT，再走 load_and_predict.read_weights()。

覆盖评审要求的 CSV 布局（无索引 21 列、列序打乱、中文表头 3×7、带建筑索引 3×7、
21 列 + 建筑列）与坏表（必须抛可读的 ContractError），并验证 `.json` / `.npy` / `.txt`
不会因形状误判而静默丢列。所有断言都以“同一个 DataFrame 直接 pack_weights”为基准。

运行：python -m unittest discover -s tests -v   （在包根目录）
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
import modellib as M  # noqa: E402
import load_and_predict as L  # noqa: E402

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


def cell(df):
    """3×7 权重表（行=建筑 1/2/3，列=指标）。"""
    return pd.DataFrame(W21.reshape(3, 7), columns=EN)


class EntryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = M.load_model(PKG / "models" / "gp-ard")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    # ── 工具 ──
    def write_csv(self, df, name="weights.csv", **kw):
        path = self.dir / name
        df.to_csv(path, encoding="utf-8-sig", **kw)
        return path

    def assert_weights(self, path, reference):
        """文件入口读出的权重必须与同一个 DataFrame/数组直接 pack_weights 逐位相同。"""
        got = L.read_weights(path)
        ref = M.pack_weights(reference)
        np.testing.assert_array_equal(got, ref)
        return got

    def assert_prediction(self, path):
        got = L.read_weights(path)
        np.testing.assert_array_equal(got, M.pack_weights(W21.reshape(1, -1)))
        self.assertAlmostEqual(float(self.bundle.predict(got)[0]), EXPECTED, delta=1e-9)

    def assert_contract_error(self, path, *fragments):
        with self.assertRaises(M.ContractError) as ctx:
            L.read_weights(path)
        message = str(ctx.exception)
        for fragment in fragments:
            self.assertIn(fragment, message)
        return message

    # ─────────────────────── 布局 ①：无索引 21 列宽表 ───────────────────────
    def test_wide21_no_index(self):
        df = pd.DataFrame([W21], columns=M.FEATURES)
        self.assert_prediction(self.write_csv(df, index=False))

    def test_wide21_reordered_columns(self):
        perm = np.random.default_rng(7).permutation(M.N_FEATURES)
        df = pd.DataFrame([W21], columns=M.FEATURES)[[M.FEATURES[i] for i in perm]]
        self.assert_prediction(self.write_csv(df, index=False))

    def test_wide21_reordered_columns_with_index_column(self):
        perm = np.random.default_rng(11).permutation(M.N_FEATURES)
        df = pd.DataFrame([W21], columns=M.FEATURES)[[M.FEATURES[i] for i in perm]]
        self.assert_prediction(self.write_csv(df, index=True))      # 多出 1 列 Unnamed: 0

    def test_wide21_plus_building_column(self):
        df = pd.DataFrame([W21], columns=M.FEATURES)
        df.insert(0, "building", ["楼1"])
        self.assert_prediction(self.write_csv(df, index=False))

    def test_wide21_three_rows(self):
        df = pd.DataFrame([W21, W21, W21], columns=M.FEATURES)
        got = self.assert_weights(self.write_csv(df, index=False), df)
        self.assertEqual(got.shape, (3, M.N_FEATURES))

    # ─────────────────────── 布局 ②：3×7 + 建筑列/索引 ───────────────────────
    def test_indexed_3x7_chinese_buildings_scrambled(self):
        df = pd.DataFrame(W21.reshape(3, 7), index=["楼1", "楼2", "楼3"], columns=CN)
        path = self.write_csv(df.iloc[::-1, ::-1])                  # 行列都打乱后落盘
        self.assert_prediction(path)
        # 基准是同一份带标签的 DataFrame，而不是按位置解释
        np.testing.assert_array_equal(L.read_weights(path), M.pack_weights(df))

    def test_indexed_3x7_numeric_buildings(self):
        df = pd.DataFrame(W21.reshape(3, 7), index=[1, 2, 3], columns=EN)
        self.assert_prediction(self.write_csv(df.iloc[[2, 0, 1]]))

    def test_indexed_3x7_english_buildings(self):
        df = pd.DataFrame(W21.reshape(3, 7), index=["building_1", "building_2", "building_3"], columns=EN)
        self.assert_prediction(self.write_csv(df))

    def test_3x7_with_building_column_english(self):
        df = pd.DataFrame(W21.reshape(3, 7), columns=EN).iloc[[2, 0, 1]]   # 物理行乱序
        df.insert(0, "building", ["b3", "b1", "b2"])                      # 楼栋列随行对应
        self.assert_prediction(self.write_csv(df, index=False))

    def test_3x7_with_building_column_chinese(self):
        df = pd.DataFrame(W21.reshape(3, 7), columns=CN).iloc[[2, 0, 1]]
        df.insert(0, "建筑", ["建筑3", "建筑1", "建筑2"])
        self.assert_prediction(self.write_csv(df, index=False))

    # ─────────────────────── 坏表：必须抛可读 ContractError ───────────────────────
    def test_bad_numeric_index_3x7(self):
        df = pd.DataFrame(W21.reshape(3, 7), columns=EN)
        path = self.write_csv(df, index=True)                       # Unnamed: 0 = 0/1/2
        self.assert_contract_error(path, "识别到 3 行 × 8 列", "不是可识别的建筑标签")

    def test_bad_21_columns_missing_one_feature(self):
        df = pd.DataFrame([W21], columns=M.FEATURES).drop(columns=["b3_resilience"])
        df["building"] = ["b1"]
        message = self.assert_contract_error(self.write_csv(df, index=False),
                                             "识别到 1 行 × 21 列", "只识别到 20 个规范特征列")
        self.assertIn("缺 1 个", message)

    def test_bad_3x7_without_any_labels(self):
        df = pd.DataFrame(W21.reshape(3, 7), columns=EN)
        self.assert_contract_error(self.write_csv(df, index=False), "未知行标签")

    def test_bad_shape_2x5(self):
        df = pd.DataFrame(np.arange(10, dtype=float).reshape(2, 5), columns=list("abcde"))
        self.assert_contract_error(self.write_csv(df, index=False), "识别到 2 行 × 5 列",
                                   "列数既不是 7")

    def test_bad_duplicate_building_column(self):
        df = pd.DataFrame(W21.reshape(3, 7), columns=EN)
        df.insert(0, "building", ["b1", "b1", "b2"])
        self.assert_contract_error(self.write_csv(df, index=False), "含重复建筑")

    def test_bad_wide21_two_extra_columns(self):
        df = pd.DataFrame([W21], columns=M.FEATURES)
        df.insert(0, "building", ["b1"])
        df.insert(1, "candidate_id", ["p01969"])
        self.assert_contract_error(self.write_csv(df, index=False), "识别到 1 行 × 23 列",
                                   "还有 2 个附加列")

    # ─────────────────────── 其他文件入口：json / npy / txt ───────────────────────
    def test_json_weights_list(self):
        path = self.dir / "w.json"
        path.write_text(json.dumps({"candidate_id": "p01969", "weights": W21.tolist()}),
                        encoding="utf-8")
        self.assert_prediction(path)

    def test_json_nested_3x7(self):
        path = self.dir / "w37.json"
        path.write_text(json.dumps({"weights": W21.reshape(3, 7).tolist()}), encoding="utf-8")
        self.assert_prediction(path)

    def test_json_without_weight_key(self):
        path = self.dir / "bad.json"
        path.write_text(json.dumps({"candidate_id": "p01969", "foo": [1, 2]}), encoding="utf-8")
        self.assert_contract_error(path, "缺少权重键", "foo")

    def test_npy_vector_and_matrix(self):
        vec = self.dir / "w21.npy"
        np.save(vec, W21)
        self.assert_prediction(vec)

        mat = self.dir / "w37.npy"
        np.save(mat, W21.reshape(3, 7))
        self.assert_prediction(mat)

    def test_npy_wrong_shape(self):
        bad = self.dir / "w20.npy"
        np.save(bad, W21[:20])
        self.assert_contract_error(bad, "长度应为 21")

    def test_txt_comma_separated(self):
        path = self.dir / "w.txt"
        path.write_text(",".join(repr(float(v)) for v in W21), encoding="utf-8")
        self.assert_prediction(path)

    def test_txt_newlines_and_trailing_comma(self):
        path = self.dir / "w_lines.txt"
        path.write_text("\n".join(f"{v}," for v in W21), encoding="utf-8")
        self.assert_prediction(path)

    def test_txt_non_numeric_rejected(self):
        path = self.dir / "bad.txt"
        path.write_text("1,2,oops", encoding="utf-8")
        self.assert_contract_error(path, "不是合法的逗号分隔数字")

    # ─────────────────────── CLI 真实入口 ───────────────────────
    def run_cli(self, *args):
        result = subprocess.run([sys.executable, "load_and_predict.py", *args],
                                cwd=str(PKG), capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_cli_weights_file_wide21(self):
        df = pd.DataFrame([W21], columns=M.FEATURES)
        path = self.write_csv(df, "cli_wide.csv", index=False)
        stdout = self.run_cli("--model", "gp-ard", "--weights-file", str(path), "--json")
        payload = json.loads(stdout[stdout.index("{"):])
        self.assertAlmostEqual(payload["mean"][0], EXPECTED, delta=1e-9)

    def test_cli_weights_file_3x7_chinese(self):
        df = pd.DataFrame(W21.reshape(3, 7), index=["楼1", "楼2", "楼3"], columns=CN)
        path = self.write_csv(df.iloc[::-1, ::-1], "cli_3x7.csv")
        stdout = self.run_cli("--model", "gp-ard", "--weights-file", str(path), "--json")
        payload = json.loads(stdout[stdout.index("{"):])
        self.assertAlmostEqual(payload["mean"][0], EXPECTED, delta=1e-9)


if __name__ == "__main__":
    unittest.main()
