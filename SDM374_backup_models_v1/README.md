# SDM374 备用代理模型包（四个模型，可直接预测）

本包包含四个在 `dataset_same8` 全部 2048 行上拟合好的代理模型，加载即可预测，**不需要重训**。
目标量统一为 `ramping`（越低越好）。评分采用外部 v3.2 评分器（协议
`sdm374.surrogate.tolerance.v3.2`），OOF 采用冻结划分 `sdm374.same8.splits.2x5x3.frozen.v1`。

| 模型 | 角色 | v3.2 总分 | OOF 低尾 RMSE（≤q01, n=42） |
| --- | --- | --- | --- |
| `gp-ard` | 基线 / σ 校准最可靠 | 83.1496 | 4.7669 |
| `tail5opt` | 低尾专精 | 83.0766 | **4.3382（四模型最低）** |
| `hp_wlearn` | 全局精度对照 | 82.5547 | 5.3066（四模型最差） |
| `blend_xgb_w20x3` | 最高总分混合 / 不可外推参照 | **83.5477** | 5.0654 |

> 四个模型的总分差异都在 10 折折间噪声之内（见 `scores/paired_delta_vs_gp-ard.csv`，
> 配对 p 值 0.329–0.917），不要据此宣称"某模型更强"。

## 三条命令跑起来

```bash
pip install -r requirements.txt          # 含 blend 需要的 xgboost
python load_and_predict.py --list
python load_and_predict.py --model gp-ard --weights "w1,w2,...,w21"
```

（第三行的 21 个数按下面"特征契约"的顺序给出；也可以 `--weights-file 第一组离线提交.csv`。）

## 特征契约（两种路径，**有标签就按标签，无标签才按位置**）

- **无标签路径（21 维向量 / `(n,21)` 数组）**：只能按位置，顺序写死为
  **建筑1 → 建筑2 → 建筑3**，每栋内部 **cost → carbon → peak → discomfort →
  degradation → ramping → resilience**：

```text
b1_cost b1_carbon b1_peak b1_discomfort b1_degradation b1_ramping b1_resilience
b2_cost b2_carbon b2_peak b2_discomfort b2_degradation b2_ramping b2_resilience
b3_cost b3_carbon b3_peak b3_discomfort b3_degradation b3_ramping b3_resilience
```

- **有标签路径（`csv` / DataFrame 的 3×7 表或 21 列宽表）**：**一律按标签重排**，行列顺序完全无关。
  列名支持中文（`电费/碳排/峰值/舒适/损耗/波动/韧性`）与英文（`cost/carbon/peak/discomfort/
  degradation/ramping/resilience` 及 `electricity/peak_load/ramp` 等同义词，大小写与空白归一）；
  行名支持 `第 1 栋/第1栋/楼1/楼栋1/建筑1` 与 `building_1/building1/bldg1/b1/Bldg_1/1` 等写法。
- **CSV 先判布局，再决定是否取索引**（不再无条件把第 0 列当索引，判定只看**列名集合**）：
  - **无索引 21 列宽表**（1 行 × 21 列，列名恰好是 21 个规范特征，顺序任意）→ 保留全部 21 列；
  - **21 列 + 1 个附加列**（如 `df.to_csv()` 写出的 `Unnamed: 0` 索引列或楼栋列）→ 丢弃附加列，仍保留 21 个特征；
  - **3×7 + 建筑列/建筑索引**（第 0 列或任意一列是 `1/2/3`、`b1/b2/b3`、`building_1`、`楼1`、`第 1 栋`）→
    按建筑标签重排；7 个指标列同样按标签重排；
  - 判不清（如 3×7 只有 `0/1/2` 自动索引、21 列里混入楼栋列而少一个特征、列数既非 7 也非 21）
    一律抛 `ContractError`，消息写明**识别到的行×列、列名与期望布局**。
- **严格报错**：未知标签、重复标签、缺指标/缺楼栋、行数≠3、21 列宽表列名与规范不符，
  一律抛 `ContractError` 并写明"得到什么、要求什么"；不再有"按位置静默当数组"的路径。
- 回归测试见 `tests/test_contract.py`（DataFrame 入口：行列逆序/打乱四种都得到同一预测 115.7941456944；
  标签写错/重复/缺失必须报错）与 `tests/test_entry.py`（**真实文件入口**：无索引 21 列 CSV、列序打乱、
  带索引 21 列、21 列+楼栋列、中文表头 3×7、建筑索引 3×7、3×7+楼栋列，以及 6 种坏表；另含
  `.json` / `.npy` / `.txt` 各入口与 CLI `--weights-file`）。
  `python load_and_predict.py --self-check` 也会验证 21 维向量与 3×7 中文表头结果一致。

## 目录结构

```text
SDM374_backup_models_v1/
├── README.md                       本文件
├── 四个模型说明.md                  每个模型的数学形式、改动、指标、场景、低端行为、风险 + A1 排除说明
├── MANIFEST.json                    全包文件清单与 SHA256
├── requirements.txt
├── modellib.py                      推理内核（纯 numpy；混合模型用 joblib 读树）
├── load_and_predict.py              统一加载/预测/--list/--self-check
├── train_four_models.py             四个配置的拟合源代码（核、加权似然、可学习交互权重、混合公式）
├── run_oof_and_score.py             逐折重拟合 → OOF → v3.2 评分（评分器取自仓库根 evaluation_v3_2/）
├── compute_scores.py                用包内已存 OOF 预测复算 v3.2（不重训）
├── verify_hashes.py                 校验 MANIFEST/CHECKSUMS（默认对 HEAD blob，支持 --working-tree）
├── self_check.json                  最近一次自检结果
├── .gitattributes                   文本统一 LF（保证哈希与提交字节一致）
├── （以下两项是仓库根资产，不在包内，脚本直接引用）
│   ├── ../evaluation_v3_2/            仓库根评分器（surrogate_scoring_v3_2.py + EVALUATION_POLICY_V3_2.json）
│   └── ../frozen_data/               仓库根冻结划分归档 SDM374_data_splits_frozen_v1.zip（内含 split_manifest.json）
├── tests/test_contract.py           标签顺序回归测试（DataFrame 入口）
├── tests/test_entry.py              CSV/JSON/NPY/TXT 文件入口与 6 种坏表
├── tests/test_probe_seed.py         --probe-seed 落在真实 log 参数空间
├── models/
│   ├── registry.csv
│   ├── gp-ard/            {model.npz, params.json, config.json, meta.json, metrics.json, CHECKSUMS.json, train_predictions.csv}
│   ├── tail5opt/          同上
│   ├── hp_wlearn/         同上
│   └── blend_xgb_w20x3/   同上 + xgb_w20x3_full.joblib（树模型）
└── scores/
    ├── scores_gp-ard.csv / scores_tail5opt.csv / scores_hp_wlearn.csv / scores_blend_xgb_w20x3.csv
    ├── fold_scores_all.csv           逐折分数宽表（10 折 × 4 模型）
    ├── paired_delta_vs_gp-ard.csv    逐折配对 Δ、均值、sd、胜/负、t、p
    ├── model_scores_oof.csv          总分 + 五个分项 + OOF/低尾/高段 RMSE + in_sample（已标注不可当泛化）
    └── oof_pred_<model>.csv.gz       逐折留出预测（供 compute_scores.py 复算）
```

## 评分复算：引用仓库根 `evaluation_v3_2/` 与 `frozen_data/`（不复制进包）

包内 `scores_*.csv` 是原始逐折结果；需要自己复算时，`compute_scores.py` 只读包内 OOF 预测，
并使用仓库里**已跟踪的真实路径**（包按 `仓库根/SDM374_backup_models_v1` 常规检出，默认值据此解析）：

| 依赖 | 默认路径（相对仓库根） | 用途 |
| --- | --- | --- |
| 评分器目录 | `evaluation_v3_2/` | `surrogate_scoring_v3_2.py` + `EVALUATION_POLICY_V3_2.json`：evaluate / policy / context |
| 数据集 | `dataset_same8/train.csv` | 每折训练标签、候选 id（`--data` 可覆盖） |
| 冻结划分 | `frozen_data/SDM374_data_splits_frozen_v1.zip` | 2×5×3 外层折成员；脚本直接读 zip 内的 `split_manifest.json`，也接受解压后的 `.json`（`--manifest` 可覆盖） |

```bash
python compute_scores.py            # 默认即指向仓库根 evaluation_v3_2/ 与 frozen_data/
python compute_scores.py --scoring-dir ..\evaluation_v3_2 --manifest ..\frozen_data\SDM374_data_splits_frozen_v1.zip
```

两个脚本都会在开始时做**前置存在性检查**：路径缺失立刻报 `[前置检查失败]` 并写明该改指哪里，
不会再跑到中途才崩。

复算已核对：四模型总分与包内 `metrics.json` 完全一致（83.1496 / 83.0766 / 82.5547 / 83.5477，
逐折最大差 ≤ 2.8e-14）。

## 训练源代码与入口命令

`train_four_models.py` 是可移植的拟合代码（分块 ARD Matérn-5/2 + 全区域交互核、
L-BFGS-B 对数空间多起点、加权对数边际似然、可学习交互权重、与树模型的固定权重混合）：

```bash
python train_four_models.py --list
python train_four_models.py --model gp-ard
python train_four_models.py --model tail5opt
python train_four_models.py --model hp_wlearn
python train_four_models.py --model blend_xgb_w20x3
python train_four_models.py --model all
```

`run_oof_and_score.py` 做逐折重拟合并调用**仓库根 `evaluation_v3_2/`** 评分器（很慢；`blend` 必须提供逐折留出的树预测）：

```bash
python run_oof_and_score.py --model gp-ard
python run_oof_and_score.py --model tail5opt
python run_oof_and_score.py --model hp_wlearn
python run_oof_and_score.py --model blend_xgb_w20x3 --tree-oof "..\SDM374_final_models_v1\evaluation\eval_predictions.csv"
```

`--seed`（默认 2026）会一路传到多起点优化器与 `GaussianProcessRegressor.random_state`，
并写入输出的 `seed_used`。`python train_four_models.py --probe-seed 2027` 可直接打印该种子
实际生成的多起点向量（不训练）：起点与边界都直接取自 `build_kernel(...)` 返回核的
`theta` / `bounds`（即 `make_optimizer` 真正消费的 **log 空间**），输出同时给出
`starts_log` 与 `starts_raw`（长度尺度 ×7 + 噪声），并自检还原值全部落在真实边界内
（长度尺度 ∈ [0.03, 6]、噪声 ∈ [0.001, 0.3]，以 `kernel.bounds` 为准）。
回归测试见 `tests/test_probe_seed.py`（seed 2026/2027/2028 的每个起点都合法，
且 2027 的第二起点与 2026 不同）。

## 环境

- `gp-ard` / `tail5opt` / `hp_wlearn`：**推理只用 numpy + scipy**，不依赖 scikit-learn 版本。
- `blend_xgb_w20x3`：混合分支需要 **scikit-learn 1.6.1 + joblib + xgboost**（joblib 读树模型）。
- 训练脚本需要 scikit-learn 1.6.1（历史拟合环境：Python 3.13.5 / numpy 2.1.3 / scipy 1.15.3 /
  pandas 2.2.3 / joblib 1.4.2 / win32）。
- OOF 复算需要仓库根 `evaluation_v3_2/` 与 `frozen_data/SDM374_data_splits_frozen_v1.zip`
  （二者均已在 origin/main 上跟踪，本包只引用、不复制）。

### 兼容性矩阵（本包**不要求**升级项目环境）

| numpy | 状态 | 验证方式 |
| --- | --- | --- |
| **1.24.4（仓库锁定）** | ✅ 通过 | 独立 venv（Python 3.11 + numpy 1.24.4 / scipy 1.10.1 / pandas 1.5.3）下跑 `--list`、`--self-check`、四模型预测与 `compute_scores.py`，逐折分数差 ≤ 2.8e-14 |
| **2.x（2.1.3 / 2.2.x）** | ✅ 通过 | 本机主环境（Python 3.13.5 / numpy 2.1.3）跑同一套命令，结果一致 |

载荷是 `model.npz`（`np.savez_compressed`，`allow_pickle=False`）+ `params.json`，**不再 pickle
numpy 数组**，因此不受 `numpy._core` / `numpy.core` 模块路径差异影响。

## 自检

```bash
python load_and_predict.py --self-check
```

最近一次结果（`self_check.json`）：

- A 加载后 vs 保存的全量 in-sample 预测：最大绝对差 **4.4e-12**
- B 21 维向量 vs 3×7 中文表头：**0.0**
- C 加载 vs p01969 的 OOF 留出预测：最大 3.31（全量 refit 与折内模型的正常差异）
- D 加载 vs 历史 gp 提交预测 121.488948：gp-ard 0.0015，其余 1.19–2.18
- E 反转特征顺序：四模型预测全部改变（无静默重排）
