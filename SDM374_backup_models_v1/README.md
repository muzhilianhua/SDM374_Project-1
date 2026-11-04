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
pip install numpy scipy pandas joblib "scikit-learn==1.6.1"
python load_and_predict.py --list
python load_and_predict.py --model gp-ard --weights "w1,w2,...,w21"
```

（第三行的 21 个数按下面"特征契约"的顺序给出；也可以 `--weights-file 第一组离线提交.csv`。）

## 特征契约（硬契约，顺序错会直接抛错）

21 维向量，**建筑1 → 建筑2 → 建筑3**，每栋内部 **cost → carbon → peak → discomfort →
degradation → ramping → resilience**：

```text
b1_cost b1_carbon b1_peak b1_discomfort b1_degradation b1_ramping b1_resilience
b2_cost b2_carbon b2_peak b2_discomfort b2_degradation b2_ramping b2_resilience
b3_cost b3_carbon b3_peak b3_discomfort b3_degradation b3_ramping b3_resilience
```

- 也支持 3×7 表格（`csv` 带表头，中英文列名/行名均可），按**列名/行名**映射，不按位置。
- 形状不是 21 维 / 3×7 / (n,21) 一律抛 `ContractError`；7×3（指标×建筑）会被明确拒绝。
- `python load_and_predict.py --self-check` 会验证"21 维向量"与"3×7 中文表头"结果完全一致，
  并验证反转顺序一定会改变预测。

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
├── run_oof_and_score.py             逐折重拟合 → OOF → 外部 v3.2 评分
├── compute_scores.py                用包内已存 OOF 预测复算 v3.2（不重训）
├── self_check.json                  最近一次自检结果
├── models/
│   ├── registry.csv
│   ├── gp-ard/            {model.pkl, config.json, meta.json, metrics.json, CHECKSUMS.json, train_predictions.csv}
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

## 评分复算：依赖的外部路径（评分器**没有**拷进包）

包内 `scores_*.csv` 是原始逐折结果；需要自己复算时，`compute_scores.py` 只读包内 OOF 预测，
并从**仓库根**的下列路径 import / 读取：

| 依赖 | 默认路径（相对包目录） | 用途 |
| --- | --- | --- |
| 评分器目录 | `../SDM374_scoring_v3_2/` | `surrogate_scoring_v3_2.py`：evaluate / policy / context |
| 数据集 | `../dataset_same8/train.csv` | 每折训练标签、候选 id |
| 冻结划分 | `../外推方案/趋势与尾部/frozen/split_manifest.json` | 2×5×3 外层折的成员 |

```bash
python compute_scores.py            # 默认使用上面的相对路径
python compute_scores.py --scoring-dir "..\SDM374_scoring_v3_2"
```

复算已核对：四模型总分与包内 `metrics.json` 完全一致（83.1496 / 83.0766 / 82.5547 / 83.5477）。
`blend_xgb_w20x3` 的 `scores_*.csv` 由 `pred_blend_xgb_w20x3_full.csv` 用同一评分器逐折重算，
与历史 `summary_blend_xgb_w20x3_full.json` 的逐折分数差 < 2e-14。

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

`run_oof_and_score.py` 做逐折重拟合并调用外部评分器（很慢；`blend` 必须提供逐折留出的树预测）：

```bash
python run_oof_and_score.py --model gp-ard
python run_oof_and_score.py --model tail5opt
python run_oof_and_score.py --model hp_wlearn
python run_oof_and_score.py --model blend_xgb_w20x3 --tree-oof "..\SDM374_final_models_v1\evaluation\eval_predictions.csv"
```

## 环境

- `gp-ard` / `tail5opt` / `hp_wlearn`：**推理只用 numpy + scipy**，不依赖 scikit-learn 版本。
- `blend_xgb_w20x3`：混合分支需要 **scikit-learn 1.6.1 + joblib** 反序列化 `xgb_w20x3_full.joblib`。
- 训练脚本需要 scikit-learn 1.6.1（历史拟合环境：Python 3.13.5 / numpy 2.1.3 / scipy 1.15.3 /
  pandas 2.2.3 / joblib 1.4.2 / win32）。

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
