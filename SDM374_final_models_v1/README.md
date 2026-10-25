# SDM374 最终代理模型包 v1（3 个独立全量模型，可直接使用）

本包是从 7 套训练策略中最终选出的 **3 个代理模型**，每个都在**全量 2048 条记录**上训练完成，
无需再训练。目标变量为 `ramping`，输入为 21 个权重特征列。

| 模型 | 来源策略 | 估计器 | 样本权重 |
|---|---|---|---|
| `xgb_w20x3_full` | low90_w20x3 + xgboost | XGBRegressor(depth=3, lr=0.1, n=600, sub=0.9) | 最低 20% ramping 样本 3× |
| `histgb_low80_full` | low80 + histgb | HistGB(leaves=15, lr=0.05, iter=600, l2=0) | 无 |
| `histgb_low60_full` | low60 + histgb | HistGB(leaves=15, lr=0.1, iter=600, l2=0) | 无 |

- 训练数据指纹：`dataset_sha256 = 877b5f8ee265245dd693148a5d7b11f28f48b426e43ac3ac1be8b7a934249023`
- 超参数 = 各自来源实验 10 个外层折中选的众数参数，**未重新调参**；随机种子与权重规则完整记录在 `MANIFEST.json`
- **选型建议见 [`模型选择指南.md`](模型选择指南.md)**

## 1. 环境

```bash
pip install -r requirements.txt
```

训练环境为 **Python 3.13.5 / scikit-learn 1.6.1 / xgboost 3.4.1 / joblib 1.4.2**。
`joblib` 反序列化对 scikit-learn 版本敏感，**请勿混用其他大版本**；版本不一致时即使能加载也要先做一次预测抽检。

## 2. 快速开始

```bash
# 三个模型全部输出（每个一列）
python predict.py --input new_test.csv --out preds.csv

# 只跑一个模型
python predict.py --input new_test.csv --out preds.csv --model histgb_low80_full

# 指定标识列（否则自动生成 row_id）
python predict.py --input new_test.csv --out preds.csv --id-column candidate_id
```

如果输入 CSV 带有真实 `ramping` 列，会自动额外输出 `preds_metrics.csv`（RMSE / MAE / R²）。

## 3. 输入 / 输出契约

**输入**：CSV，至少包含 21 个特征列，列顺序不限、按列名读取：

```
b1_cost b1_carbon b1_peak b1_discomfort b1_degradation b1_ramping b1_resilience
b2_cost b2_carbon b2_peak b2_discomfort b2_degradation b2_ramping b2_resilience
b3_cost b3_carbon b3_peak b3_discomfort b3_degradation b3_ramping b3_resilience
```

**无任何预处理**：模型在原始特征上以 `float64` 训练，没有标准化、编码或缺失值填充。
HistGB 原生支持 NaN；XGBoost 模型对 NaN 会报错。

**输出**：`<out>.csv`，首列为标识列，其后每个模型一列预测。

## 4. 目录结构

```
SDM374_final_models_v1/
├── predict.py                  # 预测入口
├── MANIFEST.json               # 特征/目标/参数/种子/权重规则/指标/模型 sha256
├── requirements.txt
├── README.md
├── 模型选择指南.md              # 三个模型的性能优劣对比与场景选型
├── models/
│   ├── xgb_w20x3_full.joblib
│   ├── histgb_low80_full.joblib
│   └── histgb_low60_full.joblib
├── evaluation/
│   ├── eval_folds.csv          # 逐折诚实评估（最终配置在 80% 数据上重训、外层测试集评分）
│   ├── eval_segments.csv       # 分段指标
│   ├── eval_predictions.csv    # 全部外层测试预测（bit-exact）
│   ├── comparison_segments.csv # 与基准/裁剪谱系的对照
│   ├── final_insample_predictions.csv  # 训练集内预测（仅供冒烟核对，偏乐观）
│   └── run_manifest.json       # 完整训练记录（参数/种子/哈希/环境）
└── checksums_sha256.txt        # 全包文件校验和
```

## 5. 注意事项

1. `.joblib` 是 pickle 格式，只加载可信来源文件（本包）。
2. **这三个模型在全量 2048 条记录上训练**。任何与这 2048 条重叠的测试数据都会产生虚高指标
   （in-sample 泄漏）；`evaluation/final_insample_predictions.csv` 里的训练集内 RMSE（0.6~2.3）
   就是这种虚高的量级，**不能**当作泛化性能。泛化性能请以 `evaluation/eval_folds.csv` 的
   外层留出评估为准（RMSE≈7.3~7.5）。
3. 三个模型各自独立，可单独使用，也可做简单平均集成（见模型选择指南）。
4. 模型只预测 `ramping`；其他指标（成本/碳排等）需另建模型。
5. 完整性校验：

```bash
# PowerShell 逐个比对（在包根目录执行）
Get-Content checksums_sha256.txt | ForEach-Object {
  $h,$p = $_ -split '  ',2
  if ((Get-FileHash $p -Algorithm SHA256).Hash.ToLower() -ne $h) { Write-Warning "MISMATCH $p" }
}
```
