# 模型尝试与结果

这里汇总已有成绩，不重新训练，也不将不同评分标准混排。

## 按 v3.2 复核的 11 组

使用同一冻结划分，取十个外层测试折的平均成绩。原始数据与分项分数见 [总表 CSV](v3_2_scores.csv)，便于阅读的表格见 [Excel](eleven_model_comparison.xlsx)，预测效果见 [散点图](prediction_scatter.png)。

| Model | v3.2 | Global MAE | Low MAE | Top-1 regret |
| --- | ---: | ---: | ---: | ---: |
| low90_w20x3 + XGBoost | 74.72 | 5.21 | 6.40 | 1.59 |
| low80 + HistGB | 74.50 | 5.48 | 6.13 | 1.48 |
| low60 + HistGB | 74.21 | 7.35 | 6.25 | 0.41 |
| low90_w20x3 + HistGB | 73.73 | 5.09 | 6.41 | 1.15 |
| tiered + HistGB | 73.26 | 4.90 | 6.85 | 1.59 |
| Original XGBoost | 73.04 | 4.81 | 6.97 | 2.24 |
| Original HistGB | 72.65 | 4.85 | 7.30 | 2.36 |
| Original weighted XGBoost | 72.18 | 4.97 | 6.44 | 1.49 |
| Extra Trees | 68.44 | 5.44 | 10.14 | 4.51 |
| RBF kernel ridge | 67.01 | 6.63 | 10.59 | 5.04 |
| Quadratic ridge | 66.19 | 6.86 | 12.23 | 4.51 |


low60、low80 表示只用训练集中真实值最低的 60%、80%；low90_w20x3 表示保留最低 90%，并给最低 20% 三倍权重。tiered 是最低 20% 五倍权重、接下来 20% 两倍权重。测试集始终完整。

原决赛圈的六组是二次岭、核岭、Extra Trees、HistGB、原始 XGBoost、低值加权 XGBoost。上述表已全部包含。

## 组员尝试的 21 组及原始误差

以下是组员模型索引中十折指标的均值，共七种策略 × 三种模型。我们只对其中截图前五组复核了 v3.2 分数；其余结果直接来自组员记录，**不是新增评估**。完整精度见 [原始指标 CSV](peer_21_models_raw_metrics.csv)。

| Strategy | Model | MAE | RMSE | R2 |
| --- | --- | ---: | ---: | ---: |
| Lowest 90%; low20 x3 | gbdt | 5.276 | 8.512 | 0.707 |
| Lowest 90%; low20 x3 | histgb | 5.090 | 8.497 | 0.709 |
| Lowest 90%; low20 x3 | xgboost | 5.215 | 8.462 | 0.711 |
| Low20 x5; next20 x2 | gbdt | 5.000 | 7.571 | 0.767 |
| Low20 x5; next20 x2 | histgb | 4.905 | 7.353 | 0.780 |
| Low20 x5; next20 x2 | xgboost | 4.889 | 7.419 | 0.776 |
| Lowest 40% x5 | gbdt | 5.032 | 7.549 | 0.768 |
| Lowest 40% x5 | histgb | 4.980 | 7.401 | 0.777 |
| Lowest 40% x5 | xgboost | 4.844 | 7.345 | 0.781 |
| First 20% by ID x5 | gbdt | 5.018 | 7.611 | 0.765 |
| First 20% by ID x5 | histgb | 4.946 | 7.474 | 0.773 |
| First 20% by ID x5 | xgboost | 4.841 | 7.380 | 0.779 |
| Lowest 60% | gbdt | 6.683 | 10.221 | 0.578 |
| Lowest 60% | histgb | 7.350 | 11.190 | 0.495 |
| Lowest 60% | xgboost | 6.722 | 10.471 | 0.557 |
| Lowest 80% | gbdt | 5.516 | 9.089 | 0.667 |
| Lowest 80% | histgb | 5.483 | 9.017 | 0.672 |
| Lowest 80% | xgboost | 5.543 | 9.127 | 0.664 |
| Full data | gbdt | 4.869 | 7.410 | 0.777 |
| Full data | histgb | 4.793 | 7.281 | 0.785 |
| Full data | xgboost | 4.834 | 7.431 | 0.776 |


其中按 ID 加权与按 ramping 加权是不同策略，不应混称为低值样本加权。

## 组员报告中另外提到的尝试

组员报告将范围扩展到 30 组：还包括两种非对称加权策略，以及 LightGBM 的三个分位数回归版本。报告中非对称加权最佳为 GBDT，72.53 分；分位数 α=0.1、0.2、0.3 分别为 71.19、70.67、71.50 分。这些分数未由我们独立复核，现有材料未给齐其余候选的逐项成绩，因此不补造完整排名。

早期也尝试过支持向量回归，随后淘汰；当前保留材料没有可核对的 v3.2 成绩。旧版本评分不与本页的 v3.2 分数混用。

## 最终三个全量配置

| 模型 | v3.2 | 全局 MAE | 低值 MAE | 第一推荐差距 |
| --- | ---: | ---: | ---: | ---: |
| histgb_low60_full | 72.47 | 4.82 | 6.67 | 1.59 |
| histgb_low80_full | 72.35 | 4.79 | 6.67 | 1.88 |
| xgb_w20x3_full | 72.34 | 4.96 | 6.75 | 1.76 |

以上由包内另外训练的外层模型测试预测复算得到，不是把全量模型放回训练集测试。全量配置取消了裁剪，不能继承原裁剪版的 74 分成绩。三者均分接近，不能仅凭微小分差断言优劣。

来源编号和口径见 [SOURCES.json](SOURCES.json)，历史逐折结果见 [历史折成绩](historical_fold_scores.csv)，全量配置见 [全量配置折成绩](final_config_fold_scores.csv)。实际新 W 的效果仍未经真实评估验证。
