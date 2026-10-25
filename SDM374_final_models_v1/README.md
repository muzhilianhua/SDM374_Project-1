# 最终三个代理模型

三个模型均已使用全部 **2048 条数据**训练，输入为 21 个原始 W 权重，输出为预测 ramping，越低越好。

| 权重文件 | 特点 | 全量配置的外层 v3.2 均分 |
| --- | --- | ---: |
| [histgb_low60_full.joblib](models/histgb_low60_full.joblib) | HistGB，学习率 0.1 | 72.47 |
| [histgb_low80_full.joblib](models/histgb_low80_full.joblib) | HistGB，学习率 0.05 | 72.35 |
| [xgb_w20x3_full.joblib](models/xgb_w20x3_full.joblib) | XGBoost，真实值最低 20% 的训练样本权重为 3 | 72.34 |

low60、low80 只是来源名称，当前均不裁剪数据。它们来自历史前三名策略，但训练方式已变化；历史裁剪版成绩分别为 74.21、74.50、74.72，不能当作这些全量权重的成绩。

环境使用 Python 3.13.5，独立安装 `requirements.txt`。参数、特征顺序、种子与模型哈希见 [MANIFEST.json](MANIFEST.json)。

```bash
python -m pip install -r requirements.txt
python predict.py --input new_test.csv --out predictions.csv
```

输入 CSV 需要 MANIFEST 中的 21 列权重；每栋楼的七个权重非负且和为 1。三个模型分别输出一列预测。只加载可信模型文件；训练集内误差不能当作泛化成绩，预测也不是真实评估结果。
