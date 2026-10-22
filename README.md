# 二次岭回归 baseline

使用第1–8周同环境版的2048条训练数据。只有baseline.py、model.npz、requirements.txt、README.md四个文件。

## 运行

Python 3.11，先安装依赖：

```bash
python -m pip install -r requirements.txt
python baseline.py optimize --output weights.json
```

重新训练只需要训练表，不需要验证文件：

```bash
python baseline.py train --train ./dataset_same8/train.csv
python baseline.py optimize --output weights.json
python baseline.py predict --weights weights.json
```

权重为3×7矩阵，楼宇及七项顺序见数据包说明。每栋楼权重非负、和为1。predict给出模型预测，不是真实控制器成绩。

## 方法

1. 输入21个控制权重，目标是第1–8周平均ramping。
2. sklearn PolynomialFeatures将输入展开为252个一次、平方和交叉项。
3. StandardScaler标准化后，Ridge用Cholesky求解回归系数。模型训练不使用梯度下降。
4. 在2048行内部做固定种子的5折交叉验证，按预测MAE从0.1、1、10、100、1000选择alpha。每折独立拟合标准化，最后用全部2048行重新训练。随包模型的alpha为100。
5. 固定预测模型，从16个起点做投影梯度下降，优化21个输入权重。投影保证每栋楼七项非负、和为1。

建议按train_model → quadratic_form → gradient_search的顺序读代码。回归系数beta与控制权重w是两套参数。训练和模型上的寻优均不调用模拟器，不需要GPU。

训练和最终评估使用同一批八周环境。内部交叉验证衡量未用于该折拟合的权重组合上的预测能力，不代表跨周泛化。模型预测最低的权重可能不如训练表里已有的实测权重，最终以真实评估为准。
