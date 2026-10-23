# SDM374 Project 1 — Data-driven Tuning of Building Energy Controllers

Find 21 controller weights (3 buildings × 7 objectives) that minimize the district's mean ramping
over weeks 1–8. Surrogate models and optimizers are plugins behind fixed interfaces; every
experiment runs from a TOML config and is recorded under `runs/` so it can be reproduced.

## Setup

Python 3.11 and [uv](https://docs.astral.sh/uv/) are required (numpy 1.24.4 has no 3.12+ wheels).

```bash
uv sync
uv run pytest
```

## Commands

```bash
uv run sdm list                                        # registered models / optimizers and their params
uv run sdm check-data                                  # verify train.csv against manifest sha256
uv run sdm train    -c experiments/baseline.toml       # fit on all 2048 rows, save model.pkl
uv run sdm evaluate -c experiments/baseline.toml       # shared offline protocol: outer 5-fold CV
uv run sdm optimize -c experiments/baseline.toml       # propose W* (add -n 50 for a batch)
uv run sdm optimize -c experiments/baseline.toml --model-run <train-run-id>
uv run sdm predict  --model-run <run-id> --weights weights.json
uv run sdm runs                                        # list recorded runs
uv run sdm reproduce <run-id>                          # re-execute and compare outputs
```

`optimize` writes `weights.json` (a 3×7 matrix, same format as the course baseline) and reports
`nearest_train_distance` next to `train_median_nn_distance`. A W* much farther from the
training rows than typical spacing is an extrapolation: the baseline W* predicts 75.9 while the best
measured row is 115.8, so treat such predictions with suspicion.

## Layout

```text
src/sdm/
  interfaces.py     Surrogate (fit / predict -> (mean, std) / optional gradient) and Optimizer (propose)
  registry.py       @register_model / @register_optimizer, auto-discovery, Params validation
  models/           surrogate plugins (ridge2 = course baseline)
  optimizers/       optimizer plugins (pgd = course baseline)
  data.py           dataset loading with checksum verification
  simplex.py        weight layout, constraint check, projection, sampling
  evaluate.py       shared CV protocol: MAE, RMSE, R², Spearman, top-50 recall
  config.py         TOML experiment configs
  runs.py           run folders: config copy, provenance (git commit, data sha256, versions), outputs
  cli.py            the `sdm` command
experiments/        one TOML per experiment
tests/              regression tests pin ridge2 + pgd to baseline.py
dataset_same8/      official data package (unchanged)
baseline.py, model.npz, weights.json, requirements.txt   course baseline (unchanged, reference only)
```

## Adding a model or optimizer

Create one file in `src/sdm/models/` (or `src/sdm/optimizers/`); nothing else needs editing.

```python
from dataclasses import dataclass
import numpy as np
from ..interfaces import Surrogate
from ..registry import register_model


@register_model("my_model")
class MyModel(Surrogate):
    """One-line description shown by `sdm list`."""

    @dataclass(frozen=True)
    class Params:
        seed: int = 0

    def fit(self, X, y):
        ...
        self.metrics = {"some_diagnostic": 0.0}  # optional, saved to metrics.json
        return self

    def predict(self, X):
        return mean, std  # both shape (n,)
```

Then copy `experiments/baseline.toml`, change `[model] name` and its params, and run
`sdm evaluate` to compare against the baseline on the shared protocol. Optimizers implement
`propose(model, X, y, n)` and must return an `(n, 21)` batch on the simplex, best first.
Gradient-based optimizers additionally need `gradient` and `smoothness` from the model.

## Collaboration

- One feature branch per task; merge into `main` through a reviewed pull request.
- Commit messages follow Conventional Commits (`feat:`, `fix:`, `test:`, `docs:` ...).
- `uv run pytest` must pass before merging; never loosen the baseline regression tolerances.
- Quote numbers in discussions and the report by run id, so they can be reproduced.
- `runs/` is git-ignored; share a run by its id and config, or attach its folder.

## Course baseline (original notes)

使用第1–8周同环境版的2048条训练数据。只有baseline.py、model.npz、requirements.txt、README.md四个文件。

### 运行

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

### 方法

1. 输入21个控制权重，目标是第1–8周平均ramping。
2. sklearn PolynomialFeatures将输入展开为252个一次、平方和交叉项。
3. StandardScaler标准化后，Ridge用Cholesky求解回归系数。模型训练不使用梯度下降。
4. 在2048行内部做固定种子的5折交叉验证，按预测MAE从0.1、1、10、100、1000选择alpha。每折独立拟合标准化，最后用全部2048行重新训练。随包模型的alpha为100。
5. 固定预测模型，从16个起点做投影梯度下降，优化21个输入权重。投影保证每栋楼七项非负、和为1。

建议按train_model → quadratic_form → gradient_search的顺序读代码。回归系数beta与控制权重w是两套参数。训练和模型上的寻优均不调用模拟器，不需要GPU。

训练和最终评估使用同一批八周环境。内部交叉验证衡量未用于该折拟合的权重组合上的预测能力，不代表跨周泛化。模型预测最低的权重可能不如训练表里已有的实测权重，最终以真实评估为准。
