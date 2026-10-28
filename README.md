# SDM374 Project 1 — Data-driven Tuning of Building Energy Controllers

Find 21 controller weights (3 buildings × 7 objectives) that minimize the district's mean ramping
over weeks 1–8. Surrogate models and optimizers are plugins behind fixed interfaces; every
experiment runs from a TOML config through the `sdm` CLI and is recorded under `runs/` so it can be
reproduced.

**Working with a coding agent?** Point it at [`AGENTS.md`](AGENTS.md) and tell it which member it works
for; it will find the role brief in [`docs/roles/`](docs/roles/). Claude Code loads `AGENTS.md`
through `CLAUDE.md` automatically.

## Setup

Python 3.11 and [uv](https://docs.astral.sh/uv/) are required (numpy 1.24.4 has no 3.12+ wheels).
On macOS, the `xgboost` surrogate also needs the OpenMP runtime once: `brew install libomp`.

```bash
uv sync
uv run pytest -q
```

## Workbench (GUI)

```bash
uv sync --group gui
uv run --group gui streamlit run app/main.py
```

Every button runs an `sdm` command as a background job (logs in `runs/.jobs/`), so results are identical
to the terminal and the page can be left or reloaded while a job runs. The server listens on localhost only.

- **总览** — best measured and predicted ramping, model leaderboard, proposals with an extrapolation flag.
- **运行实验** — create a config from a form generated from the plugin's parameters; run train /
  evaluate / optimize; follow, cancel and open background jobs.
- **模型** — registered plugins and their parameters; overlay out-of-fold scatter plots of evaluate runs.
- **实验与数据** — dataset checksum, run browser (metrics, 3×7 weights, config, provenance), reproduce a
  run in the background, and predict R for edited weights with a run's saved model.

## Commands

```bash
uv run sdm list [--json]                               # registered models / optimizers and their params
uv run sdm check-data [--json]                         # verify train.csv against manifest sha256
uv run sdm evaluate -c experiments/baseline-v32.toml   # primary protocol: frozen 2x5 splits, v3.2 score
uv run sdm evaluate -c experiments/baseline.toml       # original protocol: outer 5-fold CV
uv run sdm train    -c experiments/baseline.toml       # fit on all 2048 rows, save model.pkl
uv run sdm optimize -c experiments/baseline.toml -n 50 [--model-run <train-run-id>]
uv run sdm predict  --model-run <run-id> --weights weights.json   # '-' reads stdin
uv run sdm runs [--json]                               # list recorded runs
uv run sdm reproduce <run-id>                          # re-execute and compare outputs
```

## Where to read next

The frozen data splits ([`frozen_data/`](frozen_data/README.md)) and the v3.2 scoring standard
([`evaluation_v3_2/`](evaluation_v3_2/README.md)) are read-only releases shared by the whole team.

| Document | For |
|---|---|
| [`AGENTS.md`](AGENTS.md) | Project summary, repository map, hard rules, definition of done |
| [`docs/interfaces.md`](docs/interfaces.md) | Surrogate / Optimizer contracts, config schema, run files, CLI |
| [`docs/decisions.md`](docs/decisions.md) | Settled decisions, known issues, open questions |
| [`docs/roles/`](docs/roles/) | Per-member mission, owned files, backlog and acceptance criteria |
| `src/sdm/models/_template.py`, `src/sdm/optimizers/_template.py`, `experiments/_template.toml` | Starting points for new plugins and experiments |

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
