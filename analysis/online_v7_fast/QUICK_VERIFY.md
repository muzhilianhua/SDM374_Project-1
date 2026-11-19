# v7 快速验证指南（零求解器，几秒到几分钟）

本包让你在**不调用 `solver.exe`、不重跑 20 轮（约 3.5 小时）**的情况下，核对 v7 真实运行的
结果与可复现性。所有命令从本目录运行，路径相对于本目录（`HERE`）。

## 0. 准备（一次性）

需要 Python 3.11（课程电脑的 Python 3.14 装不了 numpy 1.24.4）。在本目录建虚拟环境并安装
锁定依赖：

```powershell
py -3.11 -m venv .venv
.venv\Scripts\pip install -r requirements-lock.txt
```

`requirements-lock.txt` 固定了 numpy 1.24.4 / scipy 1.10.1 / scikit-learn 1.2.2 / matplotlib 3.7.5
等版本，与产生本结果的环境一致（`audit` 第 8 项会校验）。**不需要 solver.exe**——下面四条
命令都不碰求解器。

## 1. 冻结一致性（必做，不调用 solver）

```powershell
.venv\Scripts\python.exe explore_v7.py verify-freeze
```

用参考运行（PR #25 / v4）自己的历史重放第 1–11 轮，按多重集合逐点核对 v7 冻结阶段是否与
参考完全一致。期望输出（11 轮全 PASS，`max|dW| = 0`）：

```
  call  1: 匹配 50/50  max|dW|=0.00e+00  PASS
  ...
  call 11: 匹配 50/50  max|dW|=0.00e+00  PASS
冻结校验：PASS（未调用 solver）
```

通过说明：在你的电脑上，同一份策略代码 + 同一份训练集，前 11 轮发出的批次与产生本结果的
运行逐点相同。本命令会把结果写入 `runs/explore_v7_s7/freeze_check.json`。

## 2. 第 12 轮提案可重放（必做，不调用 solver）

```powershell
.venv\Scripts\python.exe explore_v7.py dry-run
```

加载前 11 轮参考历史后，连续两次生成第 12 轮提案并比较，确认策略在不调用求解器时是确定性
的。期望输出关键字段：

```
"replay_exact": true, "solver_calls": 0
...
dry-run: PASS（未调用 solver）
```

## 3. 重生成图表（可选，不调用 solver）

```powershell
.venv\Scripts\python.exe plot_v7.py
```

仅从 `runs/explore_v7_s7/online_log.jsonl` 读真实日志，重生成 `viz_explore_v7/` 下的 4 张图
（R 曲线、学习曲线、步长/通道演化、结构图）与 `per_call.csv`、`summary.txt`。期望末行：

```
最终最优 R：102.8058786104（第 20 轮，批内第 25 点）
```

> 本包未随附 v6 对照日志（已裁剪），因此 R 曲线图里不会画 v6 虚线；其余内容完整。若需要
> v6 对照曲线，从 PR #29 的 `analysis/online_v7/runs/explore_v6_s7/online_log.jsonl` 拷到
> 本目录同路径下再跑一次 `plot_v7.py` 即可，脚本会自动补上。

## 4. 审计报告（必做，不调用 solver）

```powershell
.venv\Scripts\python.exe explore_v7.py audit
```

重建 `runs/explore_v7_s7/audit_report.json`，校验 10 项。期望：

```
{"complete": true, "overall_pass": true, "report": "..."}
```

10 项检查应全部为 `pass: true`：训练集来源、策略无外部答案、运行期文件访问、提案可重放、
无重复点、结果归属顺序、种子配置固定、环境可重建、冻结与求解器链路、预算。其中第 9 项的
求解器链路证据复用同一 sha256 副本的既有记录（`runs/accelerated15_s7/chain_check.jsonl`），
不在本次额度内重复对拍。

## 固定条件（全部写入 `runs/explore_v7_s7/run_config.json`，运行时锁定）

| 项目 | 固定值 |
|---|---|
| 随机种子 | `7`（`rng_uses_clock = false`，随机性只由 `(seed, call_no)` 决定） |
| 每轮点数 | `50` |
| 轮次 | `20`（求解器额度上限 20 次） |
| 冻结轮次 | `1–11`（与 PR #25 逐点一致，`max|dW| = 0`） |
| 训练集 | `data/train.csv`，2048×23，sha256 `877b5f8ee265245dd693148a5d7b11f28f48b426e43ac3ac1be8b7a934249023` |
| 求解器 | 课程 `solver.exe`，sha256 `293867b81dbf0c6194f97a57115487b9a39239d0c7dadbd51d749a203a077247`（验证不需要它） |
| Python | `3.11.18` |
| 运行依赖 | numpy 1.24.4 / scipy 1.10.1 / scikit-learn 1.2.2 / joblib 1.6.0 / threadpoolctl 3.7.0 |
| 策略文件 sha256 | 见 `runs/explore_v7_s7/run_config.json` 的 `policy_sha256`（与 PR #29 逐字节相同） |
| 运行入口 sha256 | `c0229c16f9421c4912894ee47a44c14f035195dd5e1bd5f346af8838632d4ae7`（`explore_v7.py`） |
| 线程环境 | 未强制单线程，与 PR #25 参考运行默认线程环境一致 |

## 数据边界（防泄漏）

第 1 次调用前策略只看 `data/train.csv`；之后只再用本次运行自己的调用结果。`audit` 第 2、3
项会静态扫描 `policy/` 代码与运行期文件访问，确认没有读取 `runs/`、`reference/` 或任何
`.csv`/`.jsonl` 外部答案。`runs/pr25_repro/online_log.jsonl` 只在 `verify-freeze` 对照时读取，
从不交给策略。

## 文件清单

```
analysis/online_v7_fast/
├── README.md              本文件：包说明与裁剪边界
├── QUICK_VERIFY.md        本文件：四条零求解器验证命令与期望输出
├── requirements-lock.txt  锁定依赖（含绘图）
├── requirements.txt       运行期依赖（哈希写入 run_config.json）
├── online_run.py          共享运行层：训练集校验、状态、规范化、审计钩子
├── explore_v7.py          v7 入口（verify-freeze / dry-run / run / audit），哈希锁定
├── plot_v7.py             绘图，只读已存档日志
├── policy/                6 个策略文件（与 PR #29 逐字节相同）
├── data/                  train.csv + manifest.json
└── runs/
    ├── explore_v7_s7/     本次运行证据：run_config / online_log / freeze_check / audit / summary
    ├── pr25_repro/        PR #25 参考日志（冻结一致性对照，不参与提案）
    └── accelerated15_s7/  求解器链路证据 chain_check.jsonl
```

`runs/` 在仓库根 `.gitignore` 中被忽略；本包的证据文件是用 `git add -f` 显式提交的真实运行
快照，不是可再生临时目录。
