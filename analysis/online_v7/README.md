# online_v7：保留 v4 完整批次 + 多尺度候选的真实在线搜索

本目录是一次完整、可复现的 20 轮 × 50 点真实求解器在线搜索（总计 1000 次真实评估，最优
R = 102.8058786104）。策略在 PR#25（v4）的基础上改进，用于回答“能否用更少轮次达到更低 R”。

结果与对比见 [REPORT.md](REPORT.md)，策略原理见同文件第 2 节。

## 目录内容

```
analysis/online_v7/
├── README.md                  本文件：固定条件与复现步骤
├── REPORT.md                  策略原理、结果展示、与 v4/v6 的对比报告
├── requirements-lock.txt      完整依赖锁（含绘图依赖）
├── requirements.txt           运行期依赖（与探索包一致）
├── online_run.py              共享运行层：训练集/求解器校验、状态、规范化、审计钩子
├── explore_v7.py              v7 运行入口（verify-freeze / dry-run / run / audit）
├── plot_v7.py                 绘图脚本，仅读取已存档日志
├── policy/
│   ├── policy.py              v1 基础策略（共享工具、特征、ZERO/MINOR 定义）
│   ├── policy_v4.py           PR#25 策略（冻结阶段使用）
│   ├── policy_v5.py           15 轮加速实验策略（仅保留作历史对照，未参与本次运行）
│   ├── policy_v6.py           多尺度基础策略（v7 的父类）
│   ├── policy_v7.py           本次策略：v4 完整批次 + 替换零产出名额
│   └── surrogate.py           GP / Ridge / 特征接口
├── data/
│   ├── train.csv              冻结训练集（2048×23，与仓库其他副本逐字节一致）
│   └── manifest.json          数据集元信息与 sha256
├── runs/
│   ├── explore_v7_s7/         本次运行证据：配置、逐点日志、冻结校验、审计、汇总
│   ├── pr25_repro/            PR#25 参考运行日志（仅作冻结核对与对照，不参与提案）
│   ├── explore_v6_s7/         v6 对照日志（仅用于绘图对照，不参与提案）
│   └── accelerated15_s7/      求解器链路证据（chain_check.jsonl）
└── viz_explore_v7/            四张图与逐轮表、文字汇总
```

`runs/` 与 `analysis/**/results/` 在仓库根 `.gitignore` 中被忽略；本目录的 `runs/` 证据文件是用
`git add -f` 显式提交的，属于已完成的真实运行快照，不是可再生的临时目录。

## 固定条件

| 项目 | 固定值 |
|---|---|
| 随机种子 | `7`（所有随机性只由 `(seed, call_no)` 决定，`rng_uses_clock=false`） |
| 每轮提交点数 | `50` |
| 轮次 | `20`（求解器额度上限 20 次） |
| 冻结轮次 | `1–11`（与 PR#25 逐点一致，`max|dW| = 0`） |
| 训练集 | `data/train.csv`，2048 行 × 23 列，sha256 `877b5f8ee265245dd693148a5d7b11f28f48b426e43ac3ac1be8b7a934249023` |
| 求解器 | 课程 `solver.exe`，sha256 `293867b81dbf0c6194f97a57115487b9a39239d0c7dadbd51d749a203a077247` |
| Python | `3.11.18` |
| 运行依赖 | `numpy 1.24.4` / `scipy 1.10.1` / `scikit-learn 1.2.2` / `joblib 1.6.0` / `threadpoolctl 3.7.0` |
| 策略文件 sha256 | 见 `runs/explore_v7_s7/run_config.json` 的 `policy_sha256` |
| 运行入口 sha256 | `c0229c16f9421c4912894ee47a44c14f035195dd5e1bd5f346af8838632d4ae7` |
| 线程环境 | 未强制单线程，与 PR#25 参考运行的默认线程环境一致（`MKL/OMP/OPENBLAS` 均未设置） |
| 平台 | `Windows-10-10.0.26100-SP0`，12 逻辑核；单进程顺序调用求解器 |

以上字段全部写入 `runs/explore_v7_s7/run_config.json`，运行时由 `explore_v7.py` 自动生成并锁定，
不会因环境变化而静默改变；`audit` 会校验实际环境与记录一致。

## 复现步骤

1. 克隆仓库并切到分支 `online102_8`，安装依赖：

```bash
python -m venv .venv
.venv/Scripts/pip install -r analysis/online_v7/requirements-lock.txt
```

2. 从课程 `solver.zip` 取得评估器，放到 `analysis/online_v7/solver/solver.exe`，并校验：

```bash
python -c "import hashlib;print(hashlib.sha256(open('analysis/online_v7/solver/solver.exe','rb').read()).hexdigest())"
# 期望 293867b81dbf0c6194f97a57115487b9a39239d0c7dadbd51d749a203a077247
```

`explore_v7.py` 会自动把主求解器复制成 `explore_v7_solver/` 副本再调用（避免消耗主目录的
额度计数器），该副本与 `usage.count` 均已在 `.gitignore` 中屏蔽。

3. 校验冻结阶段与 PR#25 完全一致（不调用求解器）：

```bash
python analysis/online_v7/explore_v7.py verify-freeze
```

期望：11 轮 `multiset_identical=true`、`max_abs_diff=0.0`。

4. 校验第 12 轮提案的重放一致性（不调用求解器）：

```bash
python analysis/online_v7/explore_v7.py dry-run
```

期望：`replay_exact: true`，`dry-run: PASS（未调用 solver）`。

5. 真实运行 20 轮（会消耗 20 次课程求解器额度）：

```bash
python analysis/online_v7/explore_v7.py run
```

日志 append-only 写入 `analysis/online_v7/runs/explore_v7_s7/online_log.jsonl`，中断后再次执行
同一命令会从断点续跑（会先校验调用编号连续、无未完成的半次调用）。

6. 生成/覆盖审计报告（读取环境与日志，不调用求解器）：

```bash
python analysis/online_v7/explore_v7.py audit
```

7. 重新生成图表与逐轮表：

```bash
python analysis/online_v7/plot_v7.py
```

## 说明

- 求解器为 182 MB 的可执行文件，超过 GitHub 单文件限制，故不入库；请从课程 `solver.zip` 获取。
- 仓库内的 `runs/pr25_repro`、`runs/explore_v6_s7`、`runs/accelerated15_s7` 只作为冻结核对、
  审计输入和绘图对照；v7 的提案不读取其中任何 R 值或最优点（`history_used_for_policy=false`，
  审计项 `no_external_answer_reference` 会检查策略代码与运行时文件访问）。
- `policy_v5.py` 属于 15 轮加速实验，本次未参与运行，保留仅为记录该阶段结论。
