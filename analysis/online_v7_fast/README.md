# online_v7_fast：v7 在线搜索结果的零求解器快速验证包

本目录是 PR #29（`analysis/online_v7/`，完整包）的**精简验证版**，目标是让组员在**不重跑
20 轮求解器（约 3.5 小时）**的前提下，用几秒到几分钟核对 v7 真实运行的结果与可复现性。

- **最优结果**：R = 102.8058786104（第 20 轮，批内第 25 点，move = `icombo`，楼 1；种子 7，
  20 轮 × 50 点 = 1000 次真实评估）。
- **前 11 轮**与 PR #25（v4）逐点一致（`max|dW| = 0`），第 12–20 轮在保留 v4 完整批次的
  基础上换入多尺度候选。
- 相比 v4 同预算（20 轮）R = 103.275295：**−0.4694**；相比 v6 同预算 R = 103.222939：
  **−0.4171**；仍高于 v4@35 的 102.143322（**+0.6626**），未达 R ≤ 102.3。

## 与 PR #29 完整包的差异（本包刻意裁剪）

| 裁剪项 | 大小 | 原因 |
|---|---|---|
| `runs/explore_v6_s7/online_log.jsonl` | 656 KB | 仅用于绘图时画 v6 对照曲线；`plot_v7.py` 缺失时自动跳过，不影响任何验证 |
| `viz_explore_v7/`（4 张 PNG + csv + txt） | ~323 KB | 由 `plot_v7.py` 从日志瞬时重生成，无需随包提交 |

保留的全部是验证所必需的证据：`data/train.csv`、`policy/`（与 PR #29 逐字节相同，哈希见
`run_config.json`）、`explore_v7.py`（入口哈希锁定）、`runs/explore_v7_s7/` 真实运行证据、
`runs/pr25_repro/online_log.jsonl`（冻结一致性对照）、`runs/accelerated15_s7/chain_check.jsonl`
（求解器链路证据）。包体积约 **2.2 MB**（PR #29 约 3 MB）。

## 快速验证（不需要 solver.exe）

见 [QUICK_VERIFY.md](QUICK_VERIFY.md)，四条命令全部不调用求解器：

1. `python explore_v7.py verify-freeze` → 11/11 PASS，`max|dW| = 0`
2. `python explore_v7.py dry-run` → `replay_exact: true`，`solver_calls: 0`
3. `python plot_v7.py` → 重生成 4 张图与逐轮表
4. `python explore_v7.py audit` → `overall_pass: true`，10 项检查全绿
