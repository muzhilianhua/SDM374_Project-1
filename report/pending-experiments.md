# 报告待补实验：提交策略 h7 的另外两个种子

负责人待定（需要一台 Windows 电脑）。截止 **2026-11-22 12:00** 前交回结果，报告当晚提交。

## 为什么要补

报告表 II 和第 V 节里的红字 `[tbd]` 就是这项实验。提交的策略 h7（`submission/`，PR #32）只在 seed 7 上跑过
（R = 102.806），而 h7 正是看着 v4 seed 7 的日志设计出来的。只报这一个数，评阅人会质疑是挑了种子。v4 在
seed 2026 和 seed 11 上已有结果（108.049、105.440），h7 补上这两个种子后：

- 能报出 h7 三个种子的平均值和范围，也就是正式成绩的合理预期；
- 能和 v4 做同种子的配对比较，判断 h7 的改动在别的种子上是否也有效。

## 机器要求

- Windows，x86-64，CPU 支持 AVX2，**至少 12 个逻辑 CPU**。运行包启动时把 OpenBLAS 固定为 Haswell 内核、12
  线程，然后检查实际情况，对不上就直接退出（`submission/README.md`，"Checking that a machine reproduces the
  batches"）。
- 已安装 [uv](https://docs.astral.sh/uv/)。
- 一份**开发用**的 `solver.exe` 副本，sha256 必须是 `293867b81dbf0c6194f97a57115487b9a39239d0c7dadbd51d749a203a077247`
  （`solver/CHECKSUMS.sha256`）。不要用课程电脑上用于正式评分的那份。
- 每个种子约 2–4 小时：一次 50 组的调用要 5–11 分钟 solver 时间，再加 10–25 秒重新拟合。两个种子可以在同一台机器上
  同时跑，结果不受影响（线程数是固定的，不随负载变化），只是会更慢。

## 步骤（每个种子各做一遍，以 seed 2026 为例）

1. 取得运行包。在仓库里：

   ```powershell
   git fetch origin
   git worktree add D:\sdm374-v7 origin/feat/submission-v7s7
   ```

   然后把 `D:\sdm374-v7\submission\` 整个复制到 `D:\sdm374-seeds\s2026\`。**不要在原来的 `submission\` 里改。**

2. 打开 `D:\sdm374-seeds\s2026\config.json`，只把 `"seed": 7` 改成 `"seed": 2026`，其余一个字都不要动
   （seed 11 同理改成 11）。

3. 每个种子单独准备一份开发用 solver，各自有独立的 20 次额度，例如 `D:\sdm374-dev-solver-2026\solver.exe`。
   这个目录里如果有旧的 `usage.count`，先删掉（只有开发副本可以这样做）。

4. 先检查环境，必须看到 `PASS`，并且 BLAS 显示 Haswell、12 线程：

   ```powershell
   cd D:\sdm374-seeds\s2026
   uv run run.py check --solver D:\sdm374-dev-solver-2026\solver.exe
   ```

5. 正式跑：

   ```powershell
   uv run run.py --solver D:\sdm374-dev-solver-2026\solver.exe --out runs\h7_s2026
   ```

   每次调用会打印一行 `[k/20] batch min … best …`。如果中断了，**原样再运行同一条命令**就会从最后一个完整的调用
   接着跑；已经开始但没有结果的调用算作已用，不会重发。如果运行包报 `rejected`（solver 拒收了一批）并停下，
   先不要加 `--after-rejection`，把输出发给张子木。

## 交回什么

每个种子交回以下内容（压缩后发给张子木，或推到一个新分支 `analysis/h7-seeds` 的
`analysis/online_v7_seeds/h7_s<seed>/` 下）：

- `runs\h7_s<seed>\` 整个目录（`online_log.jsonl` 和 `checkpoints\`）；
- 第 4 步 `check` 的完整输出（记录了 CPU、BLAS、各文件哈希）；
- 第 5 步的控制台输出，至少包括 20 行 `[k/20] …` 和最后的最好值；
- 机器信息：CPU 型号、逻辑 CPU 数、Windows 版本。

如果到 11-22 中午只跑完一个种子，先交这一个，并说明另一个跑到第几次调用。

## 收到结果后（张子木）

1. 从 `online_log.jsonl` 读出每次调用的最好值，填进表 II，算出 h7 三个种子的平均值，以及每个种子减去 v4 同种子的差值。
2. 检查 h7 第 1–11 次调用是否和 `analysis/online_rehearsal/rehearsals/v4_s<seed>/` 里 v4 的批次逐点相同。v4 的日志是在
   另一台 20 核机器上、用 T37 的脚本跑的，如果不相同，配对比较就要注明两边的运行环境不同，必要时再用同一个运行包把
   `config.json` 的 `variant` 改成 `"v4"` 补跑一次 v4（这一步不是必须的）。
3. 用 `report/figures/make_figures.py` 把新种子画进图 1，改写第 V 节"The submitted policy"一段和第 VII 节 Limits (i)，
   删掉所有红字 `\todo{}`。
