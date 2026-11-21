# 报告补充实验 2：分块下降与名额分配的有效性

执行：张子木，在昨天跑 h7 种子的那台 Windows 电脑上（i5-14600KF，20 逻辑 CPU）。尽量今晚开始，**2026-11-22 17:00 前**
把已完成的运行推回来，报告当晚提交。

## 为什么要补

改写后的报告重点论证两个创新点，现有证据都不够：

1. **加性分块下降**（每个候选只改一栋建筑，下一次调用把各建筑的改进组合起来）。目前没有"不分块"的对照，唯一的
   全局代理优化基线（`analysis/online_validation_105`，1 个种子 105.99）和 v4 大致相当。新增消融变体 **v4j**：除了
   分块结构，其余都和 v4 相同（同一个代理模型、同样的移动类型和步长、同样的命中率名额分配），每个候选同时改动三栋
   建筑，没有组合点。代码在 `analysis/online_rehearsal/policy_v4j.py`，已用合成评估器测通，20 次调用、每次都发满 50 个点。
2. **证据驱动的名额分配**（v4 相对 v2：平台过滤、按命中率分配名额、后期扫描）。现在只有 3 个种子（v4 赢 2 个，平均
   低 0.93），T37 估计需要 6–8 个种子。v2 和 v4 各补 3 个种子（种子 1、2、3），凑成每个变体 6 个种子。

## 机器要求

- 仓库检出 `docs/report` 分支的最新提交（含 `policy_v4j.py` 和 `run_extra.sh`），`uv sync` 过。
- `solver/solver.exe` 已就位，sha256 为 `293867b8…7247`（`solver/CHECKSUMS.sha256`），这是 T37 预演框架的要求
  （`analysis/lowr_search/pool.py` 会检查）。框架在 `analysis/online_rehearsal/workers/` 下建立 solver 的硬链接副本，
  只在这些副本里重置 `usage.count`，**不动 `solver/` 里的那份**。
- **从 Git Bash 启动**，不要从 PowerShell 工具宿主启动（见 `analysis/online_v7_seeds/README.md` 里的控制台问题；
  预演框架本身用 `CREATE_NO_WINDOW` 启动 solver，但仍建议用 Bash）。

## 运行

在仓库根目录：

```bash
git fetch origin && git switch docs/report && git pull
uv sync
bash analysis/online_rehearsal/run_extra.sh
```

- 同时跑 3 个，每个 6 个 solver 进程（T37 的 v5/v6 就是这样跑的：三次运行约 1.6 小时）。前 9 个约 5 小时，
  全部 12 个约 6.5 小时。顺序是 `v4j_s2026, v2_s1, v4_s1, v4j_s7, v2_s2, v4_s2, v4j_s11, v2_s3, v4_s3`，之后是可选的
  `v4j_s1, v4j_s2, v4j_s3`。
- 进度：`tail -f analysis/online_rehearsal/rehearsals/<name>/console.log`，每次调用一行 `call k: min … best …`，结束时一行
  `final best …`。
- 中断后原样重跑同一条命令：已完成的运行会跳过，未完成的从最后一个完整调用继续。

## 交回什么

到 17:00（或全部跑完）时，把完成的运行推到新分支 `analysis/extra-seeds`：

```bash
git switch -c analysis/extra-seeds
git add analysis/online_rehearsal/rehearsals/{v2_s1,v2_s2,v2_s3,v4_s1,v4_s2,v4_s3,v4j_s2026,v4j_s7,v4j_s11}
git commit -m "docs: v2/v4 seeds 1-3 and v4j block ablation for the report"
git push -u origin analysis/extra-seeds
```

只加已经出现 `final best` 的运行目录（`config.json`、`console.log`、`online_log.jsonl`、`pool_log.jsonl`）；未跑完的不要
推。另附机器信息（CPU、逻辑 CPU 数）以及运行期间的任何异常输出。

## 收到结果后（张子木）

1. 表：v1、v2、v4、v4j 各自的种子最佳值、平均值和标准差；v4 对 v2、v4 对 v4j 的均值差，以及置换检验或 Mann–Whitney 检验的
   p 值（种子不配对：不同策略在同一个种子下的随机数流不同）。
2. 运行内部的证据：组合点与单建筑移动的命中率、组合点的加性预测误差；v2 与 v4 后期重复值的比例、各类移动的名额与命中率。
3. 结论如实写：如果 v4j 不比 v4 差，或 v4 相对 v2 不显著，就照实写。
