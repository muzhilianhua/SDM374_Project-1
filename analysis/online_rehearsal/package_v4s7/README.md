# SDM374 在线阶段验证包：策略 v4，种子 7

用途：在组员和助教的电脑上检验两件事——

1. **同一份代码在你的电脑上是否发出和参考运行完全相同的批次**（`verify`，不需要 solver，约 10 分钟）；
2. （可选）**完整跑一遍 20 次 × 50 组的在线流程**，逐次和参考结果比对（`run`，需要 solver，约 2–4 小时，会用掉该 solver 副本 20 次额度）。

参考运行：2026-11-16 在张子木的电脑（Windows 11，Intel i5-14600KF，20 个逻辑核，Python 3.11.16）上用本地 `solver.exe`
跑完 20 次调用，最低 **R = 103.27529487758876**（`reference/v4_s7_online_log.jsonl`，最后一次调用）。

> **这是验证包，不是正式提交。** 种子 7 是在本地评估器上比较了 3 个种子后选出的；按本地成绩选种子是否允许，
> 还没有得到助教确认（`docs/decisions.md` 决策 27 与开放问题）。

## 一、准备

1. 把课程 `solver.zip` 里的 `solver.exe` 放在任意文件夹（建议单独复制一份，不要用正式提交要用的那份）。
   本包会核对它的 sha256 必须是 `293867b81dbf0c61…a077247`。
2. 安装 [uv](https://docs.astral.sh/uv/)，然后在本文件夹里运行：

   ```powershell
   powershell -ExecutionPolicy Bypass -File setup.ps1
   ```

   它会建一个 Python 3.11 的 `.venv`，装上固定版本的 numpy 1.24.4、scipy 1.10.1、scikit-learn 1.2.2。
   没有 uv 时：装好 Python 3.11 后 `py -3.11 -m venv .venv`，再 `.venv\Scripts\pip install -r requirements.txt`。
   （课程电脑的 Python 3.14 装不了 numpy 1.24.4，必须用 3.11。）

## 二、检查环境

```powershell
.venv\Scripts\python.exe online_run.py check --solver C:\路径\solver.exe
```

会列出 Python 与各包版本、BLAS 线程数、训练数据和本包文件的校验结果，最后一行是 `check: PASS` 或 `FAIL`。

## 三、核对批次（必做，不调用 solver）

```powershell
.venv\Scripts\python.exe online_run.py verify
```

对第 1–20 次调用，每次都把参考运行里前面各次调用的实测结果交给策略，让它重新算出这一批的 50 组权重，再和参考的那一批逐点比较。
每行输出 `identical` 或 `DIFFERENT`，最后一行是 `verify: PASS` 或 `FAIL`。只想快速看一下可以 `verify --calls 1 3 20`。

**已知情况（请重点关注）：** 结果和电脑的数值环境有关。在参考电脑上测得：

| 设置 | 结果 |
|---|---|
| 默认线程数（20 个逻辑核） | 20 批全部 identical |
| `OPENBLAS_NUM_THREADS=8` | 第 3 批、第 20 批 DIFFERENT |
| `OPENBLAS_NUM_THREADS=1` | 第 20 批 DIFFERENT |

原因：代理模型里高斯过程的超参数优化，会把线性代数的极小舍入差异放大成约 0.01 的预测差异，从而换掉被选中的候选点。
这些舍入差异有两个来源：

- **CPU 指令集**：OpenBLAS 按 CPU 型号选择计算内核。参考电脑上它用的是通用的 Prescott 内核，本包启动时会强制所有电脑都用这个内核
  （`OPENBLAS_CORETYPE=Prescott`），这一项因此可以对齐。
- **线程数**：OpenBLAS 默认每个逻辑核一个线程，而且不能设得比逻辑核数多。参考电脑有 20 个逻辑核；
  **逻辑核数不是 20 的电脑，很可能在某一批出现 DIFFERENT**。`check` 第一行会显示你的电脑有几个逻辑核。

一旦某一批不同，完整运行从那一次起就会走上另一条路，最终结果不会是 103.275（同一策略换种子时的结果在 103–108 之间）。
这正是本包要验证的：请把 `check` 和 `verify` 的完整输出发回。

## 四、完整运行（可选）

```powershell
.venv\Scripts\python.exe online_run.py run --solver C:\路径\solver.exe --out runs\我的名字
```

- 顺序调用 solver 20 次，每次 50 组；每次调用后打印本批最低值、目前最低值，以及这一批的权重和 R 是否与参考运行一致。
- **不会删除或修改 `usage.count`**，会用掉这个 solver 副本的 20 次额度。在自己的开发副本上，跑之前可以按课程说明删掉 `usage.count` 重置额度。
- 每次调用后写日志 `runs\…\online_log.jsonl`；中断后用同一命令续跑，从最后一次完成的调用之后继续。
  如果某次调用已经开始但没有结果，程序会停下并提示（solver 可能已经扣了额度），需要人工判断。
- 每次 50 组的 solver 调用在参考电脑上要 5–11 分钟（2026-11-16 测试时是 643 秒），加上每次 10–25 秒的模型拟合，
  全程约 2–4 小时。请在电脑空闲、不会休眠时运行。

## 五、数据边界与包内文件

- 第 1 次调用前，策略只用 `data/train.csv`（课程提供的 2048 行，按 `data/manifest.json` 核对 sha256）；之后只再用本次运行自己的调用结果。
- `reference/` 里的参考日志只在比对时读取，从不交给策略。
- 策略代码 `policy/` 与仓库分支 `analysis/online-rehearsal` 的提交 `b705164` 中 `analysis/online_rehearsal/` 下的同名文件逐字节相同（换行符除外），
  见 `provenance.json`；`CHECKSUMS.sha256` 列出本包全部文件的哈希，`check` 会核对。

| 文件 | 内容 |
|---|---|
| `online_run.py` | 入口：`check`、`verify`、`run` |
| `policy/policy.py`、`policy/policy_v4.py`、`policy/surrogate.py` | 策略 v4（分楼块下降、平台过滤、后期逐坐标扫描、按命中率分配名额）与代理模型（岭回归趋势 + 局部高斯过程） |
| `data/train.csv`、`data/manifest.json` | 训练数据 |
| `reference/v4_s7_online_log.jsonl`、`reference/v4_s7_config.json` | 参考运行的完整日志（每个点的权重、R、移动类型）和配置 |
| `requirements.txt`、`setup.ps1` | 固定版本的依赖与安装脚本 |
| `provenance.json`、`CHECKSUMS.sha256` | 来源与校验 |

策略的详细说明见仓库 `analysis/online_rehearsal/README.md`（PR #25）。
