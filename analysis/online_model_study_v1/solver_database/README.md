# Solver 数据库

`solver_observations.sqlite3` 是本项目唯一的、去重后的 solver 结果库；`solver_observations.csv` 是同一张表的便携导出。每个权重向量按 12 位小数规范化后作为主键，所以一组权重最多占一行。初始 `dataset_same8` 的 2,048 个权重会被排除。

## 当前内容

| 来源 | solver 返回记录数 | 去重后权重点数 | 用途 |
|---|---:|---:|---|
| Pilot 在线选点 | 25 | 11 | 在线反馈 |
| Pilot 测试集 | 100 | 100 | 开发阶段预测/选点评估 |
| 近优压力测试 | 50 | 50 | 低 R 邻域压力测试 |
| Online V2 | 118 | 118 | 3 个种子、每种子 10 轮的在线比较 |
| Online V3 directional | 140 | 140 | 离线方向漫游候选池，3 个种子、每种子 10 轮 |
| **合计** | **433** | **419** | 不包含初始 2,048 条 |

Pilot 在线过程中重复求解的 14 次出现已合并到 11 个唯一权重点对应的行中；重复来源和出现次数作为元数据保留，没有重复保存权重/R 行。V3 求解结果已追加。当前初始数据重合排除数为 0。

## 字段

- `weight_key`：21 个权重按 12 位小数序列化后的唯一键。
- 21 个楼宇权重列，以及真实 `actual_R`。
- `source_stages` 和 `source_call_records`：该点出现过的实验阶段与调用记录路径。
- `observations_count`：历史记录中此点出现次数，只是计数，不会产生重复数据行。

## 更新

新增实验的成功 solver 调用应将输入权重和原始输出 JSON 保存在 `online_model_study_v1/pilot/.../solver_calls/` 下。随后运行：

```powershell
.venv\Scripts\python.exe analysis\online_model_study_v1\solver_database\maintain_solver_database.py
```

脚本会扫描已有成功调用、排除初始训练点、按 `weight_key` 去重并更新 SQLite 表和 CSV。失败或超时、没有成功返回 R 的调用不会进入数据库。
