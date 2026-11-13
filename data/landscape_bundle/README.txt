landscape_bundle — 21 维单纯形地势打包
=============================================

内容
----
1. landscape_data.csv
   合并后的唯一权重点表（9798 行）。
   列：
     source   "solver_real" = 真实求解器调用；"course_train" = 课程提供的 2048 行离线表
     R        求解器返回的 R 值（越小越好）
     row_sumb1 / row_sumb2 / row_sumb3  每栋楼权重和（恒为 1.0）
     sig_thr005  结构签名：每栋楼权重 ≥ 0.05 的特征索引
     dominant   每栋楼主导特征索引（argmax）
     w_b1_cost ... w_b3_resilience  21 维权重，每 7 列一组分别是楼 1 / 2 / 3

2. landscape_summary.json
   地势概览：全局统计、分源统计、结构签名分布、全局最优 + 3R 内的盆地清单、量化阶梯证据。

3. landscape_signatures.json
   以结构签名（sig_thr005）为键，记录每个签名下的最优点 R、出现次数、最优点权重。

4. （本文件）

使用说明
--------
- 两个数据源在 21 维权重空间上有交集：去重后 solver_real 7750 点，course_train 2048 点，
  合并唯一 9798 点。重复权重只保留 solver_real（真实求解器）的 R。
- 权重顺序严格遵循 solver.exe 接口：按楼宇分组，每楼内特征顺序 = [cost, carbon, peak,
  discomfort, degradation, ramping, resilience]。楼 1 的 7 列在前，楼 2 次之，楼 3 最后。
- row_sumb1 / row_sumb2 / row_sumb3 恒为 1.0（任何权重归一化误差在打包时都已修正）。

生成时间
--------
landscape_bundle.zip 打包自 optimization/ 下的当前数据，全部 R 值来自
  * solver_real：optimization/eval_log.jsonl 的 7763 条记录（去重后 7750 唯一权重）
  * course_train：SDM374_Project-1_online/dataset_same8/train.csv 的 2048 行

地势结论摘录（详见 landscape_summary.json）
--------------------------------------------
- 全局最优 R = 104.851733（来自 solver_real）。
- 地形是分段常数阶梯，单点凹口宽度 ≈ 0.0007 权重单位，常规粗搜无法突破。
- 所有低区共享同一结构族：三栋主导特征均为 F5 或 F6，核心支撑集 {F4,F5} 或 {F5,F6}。
- 在全局最优 +3R 之内归并到 57 个签名，本质是同一主盆地 + 少数同支撑集下的量化平台。

技术联系
--------
optimize_weights_phaseE6.py — 产生 solver_real 中 E6_C0/C1/C2/C3/C4 的 1195 点
plot_phaseE6.py              — 地形图 e15–e18
RampingSolver搜索地形分析与在线策略推荐.docx — 完整分析报告
