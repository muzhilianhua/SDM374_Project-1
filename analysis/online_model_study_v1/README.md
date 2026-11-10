# Online surrogate model study V1

This folder records the online surrogate comparison agreed for the SDM374 second phase.

- Start from all 2,048 rows in `dataset_same8` for each model.
- Compare `local-gp`, `HistGB`, `XGBoost`, `boosted-consensus` (HistGB/XGBoost ensemble), and `ridge2`.
- The historical V1 pilot used pre-generated 100-point shared pools (50 global simplex points + 50 low-R local perturbations), five rounds per model, and a 100-point development test set.
- V2 used 3 seeds × 10 rounds with 300-point global + multiscale pools. V3 used 3 seeds × 10 rounds with 300-point directional-roaming pools based on the offline low-R direction/step search.
- Each model selects the lowest predicted mean from a shared round pool and receives only its own online feedback. Duplicate same-round solver requests are collapsed.
- Primary online outcomes are best-so-far actual R, target hits at R ≤ 115.68, between-seed variability, and fit time. Prediction checks use MAE, near-best-band MAE, and selected-point regret; Spearman is not a primary metric.
- No full 20-round-per-model run or final V4.1 blind test has been run. Existing pilot/stress labels are development evidence and have been viewed.

## Main documents

- Complete work summary, experiment table, reusable-data inventory, and next steps: [`deliverables/WORK_SUMMARY.md`](deliverables/WORK_SUMMARY.md).
- One-page A4 LaTeX work report: [`deliverables/runze_online_work_report.tex`](deliverables/runze_online_work_report.tex); compiled PDF: [`../../../output/pdf/runze_online_work_report.pdf`](../../../output/pdf/runze_online_work_report.pdf).
- V1 metrics: [`pilot/results/REPORT.md`](pilot/results/REPORT.md).
- Near-best stress test: [`pilot/near_best_stress/REPORT.md`](pilot/near_best_stress/REPORT.md).
- V2 multi-seed comparison: [`pilot/online_v2/REPORT.md`](pilot/online_v2/REPORT.md).
- V3 directional-roaming comparison: [`pilot/online_v3_directional/REPORT.md`](pilot/online_v3_directional/REPORT.md).
- Canonical deduplicated solver database and maintenance instructions: [`solver_database/README.md`](solver_database/README.md).

## User-provided candidate-generation reference

润泽提出：Online 阶段的代理模型比较先沿用离线寻优中围绕低 R 权重做可行方向漫游的思路；离线阶段已经用模型研究过方向，排除部分差方向并挑出有希望的方向。该思路是候选生成的参考，不预先断定候选点真实 R，也不让不同模型各自产生候选池。

The offline `offline-final` research used 20,000 zero-sum-per-building unit directions around `p01969`, removed directions too aligned with two measured failures, and scanned step lengths. It preferred short steps (up to 0.05), with direction 14840 as the GP-consensus recommendation; the later submitted direction was 15713 at step 0.05 and was reported at R=115.68. These are model-screened references, not a guarantee that all nearby points are good. Online V3 keeps one common, pre-generated directional pool per round for all five models, so the comparison still measures model choice on identical options.

## Data inventory

The canonical `solver_database/solver_observations.sqlite3` and CSV contain 419 unique non-training weights from 433 successful solver observations; no initial 2,048 training weights overlap. Unique counts by source: pilot online 11 (25 observations with repeats folded), pilot test 100, near-best stress 50, V2 118, and V3 140. The externally reported offline point at direction 15713 is not counted because its exact local solver output is unavailable; V3 excluded it from re-query.

## Directory map

```text
protocol/                 # Protocol, seeds, metric definitions, provenance
candidate_pools/          # Shared V1 candidate pools and generation manifest
solver_database/          # Canonical SQLite/CSV of unique solver-evaluated non-training weights
pilot/online_runs/        # V1 per-model trajectories, predictions and solver feedback
pilot/test_set/           # 100-point development test set; labels already viewed
pilot/results/            # V1 result table and report
pilot/near_best_stress/   # 50-point near-best stress test and predictions
pilot/online_v2/          # Three seeds × ten rounds, multiscale pools and traces
pilot/online_v3_directional/ # Three seeds × ten rounds, directional pools and traces
final_v4_1/               # Reserved for a future sealed test; not run
deliverables/             # Comprehensive work summary and LaTeX report source
```

Raw initial training data remains in the project dataset directory; this study stores its checksum and derived artifacts only.
