# Saved-artifact review under the local frozen v3.2 protocol

User-authorized scope: five teammate groups selected from the supplied ranking image plus all six
previous v3.2 finalists. No hyperparameter search, fitting, model modification, or new-W optimization.

Selected teammate groups:

| Label | Experiment | Family |
| --- | --- | --- |
| low90_w20x3 + XGBoost | nestedcv_low90_w20x3_v1 | xgboost |
| low80 + HistGB | nestedcv_xgb_gbdt_histgb_low80_v1 | histgb |
| low60 + HistGB | nestedcv_xgb_gbdt_histgb_low60_v1 | histgb |
| low90_w20x3 + HistGB | nestedcv_low90_w20x3_v1 | histgb |
| tiered + HistGB | nestedcv_tiered_w20x5_w20to40x2_v1 | histgb |

The percentages describe training-target filtering/weighting, never test-set filtering. Low60 trains
on the lowest-target 60% of each fit's training subset, but every outer test is complete. All v3.2
scores use the fixed true-target cutoff 132.80 and the full, unfiltered outer-training standard deviation.

## Reproduce locally

Run in the original Python 3.12 research environment, passing a separate peer-artifact environment:

```powershell
.venv\Scripts\python.exe analysis/artifact_review/review.py --peer-python .venv-peer/Scripts/python.exe
```

The peer environment matches Python 3.13.5, NumPy 2.1.3, sklearn 1.6.1, XGBoost 3.4.1 and joblib 1.4.2.
It only loads 50 selected artifacts and predicts their named frozen outer test folds. The six original
finalists are rescored from their 60 saved genuine held-out predictions, not from ensemble averages.
The scorer's original bytes are in `local_research/scoring_v3_2/`; the immutable split package remains
in `outputs/frozen_data_splits_20261025/`. Expected input locations are checked before running.

Outputs go to `analysis/artifact_review/results/<run-id>/`. They include provenance, exact fold mapping,
artifact and data hashes, raw predictions, full per-fold scores and summary tables. `build_workbook.mjs`
renders the comparison workbook using the supplied artifact-tool runtime.

## Interpretation and protocol boundary

This is a user-authorized local artifact analysis under the repository's `analysis/` provenance rule.
It does not replace or modify `sdm evaluate`, the GUI leaderboard, or the team-wide 5-fold protocol.
The upstream source/framework and frozen course files remain unchanged.

The five peer groups were selected using RMSE internally; the six earlier finalists were selected
using v3.2. This compares saved artifacts on matching held-out folds and does not claim unified retuning.
Peer split identities are declared in the package; no explicit fit-membership lists are included.
We additionally compare each loaded artifact's held-out MAE/RMSE against its recorded model-index values,
which checks reproducibility but is not a formal proof of every training membership.

The five peer candidates were chosen using an earlier ranking, so this is not a fresh independent
test of their superiority. Report fold variation and avoid inferring significance from small mean gaps.
