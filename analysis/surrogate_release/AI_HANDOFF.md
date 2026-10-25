# Instructions for the next agent

Work for member B. Read repository `AGENTS.md`, `docs/roles/B.md`, and this release README before
making changes. Preserve the distinction between the user's frozen v3.2 analysis and the team's
unchanged `sdm evaluate` leaderboard. Ask the user before expanding the task to training or tuning.

Use `packages/SDM374_data_splits_frozen_v1.zip` as the immutable data-membership authority. Verify
its checksums and use the explicit ID lists, not a new random split. It contains data partition
rules only. Scoring lives separately under `scoring/` and in its original ZIP. Do not change any
frozen bytes. SHA256SUMS.json provides the committed handoff inventory.

Implement v3.2 exactly: true y <=132.80; tolerant pair order (delta 0.5) 35/15 points for low/global;
MAE 15/15 points; first-choice regret 20 points. Use current allowed full training-label population
SD for scale. Follow SCORING_V3_2.md for weighting, ties, undefined folds and aggregation. No MSE,
recall or Spearman enters this score.

The 11-group historical selection run is `20261025-204551-artifact-review-v32`. The chosen historical
trimmed lineages scored 74.72, 74.50, 74.21. The supplied final models removed all trimming and use
fixed modal hyperparameters. Their supplied outer-test predictions instead score about 72.34–72.47.
Do not attach the historical trimmed scores to those all-data artifacts. Never evaluate 2048-trained
weights on the same 2048 rows and label that generalization performance.

Final model binary hashes and exact environment are in the model ZIP manifest: Python 3.13.5,
NumPy 2.1.3, sklearn 1.6.1, XGBoost 3.4.1, joblib 1.4.2. Use an isolated environment, not the team's
pinned Python 3.11 environment. Verify and load only the supplied trusted artifacts. Validate all
proposed W vectors with `sdm.simplex.check_weights`. No optimizer integration, uncertainty estimate,
real evaluation or demonstrated new-W improvement has been delivered yet.

Use `verify_release.py` to reproduce the three full-config scores from saved predictions without
training. Historical compressed held-out predictions and fold scores are retained for audit. A
provider's declared training process cannot be proven merely by verifying its test sample IDs.
The original historical run's dirty provenance must stay unchanged after later source commits.
