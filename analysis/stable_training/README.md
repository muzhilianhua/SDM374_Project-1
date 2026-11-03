# Subsample ensembles with honest residual calibration

Member C, authorized by the user on 2026-11-01. All work stays on azq in the designated checkout.
This study extends model plugins because the requested optimizer study requires changing the
training procedure. Existing models, interfaces, course files and frozen evaluation stay unchanged.

## Frozen experiment plan (before new results)

Reference: the existing XGBoost/CatBoost/HistGB OOF stack, configuration
`experiments/method-blend-stack.toml`, v3.2 75.1136584760 at seed 2026.

Train a new `stable-stack` plugin with:

- Five independently seeded 90% subsamples, each fitting the three existing boosted families.
  Mean predictions average the five convex blends. Subsample predictions are correlated, not
  independent votes. All 2048 observations are eligible for training; no synthetic labels are added.
- Three-fold training-only OOF predictions to learn nonnegative, sum-one family weights. Minimize
  low20-x3 squared error plus a subsample-variation penalty (coefficient0.5) and shrinkage0.02
  toward the old fixed prior (.5,.3,.2). Variation is measured on training inputs only; this penalty
  is a training regularizer, not an estimate of generalization error.
- Three additional calibration folds. For each held-out calibration fold, refit the entire core
  (OOF blend weights and five subsamples) without its labels. Absolute residuals divided by raw
  subsample spread with a training-derived floor are calibrated at quantile0.68 in four distance
  bins. Distances are normalized by training spacing. Prediction returns a heuristic error scale,
  not a Gaussian sigma or a guaranteed conformal interval. No calibration label chooses its own
  predictive fit or blend weights. Full-data refitting may still change residual coverage.
- Two fixed regularization profiles, each with 600 iterations: mild (XGB depth 3/min-child5/L2=10,
  CatBoost depth 5/L2=10, HistGB15 leaves/min-leaf30/L2=1); strong (2/10/20, 4/20, 7/40/5).

Discovery uses unchanged frozen-v32 at seed 2026 for reference, mild and strong. Select the highest
v3.2 new profile satisfying score>=reference-0.5, MAE<=1.05x reference, localMAE<=1.05x reference,
top1 regret<=reference+0.5 and coverage in[.55,.80]. If neither qualifies, audit the higher-score
new profile as exploratory, without recommending replacement. Audit the fixed selected profile and
reference at seeds 2027/2028. Replacement additionally requires higher three-seed mean score,
MAE<=1.05x reference at each seed, and average regret<=reference+0.5. No post-audit reselection.

Supplementary label-based checks: three KMeans regions defined from X alone (seed 2031), leave each
region out, refit reference and selected recipe. On each identical measured hold-out pool compare
mean, mean+error, and neighborhood-tail ranking. Report actual selected central-point R, regret,
low20 MAE, dangerous optimism and coverage, including support-restricted pool sizes. These reused
observations are not new independent data. Neighborhood labels remain unknown. This is an auxiliary
stress test, never a replacement for or a score comparable with frozen-v32.

New `neighborhood-trust` optimizer uses<=12000 model-input evaluations: 9600 central proposals
(24 iterations, 16 centers, 25 draws), 64 screened candidates and 16 shared directions at each actual
radius(.02,.05). Minimize max(center mean+error, 80th-percentile neighbor mean+error at each radius).
Compared reference search uses 12000 central proposals. Count all predictions, including reranking;
all proposed W must be feasible, distinct, novel, and distance ratio<=.75. Default empty radii make
the plugin usable as ordinary trust search; study configs explicitly enable both radii.

After fitting selected seed 2026 and reference via CLI, run both search strategies on both models
with matched optimizer seeds 2026/2027. Keep all trials, exact ranks and weights. Audit a fixed
shortlist using fresh directions(seeds 2037/2038/2039, 128 per radius), all five subsample predictions,
and existing judges. Input-neighbor predicted differences are not real improvements. Preserve the
operational W and existing submission proposal unless new evidence supports a separate team decision.

Deliver model/config/metrics, candidate weights, provenance, per-fold tables and a report. Archive
failures and explain tradeoffs even if the new recipe does not improve. No simulator calls or push.

## Follow-up calibration ablation (after discovery, before its own results)

Discovery rejected both regularized recipes: mild73.6451 and strong73.1491 versus reference 75.1137.
Keep the frozen mild audit and all its gates unchanged. Add one explicitly adaptive, fixed ablation:
`honest-stack` preserves the old predictor exactly and changes only residual calibration. Refit the
entire old OOF stacking recipe within three calibration folds, excluding each calibration label
from its own prediction and blend weights. Keep the old0.68 absolute-residual distance-bin rule.
No new hyperparameter search. Evaluate seeds 2026/2027/2028 through frozen-v32 and require identical
mean predictions/score/MAE/regret to each matched reference plus outer coverage in[.55,.80].
This can support correcting the error proxy, never claiming a predictive-score improvement.
Train a portable bundle and run the same seed 2026 neighborhood search, archiving its50 proposals.
This ablation is motivated by observed discovery failures and is not part of the original blinded
profile selection. The calibration proxy remains empirical after cross-fitting and full refitting.
