# Selection-conditioned retraining: frozen plan (2026-11-01)

This study implements the user's request to retrain from scratch after the HTML and tiny-weight
audits. It is a new procedure, not another identical rerun. Member C extends the model plugin area
because the user explicitly requested retraining and selection-bias correction; shared interfaces,
course data, frozen splits/scorer, published artifacts and operational weights remain unchanged.
No simulator calls or remote pushes are part of this study.

## Procedure fixed before evidence

Keep the validated XGBoost/CatBoost/HistGB OOF convex stack as the base. Add a small ridge residual
corrector with fixed alpha=10. Its features are normalized base prediction, low-tail excess,
member disagreement, log nearest-distance ratio, and weight concentration. Weight the lowest
predicted 20% of correction rows four times as much; no synthetic continuous-candidate labels.
Clip residual correction to half the base training-target standard deviation.

Choose a multiplier from [0, .25, .5, 1] using three deterministic 60/20/20 training-internal
splits: base model fits only the first part, residual corrector only the second, and the third
provides measured candidate pools of about 50 rows. These three parts are disjoint in each trial.
Minimize average top1 pool regret, require validation MAE <=1.05 times the raw model's MAE, break
ties toward smaller multipliers. Trials overlap across repetitions: they are tuning, not independent
validation. A zero multiplier is an intended outcome, not a reason to add more profiles.

For the final fit, generate three-fold nested OOF features with complete stack refits (including
blend-weight learning) that exclude the predicted row's label. Fit the final corrector to those
OOF residuals, and fit the final base stack from scratch on all permitted rows. Raw OOF absolute
errors provide the same distance-binned .68 error proxy as honest-stack; add the absolute applied
mean correction to that scale. This triangle-style inflation is heuristic, not a calibrated
posterior standard deviation or a selection-conditional confidence interval. No own-label exclusion
is claimed for the corrected mean's internal residuals. Honest final evaluation happens outside
the entire procedure in frozen outer folds.

## Budget and decision gate

Fresh frozen-v32 CLI evaluations of honest-stack and selection-stack at seeds 2026/2027/2028,
plus a fresh seed 2026 Ridge baseline. All ten frozen folds and scoring definitions are unchanged.
Fresh full-data fits of both models at seed 2026. Four searches: both models, optimizer seeds 2040
and 2041, neighborhood-trust at radii .02/.05, q80, risk1, distance ratio<=.75, <=12000 optimizer
model-input evaluations each; 50 proposals each. Search directions are not used for later audits.

Replace the working predictive recipe only if: three-seed mean v3.2 improves; global and local MAE
are each <=1.05 times the reference at every seed; mean top1 regret does not increase; and mean
actual R of the frozen-fold selected rows does not increase. These last two use historical measured
held-out rows, never the predicted R of a newly generated W. No adaptive profiles or reselection.
Outer scores have already been reused in this project; this is controlled comparison, not an
independent SOTA test. Repeat seeds do not create new measurements.

Separately report disjoint-in-each-fold measured pools of about50 held-out rows: selected actual R,
regret and signed optimism before/after correction. Pool partition seed 2042 is label-free. Auxiliary
three-region holdouts reuse KMeans seed 2031 to reveal shift; they do not enter model selection.

## Precision, stability and diversity

Before new searches, audit the archived15 method representatives, operational W, consensus
challenger, last study's two exploratory points and controls. Compare raw, integer-normalized
8/6/4-decimal versions and <1e-6 cleanup plus8-decimal normalization under both freshly fitted
models and archived HistGB/XGBoost/ExtraTrees. Record mean, risk, rank, feasibility, novelty,
distance, max coordinate and whole-vector changes. Repeat for all 200 new proposals. Canonical
exports use 8 decimals with exact integer row totals; original raw bytes remain archived. Rescore
the exported bytes. Precision agreement in surrogates says nothing about simulator branches.

The exploratory point is fixed as the first new-model search's rank 1 (seed 2040); no post-audit
winner substitution. Audit it, new reference rank 1, the original challenger, operational W,
previous exploratory W, baseline and p01969. Use shared actual-radius directions at .02/.05,
128 each, seeds 2043/2044/2045. Judges include the two new fits and legacy HistGB/XGBoost/ExtraTrees.
Report legacy three-judge success and model-specific deltas; correlation is a limitation.

Five fresh90% training subsets, seeds 2046..2050, refit both complete recipes on matched rows.
Report central mean and mean+error versus p01969 and operational W. This is separate from the
old all-five risk replacement gate. Retain the existing operational W and real-test priority
unless later measured feedback justifies changing them.

Produce a diversified20-point exploratory pack from the four canonical search batches using
label-free farthest-first distance between directions from p01969, seeded with the fixed new-model
rank 1. All retained rows must be novel, feasible and distance ratio<=.75. Report pairwise distances,
direction cosine similarities, nearest-observed anchors and rounded duplicates. This pack does
not replace the already prepared online50-point packet.

## Reproducibility

Commit source/config before CLI evidence. Record run IDs, git state, package versions, dataset and
source hashes, raw/canonical weights, predictions, full fitted bundles, negative results and tests.
Use run.py for CLI phases, audit.py for auxiliary provenance-bearing studies, and publish.py for
the delivery. All writes stay inside this repository on azq; simulator testing remains deferred.

## Adaptive direction-only extension (declared before probe generation)

The fixed four-batch search returned100 distinct W among200 records, all with the same nearest observed
anchor. Even farthest-first selection left median direction cosine .946. Preserve that failed
candidate-only diversification result. Do not change the predictive decision or select a new best W.

Add a separate 20-point exploration probe pack: keep the fixed seed 2040 rank 1, and choose19 other
measured anchors by label-free farthest-first geometry within the measured lowest 20% R rows, starting
from p01969. Use seed 2051, one shared tangent direction per new anchor, and actual perturbation radius
.2 times the full-table median spacing. Canonicalize to8 decimals, recheck feasibility, uniqueness,
novelty and trust, and rescore all five judges. No prediction-based screening, optimizer-label creation,
additional model profile or leaderboard claim. Anchor R is measured; the perturbed probe R is unknown.
This ancillary analysis uses the provenance exception in AGENTS.md. It supplements the existing online
packet and does not replace it. Report whether nearest-anchor and angular diversity actually improve.
