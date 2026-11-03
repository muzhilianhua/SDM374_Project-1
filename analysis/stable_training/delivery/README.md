# Stable training: completed comparison

> Historical study completed before real feedback. P1 subsequently measured 119.71206678869203
> (failed); baseline W* 119.0916 is TA-confirmed. Earlier first-test recommendations are superseded.
> Current scores/rules: [team decisions](../../../docs/decisions.md).
> Large evidence and model files: [release restoration](../../submission_handoff/README.md).


The selected **mild** recipe does not pass the prespecified surrogate-replacement gate. This decision concerns the surrogate recipe only;
the operational W, old best-model artifact and actual submission proposal remain unchanged.
No simulator was queried. The new exploratory candidate has no measured R.

## Frozen-v32 evidence

| recipe | v32_score | mae | v32_local_mae | v32_top1_regret | std_coverage |
|---|---|---|---|---|---|
| reference-2026 | 75.1137 | 4.6933 | 6.3988 | 1.0632 | 0.7036 |
| mild-2026 | 73.6451 | 4.8092 | 6.4296 | 2.2362 | 0.6689 |
| strong-2026 | 73.1491 | 5.1319 | 6.7521 | 2.0132 | 0.6655 |
| reference-2027 | 74.0689 | 4.7177 | 6.3515 | 2.0084 | 0.6953 |
| reference-2028 | 76.8946 | 4.6833 | 6.2596 | 0.2518 | 0.6997 |
| mild-2027 | 73.9397 | 4.8105 | 6.4476 | 2.0084 | 0.6731 |
| mild-2028 | 73.8075 | 4.7985 | 6.4671 | 2.2362 | 0.6794 |
| honest-2026 | 75.1137 | 4.6933 | 6.3988 | 1.0632 | 0.7070 |
| honest-2027 | 74.0689 | 4.7177 | 6.3515 | 2.0084 | 0.6982 |
| honest-2028 | 76.8946 | 4.6833 | 6.2596 | 0.2518 | 0.7024 |

Three-seed mean score: reference **75.3591**, new
**73.7974**. Exact run IDs/configs/provenance are in run_evidence.json;
all outer predictions and per-fold scores are archived. The profile was frozen before seed audit;
no post-audit reselection occurred. A reused benchmark is selection-aware, not independent SOTA.

Mean historical R of the actually selected held-out points (same measured pools across recipes):

| family | model_seeds | selected_actual_R | regret |
|---|---|---|---|
| honest | 3 | 119.437 | 1.1078 |
| mild | 3 | 120.4894 | 2.1603 |
| reference | 3 | 119.437 | 1.1078 |
| strong | 1 | 120.3424 | 2.0132 |

These are measured historical labels of held-out dataset rows, not real evaluations of any new W.

## What changed in training

Five 90% training subsamples, stability-regularized convex family weights, and nested calibration:
each calibration prediction excludes that point's label from every predictive and blending step.
The residual scale is empirical, not a guaranteed interval. Full refitting changes sample size;
report outer coverage rather than inferring validity from internal calibration.

The two regularization profiles were compared under the unchanged protocol. This experiment is
materially different from rerunning the previous model on identical data and parameters.
The nested procedure costs more fits; exact CLI runtimes and thread limits are recorded.

## Adaptive calibration-only ablation

After both initial profiles failed discovery, a separately documented fixed ablation preserved the
old stacking mean exactly and recalibrated absolute residuals using complete label-excluded nested
fits. `calibrated_reference/` contains this fitted model; calibration_ablation.json verifies exact
outer-mean agreement at all three matched seeds. Coverage is reported per seed; no predictive-score
gain is claimed. This follow-up does not change the failed original profile-selection decision.

Calibration-only coverage gate passed: **True**. The working reference is
`calibrated_reference/` when this gate passes: preserve the validated best mean while correcting
the calibration procedure. The regularized `trained_model/` remains an exploratory comparison.
Empirical coverage is not evidence of a statistically significant calibration improvement.

calibrated_exploratory_weights.json is rank 1 of `20261101-184549-optimize-stable-study-honest-neighbors-2026`, using this corrected error
proxy and the same neighborhood-search policy. It remains exploratory. Training-subset mean audits
for this recipe equal the reference by construction; its interval coverage must be checked separately.

## Measured-region stress test (auxiliary)

Three label-free KMeans regions were each held out; models refit using only the other regions.
Each rule selects from the same actually measured held-out pool. The selected central point has
a known historical R; the perturbed neighbors do not. These results are not frozen-v32 scores.

| model | rule | regret | selected_actual_R | mae | low20_mae | coverage |
|---|---|---|---|---|---|---|
| honest | mean | 10.8469 | 129.2101 | 5.4609 | 5.9154 | 0.6909 |
| honest | mean+error | 10.8469 | 129.2101 | 5.4609 | 5.9154 | 0.6909 |
| honest | neighborhood | 10.8469 | 129.2101 | 5.4609 | 5.9154 | 0.6909 |
| mild | mean | 11.0199 | 129.3831 | 5.6707 | 6.0698 | 0.6643 |
| mild | mean+error | 9.7203 | 128.0834 | 5.6707 | 6.0698 | 0.6643 |
| mild | neighborhood | 10.4796 | 128.8427 | 5.6707 | 6.0698 | 0.6643 |
| reference | mean | 10.8469 | 129.2101 | 5.4609 | 5.9154 | 0.6893 |
| reference | mean+error | 10.8469 | 129.2101 | 5.4609 | 5.9154 | 0.6893 |
| reference | neighborhood | 10.8469 | 129.2101 | 5.4609 | 5.9154 | 0.6893 |

Also inspect regions/selection.csv for every fold, the support<=.75 pool sizes, and empty pools.
Mean summaries alone can hide one poor region. Regions and observations were reused; no fresh
labels or independent model-validation dataset were created.

**Regional generalization remains unresolved.** The reference's worst region has first-choice
regret19.2196. Only422/2048 held-out rows satisfy distance ratio<=.75
in this stress test; restricting support changes the eligible pool and may exclude its true best.
The three-region results do not establish a universal winner. Neither neighborhood stability nor
approximately70% global error coverage proves that the selected optimum is reliable under shift.
Future evaluator budget should target insufficiently supported regions and validate new labels.

## Candidates and stability

Two models x two search strategies x two optimizer seeds, 50 proposals per run. Ordinary search
uses 12000 central trials; neighborhood search uses 9600 central trials plus64-point reranking at
actual radii .02/.05, 16 directions each, within12000 optimizer input evaluations. CLI export
predictions are bookkeeping outside that optimizer budget. Fresh audits use 128 directions per
radius and seeds 2037/2038/2039, separate from search directions. Input-neighbor performance remains
a model prediction. Legacy judges are correlated, and ExtraTrees dissent remains visible.

The first real-test priority remains the original consensus challenger: exact source bytes are
copied to first_test_weights.json. New searches do not dominate every stability criterion and are
not promoted automatically. recommendation.json separates this priority from exploratory exports.

candidates/central_predictions.csv contains the full fixed shortlist. candidates/neighbor_summary.csv
contains fresh-direction checks; training_subset_summary.csv contains five new matched90% training
subset audits. New-model mean audits execute its exact final predictive-core fit without residual
calibration, because calibration cannot affect mean predictions. This saves computation without
changing the mean estimator. No error-scale or interval claim is made for these reduced audits.

**exploratory_weights.json** is prespecified as rank 1 of `20261101-184504-optimize-stable-study-mild-neighbors-2026` (seed 2026), not a winner
chosen after inspecting the fresh-direction audit. Original training-subsample replacement rules
remain separate. Do not promote this point to an actual submission solely from self-predicted R.

## Load and reproduce

Both model folders are portable CLI model-run bundles; source binary bytes are preserved.
From the repository root:

```sh
env -u PYTHONPATH uv run --locked --group gui sdm predict \
  --model-run analysis/stable_training/delivery/calibrated_reference \
  --weights analysis/stable_training/delivery/calibrated_exploratory_weights.json
```

The source plan and scripts are under analysis/stable_training/. Execute run.py phases discovery,
audit,fit,calibration,search,calibration-search; then diagnostics.py regions/candidates and publish.py. Source hashes and model
versions are recorded. checksums.json covers the package except itself.
