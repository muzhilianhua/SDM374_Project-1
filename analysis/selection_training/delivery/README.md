# Selection-conditioned retraining delivery

> Historical study completed before real feedback. P1 subsequently measured 119.71206678869203
> (failed); baseline W* 119.0916 is TA-confirmed. Earlier first-test recommendations are superseded.
> Current scores/rules: [team decisions](../../../docs/decisions.md).
> Large evidence and model files: [release restoration](../../submission_handoff/README.md).


The new recipe **did not pass** the prespecified replacement gate.
Working surrogate: **reference_model/model.pkl**. The full fresh corrected model is retained for review,
including negative results. The operational W and original consensus challenger priority are preserved.
No real simulator was called; all newly proposed W have unknown actual R. Historical measured R below
belongs only to held-out rows in the unchanged 2048-row dataset.

## Fresh frozen-v32 evaluation

Three-seed means (higher v3.2 is better; lower MAE/regret/R is better):

| model | v32_score | mae | local_mae | regret | selected_actual_R | coverage |
| --- | --- | --- | --- | --- | --- | --- |
| corrected | 75.29205 | 4.6934 | 6.22285 | 1.10782 | 119.43697 | 0.71216 |
| reference | 75.35905 | 4.69808 | 6.33667 | 1.10782 | 119.43697 | 0.70256 |

| experiment | v32_score | mae | regret | selected_actual_R |
| --- | --- | --- | --- | --- |
| baseline | 66.45076 | 6.91512 | 4.50954 | 122.83869 |
| reference-2026 | 75.11366 | 4.69329 | 1.06324 | 119.39239 |
| corrected-2026 | 75.06793 | 4.69633 | 1.06324 | 119.39239 |
| reference-2027 | 74.0689 | 4.71767 | 2.00842 | 120.33757 |
| corrected-2027 | 73.88163 | 4.70402 | 2.00842 | 120.33757 |
| reference-2028 | 76.8946 | 4.68328 | 0.25179 | 118.58095 |
| corrected-2028 | 76.92659 | 4.67987 | 0.25179 | 118.58095 |

Exact clean CLI run IDs, commits, configs, packages, timing and full metrics are in runs.json.
Per-row held-out predictions and frozen per-fold scores are archived. All outer labels were excluded
from the complete fitted procedure, including residual-correction strength tuning. The frozen protocol
and all its data/scorer hashes are unchanged. These reused folds and three model seeds do not constitute
fresh independent evidence or establish SOTA.

## What changed in training

Keep the proven three-family OOF convex blend. Three internal60/20/20 trials separate base fit,
residual fit and measured-pool selection validation. A fixed-alpha ridge correction uses low-prediction
tail, model disagreement, distance and concentration features. Tune only its multiplier against measured
top1 regret, with a global MAE guard. Refit the corrector on honest nested OOF residuals and the base on
all allowed training rows. No continuous optimizer point receives an invented R label. Internal trials
overlap between repetitions and are tuning evidence only. The final mean's residual corrector uses OOF
labels; the honest final evaluation is the outer protocol, not an in-sample calibration score.

Full-data selected multiplier: **0.0** (outer models can select different values).

| strength | mae | pool_regret | selected_actual_R | selected_signed_optimism | pools | mae_eligible |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 4.90346 | 2.12975 | 124.73102 | -4.02841 | 24 | True |
| 0.25 | 4.89498 | 2.85825 | 125.45952 | -2.65462 | 24 | True |
| 0.5 | 4.91162 | 2.66466 | 125.26594 | -2.20091 | 24 | True |
| 1.0 | 4.98614 | 3.01384 | 125.61512 | -0.53242 | 24 | True |

The error proxy is raw label-excluded absolute residual quantiles in distance bins, enlarged by the
absolute applied correction. It is heuristic; no conditional interval or real-margin guarantee follows.
Training on OOF predictions from smaller models and applying to a full-data base introduces a transfer
assumption. The outer evaluation measures consequences rather than assuming this transfer is harmless.
Measured candidate pools still underrepresent continuously optimized candidates with predictions below
the observed minimum. candidates/central.csv reports each point's maximum standardized correction-feature
magnitude and applied correction. These diagnostics do not certify support or correct optimization bias
outside the measured pool distribution.

## Selection on measured candidate pools and region shift

Disjoint-in-each-fold label-free pools of approximately50 historical held-out rows:

| model | rule | selected_actual_R | regret | signed_optimism | pools |
| --- | --- | --- | --- | --- | --- |
| corrected | mean | 127.10241 | 2.47592 | -0.68848 | 240 |
| corrected | risk | 127.05426 | 2.42776 | -0.98684 | 240 |
| reference | mean | 127.09371 | 2.46722 | -1.21055 | 240 |
| reference | risk | 127.05426 | 2.42776 | -1.49108 | 240 |

Signed optimism = measured R minus predicted R at the selected row. Positive means optimistic prediction;
negative means pessimistic. Pools across model seeds/repeats share observations, so their counts are not
independent sample counts. See measured/fold_selection.csv for whole frozen-fold choices.

Auxiliary label-free KMeans three-region holdouts (separate from v3.2):

| model | rule | regret | worst_regret | selected_actual_R |
| --- | --- | --- | --- | --- |
| corrected | mean | 10.84692 | 19.21964 | 129.2101 |
| corrected | risk | 10.84692 | 19.21964 | 129.2101 |
| reference | mean | 10.84692 | 19.21964 | 129.2101 |
| reference | risk | 10.84692 | 19.21964 | 129.2101 |

See regions/selection.csv for per-region results, restricted-support pool sizes and omitted choices when
a pool is empty. Shift generalization remains unresolved. Auxiliary regions are reused from the prior study.

## Precision audit

All15 previous method representatives, recommendations/controls, and all 200 new proposals are compared
raw, 8/6/4 decimals and <1e-6 cleanup plus8 decimals. Both new fits and legacy HistGB/XGBoost/ExtraTrees
rescore every representation; rankings, risk, novelty and distance are recomputed. Source raw values
remain in precision-*/raw_inputs.json. Canonical exports normalize by exact integer row totals and use
fixed decimal notation. No arbitrary positive lower bound is imposed.

Eight-decimal sensitivity:

| panel | judge | maximum_mean_change | maximum_risk_change | maximum_mean_rank_change |
| --- | --- | --- | --- | --- |
| precision-old | corrected | 0.0 | 0.0 | 0.0 |
| precision-old | extratrees | 0.0 | 0.0 | 0.0 |
| precision-old | histgb | 0.0 | 0.0 | 0.0 |
| precision-old | reference | 0.0 | 0.0 | 0.0 |
| precision-old | xgboost | 0.0 | 0.0 | 0.0 |
| precision-new | corrected | 0.06058 | 0.06058 | 0.0 |
| precision-new | extratrees | 0.0 | 0.0 | 0.0 |
| precision-new | histgb | 0.0 | 0.0 | 0.0 |
| precision-new | reference | 0.06058 | 0.06058 | 0.0 |
| precision-new | xgboost | 0.20746 | 0.20746 | 6.0 |

See precision-*/summary.csv and predictions.csv.gz for 6/4-decimal and cleanup effects. A small input
change can cross a tree split. Surrogate agreement cannot establish that a real simulator treats tiny
positive values and exact zeros identically. Exported candidates are rescored after serialization.

## Candidates and dual stability

exploratory_weights.json is canonical8-decimal rank 1 of `20261101-224445-optimize-selection-study-corrected-search-2040`; raw source W is retained in
exploratory_raw_weights.json. It was fixed before fresh-direction and training-subset audits; no
post-audit best-of-many promotion. Both trained models each produced two50-point CLI batches, with
optimizer seeds 2040/2041, trust radius<=.75, actual neighborhood radii .02/.05 and budget12000 inputs.

| candidate | distance_ratio | reference_mean | corrected_mean | extratrees_mean |
| --- | --- | --- | --- | --- |
| challenger | 0.63817 | 111.419 | 111.419 | 126.09038 |
| operational | 0.541 | 112.08115 | 112.08115 | 125.24726 |
| previous-honest | 0.61913 | 109.63261 | 109.63261 | 125.55675 |
| baseline | 1.74841 | 118.73161 | 118.73161 | 127.20796 |
| p01969 | 0.0 | 118.17768 | 118.17768 | 121.16471 |
| reference-search-2040-r1 | 0.67937 | 110.61312 | 110.61312 | 125.05851 |
| corrected-search-2040-r1 | 0.67937 | 110.61312 | 110.61312 | 125.05851 |

Fresh actual-radius128 directions per radius, seeds 2043/2044/2045, legacy correlated three-judge success
fraction for an improvement>3 over p01969:

| candidate | radius | legacy_consensus_success |
| --- | --- | --- |
| baseline | 0.02 | 0.0 |
| baseline | 0.05 | 0.0 |
| challenger | 0.02 | 0.79688 |
| challenger | 0.05 | 0.44531 |
| corrected-search-2040-r1 | 0.02 | 0.51562 |
| corrected-search-2040-r1 | 0.05 | 0.36198 |
| operational | 0.02 | 0.57552 |
| operational | 0.05 | 0.3151 |
| p01969 | 0.02 | 0.0 |
| p01969 | 0.05 | 0.0 |
| previous-honest | 0.02 | 0.5599 |
| previous-honest | 0.05 | 0.3776 |
| reference-search-2040-r1 | 0.02 | 0.51562 |
| reference-search-2040-r1 | 0.05 | 0.36198 |

Five fresh matched90% training subsets, seeds 2046..2050, refit both complete recipes. Central mean and
risk are distinct from input stability and are not a real-R measurement:

| model | candidate | mean_R | mean_std | risk_delta_operational | risk_improved_subsets |
| --- | --- | --- | --- | --- | --- |
| corrected | baseline | 119.30302 | 2.14294 | 11.25645 | 0 |
| corrected | challenger | 110.51327 | 3.76606 | -0.85214 | 5 |
| corrected | corrected-search-2040-r1 | 110.74938 | 3.95356 | -0.52571 | 4 |
| corrected | operational | 111.20706 | 3.89534 | 0.0 | 0 |
| corrected | p01969 | 115.40233 | 3.60146 | 4.36867 | 0 |
| corrected | previous-honest | 110.599 | 3.73652 | -0.6945 | 4 |
| corrected | reference-search-2040-r1 | 110.74938 | 3.95356 | -0.52571 | 4 |
| reference | baseline | 120.81367 | 0.50381 | 11.25645 | 0 |
| reference | challenger | 112.93722 | 0.5366 | -0.85214 | 5 |
| reference | corrected-search-2040-r1 | 113.26364 | 0.57613 | -0.52571 | 4 |
| reference | operational | 113.78936 | 0.40402 | 0.0 | 0 |
| reference | p01969 | 118.15803 | 0.32669 | 4.36867 | 0 |
| reference | previous-honest | 113.09486 | 0.67744 | -0.6945 | 4 |
| reference | reference-search-2040-r1 | 113.26364 | 0.57613 | -0.52571 | 4 |

Full neighbors, subset indices, selected multipliers, predictions and source provenance are archived.
The first_test_weights.json retains the exact original challenger bytes, and operational W is unchanged.

## Diversified exploratory pack

diversity/weights.json contains20 novel canonical feasible candidates selected by
farthest-first directions from p01969, initialized with the fixed corrected-search rank 1. Minimum pair
distance=0.131084; median=0.202561;
maximum direction cosine=0.976473; distinct nearest observed anchors=
1. Correlated directions can remain; geometry does not create
independent model evidence. This supplementary pack does not replace the previously prepared online50.

The candidate-only pack **failed meaningful direction diversification**: all 20 share one nearest
observed anchor and median direction cosine remains high. Preserve this negative result. A separately
declared adaptive, direction-only extension creates direction_probes/weights.json: the fixed exploratory
point plus19 probes around different farthest-first anchors among the measured lowest 20% R rows,
seed 2051, actual raw perturbation radius .2 times median spacing. No prediction screening or new-model
reselection. All20 serialized points are novel, feasible and inside the trust limit. Anchor measured R
is explicitly separate from the unknown probe R in the manifest.

Direction probes: distinct nearest anchors=20, median direction
cosine=0.135863, maximum=0.684523,
minimum pair distance=0.938483. These are exploration points, not20
independent winners or a claimed lower-R batch. Their central predictions are rescored by all five judges.
The original online50 packet and first-real-test priority remain unchanged.

## Use and replay

```sh
env -u PYTHONPATH uv run --locked --group gui sdm predict \
  --model-run analysis/selection_training/delivery/reference_model \
  --weights analysis/selection_training/delivery/exploratory_weights.json
```

run.py phases: evaluate,fit,search,replay. audit.py phases: precision-old,precision-new,measured,regions,
candidates,diversity. The frozen plan is ../README.md. Full-fit CLI reproduction passed; portable model
CLI replay matches the exported bytes. tests are recorded in test_result.json. All source changes are
on azq within the requested repository. No remote push, official submission or simulator call.
The separate adaptive direction-only pack is generated by probes.py.
