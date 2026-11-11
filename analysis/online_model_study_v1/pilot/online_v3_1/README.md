# Online V3.1 — completed experiment report

**Status:** all 8 replicates (80 online rounds), protocol validation, and the separate global-candidate solver audit are complete.

## Agreed protocol so far

- **Cold start:** every model begins with exactly the original 2,048 training rows. Historical solver labels, including R=112.907, are not added to the training labels. The corresponding weights may be used as a shared candidate-generation anchor for this local model-comparison study.
- **Anchors:** (1) V3 best point, R=113.444; (2) V2 best point, R=112.907; (3) spatially diverse V2 point, R=113.951. The exact solved anchor weights themselves are excluded from candidates.
- **Models:** `local-gp`, `gp-ard` (registered as `bayes-search` with `family=gp`, `kernel=additive-ard`, `lengths=[0.5]`, `noises=[0.03]`, `seed=2026`), `histgb`, `xgboost`, and `boosted-consensus`. Ridge2 is excluded. Fit time is reported separately, not used as a quality penalty.
- **Replicates and rounds:** 8 new replicates × 10 rounds. Each round has one shared 300-point pool for all models. Each model selects the minimum predicted mean, gets only its own new solver feedback, and is refit for the next round. Duplicate selected weights are sent to solver once.
- **Pool composition:** 10 global candidates sampled directly from the valid product of three 7-weight simplexes; 290 anchored local candidates. Local points are allocated about 40% to the R=112.907 anchor and 30% to each other anchor. Aim for an approximately 50/50 split between offline-shortlist directions and broad feasibility-conditioned directions among the 290 local candidates. Generate shortlist candidates only when the original direction yields a valid, unique weight vector at the scheduled anchor and step; do not perturb or relax shortlist directions to force the quota. Fill every unfilled slot with a broad feasible direction, and record requested and achieved source counts.
- **Step coverage per 300-point pool:** exact Euclidean steps 0.01–0.06 receive 36 candidates each; step 0.07 receives 20, 0.08 receives 19, 0.09 receives 18, and 0.10 receives 17. This totals 290 local points plus 10 global points. All models share the resulting pool.
- **Feasible local-point construction:** for anchor A, sample a valid target T in the product simplex and let L=||T-A||. For a scheduled step s<=L, generate W=A+(s/L)(T-A). This is a convex interpolation, so each building's weights remain nonnegative and sum to one, while ||W-A||=s. Apply this generator consistently at all three anchors; reject already-solved and duplicate weights.
- **Unselected points:** do not solve unselected local candidates. After the experiment, solve every unique global candidate not already solved through model selection; keep these global-audit results separate from model trajectories and do not feed them back into training.
- **Evaluation:** report each model's 10-round minimum in each of the 8 replicates and the mean of those 8 minima; do not report a standard deviation for minima because the replicate pools differ. Count first-place wins per shared-pool round across 80 rounds, assigning fractional credit for ties or identical selected points. For prediction error, retain only the selected-point prediction and solver result for each model and round. Fit time is reported separately, not used as a quality penalty.
- **Test set:** V3.1 has no test set. Its primary outcomes are the online search trajectories and solver scores of model-selected points. The existing 100-point pilot test set has already been inspected and is development data, not a blind test set. A sealed test set belongs to the later, frozen-model evaluation stage.

## Candidate-generation constraints

- Keep the 50/50 source split as an approximate target, not a hard constraint. The historical shortlist has 17 exact directions; from the R=112.907 anchor, only 7/17 are feasible at step 0.01, 4/17 at 0.02–0.03, and 1/17 at 0.04–0.10. Use only valid, unique original shortlist directions; fill any shortage with broad directions. Log requested and achieved counts by anchor, source, and step.
- Preserve the selected-point-only error protocol: save each model's prediction and solver result for the point it selected in each round. Do not compute full-pool prediction errors or aggregate bias/MAE/SD for this experiment.

There are no remaining protocol decisions pending. The shortlist share is a target only; broad directions fill any shortage under the original feasibility and uniqueness rules.

## Generated candidate-pool artifacts

- `candidate_generation_manifest.json`: seeds, anchors, candidate-generation rules, data checksum, and per-pool counts.
- `candidate_generation_summary.csv`: one row per replicate and round with counts by source, anchor, and step.
- `replicate_01/candidate_pools/round_01.csv` through `replicate_08/candidate_pools/round_10.csv`: the 80 shared pools. These contain weights and source metadata only; they do not contain solver labels.
- Generator: [`../../generate_online_v31_candidate_pools.py`](../../generate_online_v31_candidate_pools.py). It does not call the solver.
## V3 finding that informs V3.1: shortlist availability vs. selection

V3 had 30 pools × 300 candidates = 9,000 candidate appearances:

| Source | Candidate appearances | Share of pools | Model selections | Share of selections |
|---|---:|---:|---:|---:|
| Broad screened directions | 8,918 | 99.09% | 139 | 92.67% |
| Offline shortlist | 82 | 0.91% | 11 | 7.33% |

The shortlist was selected about **8× its availability share**. By model, Local-GP selected 9 shortlist points out of 30 selections, Boosted-consensus selected 2, and HistGB, XGBoost, and Ridge2 selected none. The absolute count 139 vs. 11 should not be read as broad directions being preferred, because broad candidates made up over 99% of the pool.

The lowest V3 result still came from a broad screened direction (XGBoost, R=113.444). A shortlist direction was also useful: Local-GP selected offline consensus direction 14840 at step 0.05 and solver measured R=114.211. These observations show shortlist attraction/utility for some model trajectories, but do not establish that shortlist directions have better true R: only model-selected points were solved, and just 11 shortlist selections were observed.

## Solver-observed outcomes by source

Deduplicating same-point selections across models leaves 11 unique solver-scored shortlist points and 129 unique broad-direction points (the broad source had 139 model-selection events, including 10 repeated selections). Their observed outcomes were:

| Source | Model-selection events / unique solver points | Mean actual R | Lowest actual R | Unique points with R <= 115.68 |
|---|---:|---:|---:|---:|
| Offline shortlist | 11 / 11 | 115.442 | 114.211 | 6/11 (54.5%) |
| Broad screened directions | 139 / 129 | 115.764 | 113.444 | 58/129 (45.0%) |

This is a modest positive signal for shortlist points around the target region: their selected unique points had a lower mean R and a higher observed target-hit fraction. The broad set nevertheless produced the lowest V3 point. This is not a controlled estimate of direction-family quality: only model-selected candidates were scored, the shortlist sample is small, model choices are confounded with source (only local-GP and boosted-consensus selected shortlist points), and step distributions differ. The 6/11 versus 58/129 comparison is descriptive, not a claim of statistical significance.
## V3.1 logging requirements

Record candidate counts and selection counts by source, anchor, and step; report availability-adjusted selection rates alongside solver outcomes. Keep candidate pools shared across models. If the pool source proportions change, interpret raw source selection counts against those proportions. Treat V3 results as context, not pooled replicates, once the V3.1 generator changes.

## References

- V3 full report and candidate-source audit: [`../online_v3_directional/REPORT.md`](../online_v3_directional/REPORT.md).
- V3 trajectories: [`../online_v3_directional/trajectories_all.csv`](../online_v3_directional/trajectories_all.csv).
- V3 candidate pools and per-pool manifests: [`../online_v3_directional/replicate_01/candidate_pools/`](../online_v3_directional/replicate_01/candidate_pools/), with equivalent folders for replicates 02–03.

## V3.1 online results (model-selected points only)

All eight groups completed with five models and ten rounds per group (400 selected-point records total). Each model started from the same original 2,048 rows within a group, then received only its own ten solver labels. Every selected candidate was verified to belong to that round's shared 300-point pool. The final protocol validation passed, including finite selected-point residuals and one total first-place credit per shared-pool round.

Group minima below are each model's minimum actual R among its ten selected points in that group. The mean is the unweighted mean across eight group minima; no standard deviation of minima is reported because each group used a different candidate pool.

| Model | Group minima (1–8) | Mean of group minima | Lowest selected R | First-place credit / 80 | Model fit time (s) |
|---|---|---:|---:|---:|---:|
| Local-GP | 112.897, 114.526, 114.526, 113.520, 114.526, 114.526, 114.526, 114.526 | 114.196 | 112.897 | 3 | 3,711.1 |
| GP-ARD | 112.658, 112.827, 113.117, 112.658, 112.696, 112.571, 112.161, 112.733 | **112.678** | 112.161 | **33** | 10,146.0 |
| HistGB | 112.499, 112.902, 113.442, 112.677, 113.115, **111.704**, 112.987, 112.462 | 112.724 | **111.704** | 16 | 437.8 |
| XGBoost | **112.344**, 113.035, 112.840, 114.019, 113.439, 113.439, 112.689, 112.548 | 113.044 | 112.344 | 14 | 211.2 |
| Boosted-consensus | 112.888, 112.840, 113.035, 114.089, 113.035, 112.840, 112.239, 112.840 | 112.976 | 112.239 | 14 | 648.4 |

### Initial interpretation

- GP-ARD has the best mean across group minima and the most first-place credit in shared-pool rounds, while taking substantially longer to fit.
- HistGB found the experiment's lowest model-selected value, R=111.704340, and has the second-best mean of group minima with far lower fit time.
- GP-ARD emitted convergence warnings in some fits where length-scale/noise parameters approached configured bounds; this is preserved in run logs and should be considered when interpreting its predictions.
- These are exploratory results from eight groups. They do not establish a universal winner; the result supports carrying GP-ARD and HistGB forward, with XGBoost and boosted-consensus available as faster comparison models.
- The full 400 selected-point predictions, actual solver labels, residuals, round wins, and fit times are in each replicate's `trajectories.csv`. Aggregated group minima are in `v3_1_group_minima.csv`, with machine-readable model summaries in `v3_1_model_summary.json`.
- V3.1 has **no test set**. The already inspected 100-point pilot set is development data. A separate sealed test set should be created and opened only after model and selection rules are frozen.
- All 800 unique global candidate appearances have been solver-audited separately. Those labels are for candidate-generation analysis only and are not fed into any model trajectory.

### Compute cost

Summed recorded model-fit time was about 15,154 s (4.21 h): GP-ARD 10,146 s, Local-GP 3,711 s, Boosted-consensus 648 s, HistGB 438 s, and XGBoost 211 s. Online selected-point solver calls took 2,199 s (36.7 min across 73 invocations). The additional global audit took 9,425 s (2.62 h across 16 invocations). Fit and solver time are kept separate from model quality.
### Candidate-source use during model selection

Across 80 pools, shortlist and broad local directions each supplied 11,600 candidate appearances (48.33% apiece); global points supplied 800 (3.33%). The 400 model selection events comprised 307 shortlist selections (76.75%), 92 broad-direction selections (23.00%), and 1 global selection (0.25%). Relative to availability, shortlist directions were selected 1.59× their pool share, broad directions 0.48×, and global points 0.075×. These are descriptive selection-rate comparisons; solver outcomes for all global points are summarized below.

Per-model counts: Local-GP selected 79 shortlist and 1 global point; GP-ARD selected 32 shortlist and 48 broad; HistGB selected 43 shortlist and 37 broad; XGBoost selected 80 shortlist; Boosted-consensus selected 73 shortlist and 7 broad. Full source, anchor, and step breakdowns are in `v3_1_selection_composition.json`.


### What the selected candidates actually achieved

The 400 model-selection events (not deduplicated across models) had these solver outcomes:

| Candidate source | Selected events | Mean actual R | Lowest actual R | Events at R <= 115 |
|---|---:|---:|---:|---:|
| Broad feasible directions | 92 | 113.509 | **111.704** | 91 |
| Offline shortlist directions | 307 | 114.392 | 112.239 | 235 |
| Global Dirichlet points | 1 | 118.645 | 118.645 | 0 |

These are outcomes conditional on a model selecting the point. They do not estimate the quality of every candidate in a source: unselected local points were not solved, and multiple models can select the same weight. The separately completed global audit does cover every global candidate and found no point below 115.

The lowest V3.1 result, **R=111.704340**, was selected by HistGB in replicate 6, round 6. It came from the broad feasible-direction generator, anchored at the V2 best point (R=112.907), at Euclidean step 0.09. It was not an offline-shortlist ray. The next-lowest, R=112.161299, was selected by GP-ARD in replicate 7, round 4, also as a broad feasible direction from the same 112.907 anchor, at step 0.07. A strong shortlist result was R=112.239049, selected by Boosted-consensus from the diverse V2 anchor (R=113.951), step 0.08. V3's overall minimum, R=113.444010, was likewise a broad screened direction, selected by XGBoost in replicate 1, round 4.

Among selected events, the V3.1 anchor counts were 79 from the 112.907 V2-best anchor (mean R=113.455, minimum 111.704), 229 from the diverse 113.951 V2 anchor (mean 114.150, minimum 112.239), and 91 from the V3-best 113.444 anchor (mean 114.922, minimum 112.897); one global selection has no anchor. These counts reflect what the models chose, not equal solver sampling around each anchor. By step, the most selected were 0.10 (132 events), 0.08 (63), 0.09 (61), and 0.06 (49); the best single point occurred at 0.09. Since availability was deliberately denser at short steps and varied by source feasibility, raw counts should not be interpreted as a step preference without dividing by pool availability. The exact realized counts by anchor/source/step are preserved in the candidate-generation summary and selection-composition JSON.

Together, the V3 and V3.1 low-R examples suggest a useful next candidate-generator study: broad generation can still find the current extremes, while shortlist rays attract most models and produce many useful low-R selections. Shortlist is a promising prior, not a replacement for broad directions. These results do not support declaring 0.09 optimal or claiming broad directions are intrinsically better; the best point is one observation and model selection creates sampling bias.

### Global candidate audit (excluded from training)

All 800 unique global candidate points were solver-scored: one had already been scored as a model selection, and the other 799 were solved in 16 batches. Results were kept out of model training. The minimum was R=117.261576; the median was 160.726, and none of the 800 points had R below 115 (4 were at or below 120). In this run, the global random component added no points in the target low-R region. This is evidence about this particular Dirichlet global generator, not a blanket conclusion about all global exploration methods.

The complete point-level audit is in `global_audit/global_candidate_results.csv`; its summary is in `global_audit/global_audit_summary.json`. The canonical deduplicated solver database has been updated and now contains 1,370 unique non-training points across the tracked stages.
