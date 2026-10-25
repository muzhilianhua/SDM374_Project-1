# Model selection and final results

## Historical selection evidence

Analysis id: `20261025-204551-artifact-review-v32`. Exactly five supplied groups from the screenshot
and six previous finalists were compared on the same ten complete frozen outer test folds. This was
saved-artifact inference/rescoring, with no fitting or hyperparameter search. Peer candidates had
historically used inner RMSE selection; original finalists used v3.2 selection. This is not a claim
of uniform retuning. The original run was dirty; its immutable provenance remains unchanged even
though the source has now been committed.

| Rank | Training strategy and model | v3.2 | Global MAE | Low MAE | First-choice regret |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | low90_w20x3 + XGBoost | 74.72 | 5.21 | 6.40 | 1.59 |
| 2 | low80 + HistGB | 74.50 | 5.48 | 6.13 | 1.48 |
| 3 | low60 + HistGB | 74.21 | 7.35 | 6.25 | 0.41 |
| 4 | low90_w20x3 + HistGB | 73.73 | 5.09 | 6.41 | 1.15 |
| 5 | tiered + HistGB | 73.26 | 4.90 | 6.85 | 1.59 |
| 6 | Original XGBoost | 73.04 | 4.81 | 6.97 | 2.24 |
| 7 | Original HistGB | 72.65 | 4.85 | 7.30 | 2.36 |
| 8 | Original weighted XGBoost | 72.18 | 4.97 | 6.44 | 1.49 |
| 9 | Original Extra Trees | 68.44 | 5.44 | 10.14 | 4.51 |
| 10 | Original RBF kernel ridge | 67.01 | 6.63 | 10.59 | 5.04 |
| 11 | Original quadratic ridge | 66.19 | 6.86 | 12.23 | 4.51 |

The three historical leaders were selected as model lineages. Low60/low80 used only the lowest-target
60%/80% of their training data. Low90_w20x3 removed the highest 10% and gave the lowest 20% weight 3.
All tests stayed complete. The first lineage balanced the score components best; low80 reduced
low-region numerical error; low60 gave the best low-region ordering and first choice, with much
worse global error. Their small mean-score gaps do not establish a reliable ordering of superiority.

## Delivered full-data models

The provider then removed all trimming, used modal parameter choices from the historical ten folds,
and refit each model on all 2048 samples. No additional search was performed. This changes both
training support and parameter-selection procedure relative to the trimmed fold-wise workflows.
Consequently, historical scores must not be attached to the delivered weights.

| Model | Fixed parameters | Full-data weighting | Seed |
| --- | --- | --- | ---: |
| xgb_w20x3_full | depth 3; rate 0.1; 600 trees; subsample 0.9 | Lowest floor(0.2*2048)=409 targets weight 3; others 1 | 20261024 |
| histgb_low80_full | 15 leaves; rate 0.05; 600 iterations; L2 0 | Uniform | 20263024 |
| histgb_low60_full | 15 leaves; rate 0.1; 600 iterations; L2 0 | Uniform | 20263025 |

The strings low80 and low60 are lineage names, not current data fractions. Their HistGB distinction
is primarily learning rate. XGBoost's lowest-20% training weight rule is not the scoring low region
(y <=132.80). The three artifacts share raw 21-feature inputs without a fitted scaler. Exact class,
parameters, environment, seeds and binary hashes are in the supplied package manifest.

## Saved full-config outer-test predictions rescored under v3.2

Analysis id: `20261025-152147-release-verification` (UTC id). The package contains separate predictions
from refitting each fixed configuration on every complete outer training set. We verified their test
membership and true labels against the frozen manifest, then applied the unchanged v3.2 scorer.
These are provider-supplied evaluation predictions, not predictions obtained by testing the final
2048-trained weights on their own training data. Outer-fold training membership and execution are
provider-declared; the package does not contain those 30 fold weights for independent recomputation.

| Rank among delivered configs | Model | Total | Low order /35 | Low MAE /15 | Global order /15 | Global MAE /15 | First choice /20 | Sample SD |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | histgb_low60_full | 72.47 | 18.73 | 10.55 | 13.18 | 11.49 | 18.52 | 7.41 |
| 2 | histgb_low80_full | 72.35 | 18.89 | 10.55 | 13.19 | 11.50 | 18.21 | 7.02 |
| 3 | xgb_w20x3_full | 72.34 | 19.11 | 10.52 | 13.07 | 11.41 | 18.22 | 6.53 |

| Model | Global MAE | Low MAE | Global bias | Low bias | First-choice regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| histgb_low60_full | 4.818 | 6.668 | -0.015 | 5.452 | 1.592 |
| histgb_low80_full | 4.792 | 6.673 | -0.042 | 5.787 | 1.877 |
| xgb_w20x3_full | 4.958 | 6.748 | -0.109 | 6.325 | 1.758 |

Full-data configurations improve global error relative to their trimmed ancestors, while their
low-region ordering and total score weaken. HistGB_low60_full has the highest descriptive v3.2
mean among the delivered configurations; histgb_low80_full has the smallest global MAE;
xgb_w20x3_full has the strongest low-region ordering component. Differences in total are small
relative to fold variation. No deployment winner should be inferred from this tiny mean gap alone.
The originally tuned raw XGBoost finalist still has a higher descriptive mean (73.04) than these
three full configurations on the reused partitions; the new package is not evidence of a further
score improvement.

The supplied all-data weights' in-sample MAEs (about 1.68, 1.22, 0.44 respectively for XGB, HistGB80,
HistGB60) only check training fit. They must not replace the outer-test MAEs above. The dataset has
been used repeatedly for strategy selection, including selecting the modal parameters across folds;
these reused outer results are selection-aware diagnostics, not a fresh independent final test.

## Interpretation for W optimization

Keep all three supplied full-data artifacts as candidate surrogates for downstream study. Using all
known samples avoids deliberately discarding high-target training observations, but does not prove
accuracy outside the observed W distribution. All three are piecewise-constant tree models; direct
input gradients are unsuitable for the baseline projected-gradient optimizer. They supply no
calibrated per-point uncertainty. CLI plugin integration and optimizer compatibility remain separate
work, and no new W was proposed or evaluated in this release.

Low-region ordering measures whether clearly different good observations are ordered correctly.
First-choice regret measures how much is lost by selecting one held-out record. Neither directly
measures a newly optimized W. With only these 2048 measured records, new-W improvements remain
unverified until the real evaluator is available.

Sample SD uses ddof=1 to match the teammate screenshot; historical population SD uses ddof=0.
Both are descriptive fold spread, not confidence intervals or significance tests: the folds reuse
records and training sets overlap. We do not adopt the supplied Word report's independent-fold
confidence or statistical-equivalence claims. Its 30-candidate overview is attributed background;
our independently reproduced selection scope remains the specified 11 groups, not all 30.
