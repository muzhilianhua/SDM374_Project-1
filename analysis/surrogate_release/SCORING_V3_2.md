# Approved v3.2 score

Protocol: `sdm374.surrogate.tolerance.v3.2`. Lower true `ramping` is better. The final score is the
arithmetic mean of ten outer-fold scores; every fold contributes equally. Scores do not include
a fold-standard-deviation penalty. The authoritative executable definition and policy are in
`scoring/`; the original standalone distribution is preserved under `packages/`.

For each fold, define the low region by **true y <= 132.80**, never by the best 5% within that test
fold. The cutoff is a known-dataset internal benchmark. Let `s` be the population standard deviation
(`ddof=0`) of the current full, unfiltered, unweighted allowed training targets. Do not estimate it
from validation/test targets or change it for a particular training-filter strategy.

| Component | Maximum | Formula |
| --- | ---: | --- |
| Low-region ordering | 35 | 35 × max(0, 2*A_low - 1) |
| Low-region MAE | 15 | 15 / (1 + MAE_low/s) |
| Global ordering | 15 | 15 × max(0, 2*A_all - 1) |
| Global MAE | 15 | 15 / (1 + MAE_all/s) |
| First recommendation regret | 20 | 20 / (1 + regret/s) |

MAE is the mean absolute prediction error, in ramping units. An error of +3 and one of -3 each count
as 3. Lower MAE gives more points when the comparison uses the same scale. Signed mean error
(`bias = mean(predicted - actual)`) is diagnostic only; positive and negative errors can cancel.

For ordering, compare every eligible unordered pair inside the low region, and separately every
eligible unordered pair in the whole test set. Let `g` be their absolute true-value difference:

1. Ignore a pair when `g <= 0.5`. This is the tolerance rule: nearly equivalent true outcomes do not
   lose ranking points merely because their order swaps. It does not reclassify a record as low.
2. Give an eligible pair weight `min((g - 0.5)/s, 1)`.
3. Orient the pair from the truly better record to the worse record. Let
   `d = (prediction_worse - prediction_better)/s`. Credit is 1 if d > 1e-12,
   0 if d < -1e-12, and 0.5 otherwise.
4. `A = sum(weight*credit)/sum(weight)`. Convert this to skill `max(0, 2*A - 1)`.

Thus a weighted accuracy of 0.5 earns no ordering points; perfect eligible ordering earns full
points. These are tolerant pair-comparison scores, **not Spearman correlations**.

The first recommendation is the record with the smallest predicted target in the complete current
test set. Exact prediction ties are broken by ascending candidate_id. Regret is its true target
minus the minimum true target in that same test set. The 0.5 tolerance does not apply to regret,
MAE or the low-region cutoff. A perfect first choice earns 20, regardless of the oracle's value.

Stop and report an undefined fold if training scale is invalid, either region is missing, or no
eligible low-region pair exists. Never drop a difficult fold or fill in a convenient score. Perfect
predictions earn 100. Recall, bias, R2, RMSE and Spearman do not enter this total.

When tuning a future model under this research protocol, maximize the mean v3.2 score of the three
fixed inner validation folds, using the corresponding inner-training scale. The score is a selection
objective; it does not automatically become the model's differentiable training loss. In the current
saved-artifact review there was no retuning. The supplied final configurations use modal historical
hyperparameters and squared-error training; their XGBoost weighting rule is lowest 20% ×3, which
differs from the older finalist's fixed-cutoff ×8 rule. The preserved policy file's historical
`low_training_weight` annotation describes that older finalist, not a mandatory weight rule for every
model or an extra part of the score.
