# Candidate perturbation audit

> Historical study completed before real feedback. P1 subsequently measured 119.71206678869203
> (failed); baseline W* 119.0916 is TA-confirmed. Earlier first-test recommendations are superseded.
> Current scores/rules: [team decisions](../../../docs/decisions.md).
> Large evidence and model files: [release restoration](../../submission_handoff/README.md).


Independent audit of the supplied HTML; no retraining or real-evaluator calls.
Single-point differences for all five HTML shortlist candidates match displayed values using the
archived judges. This does not recover the missing original direction generator or model binaries.

## Findings and submission implication

- The challenger ranks first by the predefined success fraction in the 29-candidate panel at both
  radii and separately at each of the three audit seeds. This supports prioritizing it for a real
  test over the original incumbent and the HTML's fixed-blend-rank 1 alternative.
- At actual distance0.02, success is81.90% and the worst-judge median delta is-4.1698. At actual
  distance0.05, success is46.55% and median delta is-2.8265. The HTML's82% is numerically close
  under a different, larger sample; 62% is not reproduced. Neither percentage is an exact replay.
- The closest0.05 competitor is fixed-blend search2027 rank 2 (`pool-0054`), success45.05%, only
  1.50 percentage points below challenger. Its paired candidate-minus-challenger conditional95%
  direction-sampling interval is[-3.80,+0.80] percentage points. This does not establish a clear
  advantage over that runner-up. Its median delta(-2.8637) is also slightly better than the
  challenger's, so no candidate dominates every robustness metric.
- The two40-direction reconstructions give challenger77.5%/47.5% (nominal projection) and
  77.5%/35.0% (clipping/normalization), respectively at nominal0.02/0.05. Geometry and small-sample
  variation materially affect reported percentages; the nominal displacement shrinks after
  enforcing simplex feasibility. The main audit controls actual distance instead.
  Under clipping/normalization at nominal0.05, challenger is not first: stack-search2027/rank 2
  and fixed-blend-search2027/rank 2 both achieve40%, versus challenger's35%. Under nominal
  projection it ties stack-search2027/rank 2 at 47.5%. Thus first place is not invariant to the
  perturbation definition, especially with only 40 directions.
- Conclusion: retain challenger as a priority real-test candidate, without changing the original
  operational-W gate or claiming a real-R margin. Its small-perturbation advantage is stronger
  than its large-perturbation advantage. These sampled-neighborhood results are conditional on
  the chosen Gaussian/projected perturbation distribution, not a worst-case robustness proof.

Panel: 29 fixed candidates, including every original proposal with worst-judge delta<-6,
the HTML shortlist, and original seed 2026 fixed-blend rank 1. Neighbor count: 93728.
Main audit: three new direction seeds, 512 shared directions each, actual Euclidean radii0.02/0.05.
Success means all three core judges predict R at least3 lower than their own p01969 prediction.
All reported improvements below are predicted differences, never actual-R estimates.

## Main shortlist results

| candidate | radius | success_fraction | median_delta | rank |
|---|---|---|---|---|
| challenger | 0.02 | 0.8190 | -4.1698 | 1.0 |
| stack-2027-r2 | 0.02 | 0.6868 | -3.6181 | 3.0 |
| incumbent | 0.02 | 0.5859 | -3.2646 | 11.0 |
| fixed-2027-r1 | 0.02 | 0.5326 | -3.1454 | 18.0 |
| fixed-2026-r1 | 0.02 | 0.5299 | -3.1125 | 19.0 |
| cat-2026-r4 | 0.02 | 0.1328 | -1.4424 | 29.0 |
| challenger | 0.05 | 0.4655 | -2.8265 | 1.0 |
| stack-2027-r2 | 0.05 | 0.4219 | -2.7262 | 3.0 |
| fixed-2026-r1 | 0.05 | 0.3665 | -2.4895 | 6.0 |
| incumbent | 0.05 | 0.3529 | -2.4350 | 7.0 |
| fixed-2027-r1 | 0.05 | 0.3105 | -2.1331 | 13.0 |
| cat-2026-r4 | 0.05 | 0.0560 | -0.5548 | 29.0 |

Rank is within the complete screened panel, not only this shortlist. Lower median delta is better;
higher success fraction is better. See main_ranking.csv for every candidate, per_seed_summary.csv
for seed sensitivity, and paired_comparisons.csv for paired direction differences.

## Scope of the conclusion

The HTML's82%/62% cannot be claimed as exactly reproduced without its original scripts/samples.
html_neighborhood_comparison.csv compares them with both reconstructed40-direction nominal-step
methods and the larger actual-distance audit. summary.csv reports achieved distances as well.
Three core judges are correlated: the stack contains HistGB/XGBoost-family members. Relative
prediction to a known reference does not remove unknown differential surrogate error.
ExtraTrees dissent is retained in pool_scores.csv; it is not silently used or removed from the
three-judge endpoint. Real baseline R119.0916 mentioned in HTML is not verified by this experiment.

Training-subsample stability and input-neighborhood stability are different checks. This audit
does not alter the prior operational-W replacement gate. It can support a prioritized real test,
but cannot establish the true improvement, a true-R margin of3, or global optimality.

## Reproducibility

provenance.json records committed source, HTML/model/data hashes, packages and every CLI command.
panel.json identifies exact source ranks and weights. Compressed weights, predictions and sample
tables preserve every evaluated neighbor. No frozen data, model, candidate or scorer is modified.
