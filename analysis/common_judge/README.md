# Controlled candidate comparison against course baseline weights

Continue role C on `azq`. This is a comparison of **controller candidates using a fixed surrogate**,
not a new surrogate leaderboard evaluation. The user explicitly defers real-simulator confirmation.

## Protocol fixed before computing this comparison

- Fixed judge: the existing audited OOF-stacking recipe, **one unchanged full-data fit**, model seed
  2026, `20261029-171251-train-method-blend-stack`. Its frozen-v32 recipe evidence is in
  `analysis/method_expansion/delivery/`. No judge is re-fit or selected on candidate predictions.
- Fixed baseline: the course-provided root `weights.json`, unmodified. Score this exact W with the
  same judge as all other candidates. Do not use Ridge's 75.8957 prediction as the comparison target.
- Include all 15 methods in the original discovery manifest: two existing searches each (optimizer
  seeds 2026 and 2027), 50 proposals per search. Reuse exact original floating-point weights, without
  rounding, projection, additional search, or dropping candidates. All 1500 proposals are scored.
- Primary comparison: **seed2026 rank1**, as selected by each method's own mean+proxy search objective.
  Also report seed2027 rank1 and their average. These are the two original top choices, not selected
  using the common judge. All original optimizer parameters and proposal ranks are retained.
- Secondary comparison: minimum fixed-judge predicted R among each method's 100 candidates.
  Explicitly label this **post-hoc reranking** and report exact source run/rank. It is not evidence
  that the original method would have selected that candidate. Report candidate mean/median and
  fraction predicted below baseline to avoid reporting only a minimum.
- Delta = candidate prediction minus baseline prediction (negative is better). Reduction percent =
  100*(baseline-candidate)/baseline. Report predicted mean and heuristic error proxy separately,
  plus the paired change in mean+proxy. These are not confidence bounds or measured improvements.
- Include the existing operational recommendation and best observed row p01969 as references.
  The known R of p01969 is a separate measured column; no other point receives a measured score.
- Report distance to training and distance ratio. The original baseline lies outside the .75 radius
  used by the 15 methods; retain it in every comparison and disclose that the search constraints differ.
- Single batch `sdm predict` scores all inputs, with exact CLI output, source configs/meta/metrics,
  input hashes, packages, source commits, weights and per-candidate predictions archived. Verification
  recomputes summaries from the archived batch; optional CLI replay requires the local judge run.

## Interpretation

This controls the evaluation model, fit, data, feature ordering and baseline W. It does not control
surrogate error. The judge recipe was selected on a repeatedly reused frozen benchmark; the judge
also participates in the blend-stack search and shares model families with several other methods.
The comparison is descriptive and can favor candidates exploiting that judge. It is not an
independent validation, simulator test, statistical significance claim or proof of a real R reduction.
No model, optimizer, frozen protocol or operational submission is changed by this comparison.
