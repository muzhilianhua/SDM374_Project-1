# Training-only candidate research

`elite_candidates.py` is a reproducible research tool for the online policy. It reads only the checked
2048-row training table, fits an ExtraTrees surrogate, and proposes a legal batch from three data-derived
streams:

* local Dirichlet perturbations of the lowest training quantile;
* perturbations around the component-wise elite centroid, which preserves the observed building roles;
* sparse simplex samples for generic boundary exploration without naming a preferred objective;
* a global simplex sample for exploration.

The streams are scored with a lower confidence bound (`mean - k * std`) and are quota-limited so that a
surrogate minimum cannot consume the whole batch. The sparse stream is kept as a separate quota so that
model extrapolation cannot remove all boundary probes. The script never reads `solver/`, historical search
logs, or archived low-R points. Its output is a candidate artifact for review; it is not a formal evaluator call.

```powershell
uv run python analysis/online_policy_research/elite_candidates.py `
  --out analysis/online_policy_research/candidates.json
```

The JSON records the dataset hash, seed, fitted-model settings, elite threshold, data-derived ramping /
degradation ratios, and each candidate's source and surrogate diagnostics. The 21 weights are represented
as three rows of seven values and are checked against the product-of-simplexes constraint before writing.
