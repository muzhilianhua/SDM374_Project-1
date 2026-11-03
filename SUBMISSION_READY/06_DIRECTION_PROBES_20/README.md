# EXPLORATORY POOL — 20 DIRECTION PROBES

This is a research pool, **not the course's 20-call sequential policy**. Every proposed online call
still requires budget accounting, exclusion of known points and use of earlier feedback under
[decisions 19–22](../../docs/decisions.md).

weights.json contains 20 feasible, novel 8-decimal controllers. Row p01 is P3. The other 19 are
fixed-policy perturbations around different measured low-R anchors. Selection-study direction
diagnostics found 20 distinct nearest anchors and median direction cosine 0.135863; see the original
`analysis/selection_training/delivery/direction_probes/summary.json` and its provenance.

manifest.csv separates historical `anchor_measured_R` from unknown `candidate_measured_R`.
The probes target direction coverage without prediction screening. None has a supplied real result
or a guaranteed improvement. The historical 50-point packet is superseded as well.
