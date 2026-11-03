# Revised offline-candidate decision package

> Historical study completed before real feedback. P1 subsequently measured 119.71206678869203
> (failed); baseline W* 119.0916 is TA-confirmed. Earlier first-test recommendations are superseded.
> Current scores/rules: [team decisions](../../../docs/decisions.md).
> Large evidence and model files: [release restoration](../../submission_handoff/README.md).


Open candidate_table.html for the Chinese user-facing review page. candidate_plan.csv and
decision_manifest.json record the proposed role changes, exact source ranks and pending status.
The first challenger is byte-identical to the audited source; all 9 candidate/control files pass
simplex checks and old/new stack CLI prediction agreement. No true measurements are fabricated.

This post-audit proposal prioritizes the challenger, promotes fixed-blend2027/rank 2 as an alternative,
and makes the second submission depend on verified feedback. It preserves the incumbent and
separate training-subsample gate. The reported baseline 119.0916 remains unverified. Input-neighbor
medians are not actual-R estimates. Exact-radius audit and nominal-step alternatives are distinguished.

The existing consensus wrapper's risk1 identity is preserved; changing its risk coefficient does
not implement max(member mean+k*proxy). No model, dataset, scorer, or original HTML is changed.
Source hashes and export prediction checks are in provenance.json and verification.json.

Final checks: all source hashes and local evidence links match; 115 tests pass with 1 known xfail.
The rendered page was reviewed at a1280px viewport, including all eight candidate-table columns.
