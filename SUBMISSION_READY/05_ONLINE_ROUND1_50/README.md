# SUPERSEDED — 50-CONTROLLER HISTORICAL IDEA POOL

**Do not submit this batch.** Decisions 19–20 allow at most 20 sequential calls, one W per call,
with no repeated known points or training rows. See [current decisions](../../docs/decisions.md).

The original 50 × 3 × 7 weights, individual files, order and IDs are preserved for traceability.
They were assembled as 30 exploitation, 15 exploration and 5 controls under the superseded rules.
The following aliases must not consume a graded call:

| ID | Controller | Known real R |
|---|---|---:|
| r01-004 | P1 / offline submission 1 | 119.71206678869203 |
| r01-046 | Course baseline W* | 119.0916 |
| r01-047 | Training row p01969 | 115.76481925917324 |

r01-003 is P2, a historical recommendation with unknown real R. P1 has no current testing priority.
The updated **pending_results.csv** records the three known results as `known_do_not_resubmit` and
other rows as `unmeasured_not_scheduled`. `historical_pending_results.csv`, `historical_budget.json`
and `historical_constraints.json` preserve the original documents; they are not current instructions.
`current_policy.json` records the current limits. No policy is implemented by this pool.

JSON/CSV/individual weight values agree. Tiny positive entries and original anchor precision are
preserved. The separate 20-point probe pool is not appended here or automatically submitted.
