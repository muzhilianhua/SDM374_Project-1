# Online round 1: prepare 50 feasible controller weights

Role C, continuing on `azq`. Preparation only; the user defers actual simulator calls.

Completed [50-point packet and handoff](delivery/README.md), [batch JSON](delivery/weights.json),
[PPT-column CSV](delivery/weights_w0_w20.csv). All50 are feasible and mutually distinct; one known
offline row is deliberately included for calibration. Real evaluator calls remain zero.

## Course requirements checked before selection

Source: the locally supplied `SDM374_Project1_v3.pdf`, pages3 and7, visually checked after rendering.
Page3 requires each building's seven weights to be nonnegative with at least one positive entry;
columns are cost, carbon, peak, discomfort, degradation, ramping, resilience, buildings1/2/3.
Page7 allows at most20 iterations, at most50 new evaluated records per iteration, and minimizes R.
No file schema, decimal precision, trust radius or explore/exploit ratio is specified on those pages.
Use the repository's sum-one-per-building representation, which meets the stated feasibility rule.

## Prespecified first-round portfolio

Exactly50 distinct W: **30 exploitation,15 exploration,5 diagnostic controls**. This ratio is a
practical first-round choice, not a requirement from the slides or a replay-validated optimum.
Use fixed RNG seed20261029. Preserve all named reference/controller weights at original precision.
Round newly sampled points to8 decimals and repair the largest component to keep each sum exactly1.

- Exploitation: keep the fixed-blend original primary W, its separately labeled post-hoc W, the current
  operational W and the prior consensus challenger. Fill to30 by increasing maximum mean+proxy over
  stack, weighted CatBoost, HistGB and XGBoost. Novel points only, distance ratio<=.75, minimum batch
  separation=.25 times median training spacing. The four anchors satisfy this same separation rule.
- Exploration:15 novel points, ratio<=1, stack prediction<=baseline stack prediction+15. Select using
  strong-committee disagreement +.5*stack error proxy +2*distance-to-selected/training spacing, with
  the same minimum separation. ExtraTrees is reported diagnostically and does not select the batch.
- Controls: course baseline W, known measured best p01969, uniform weights, pure ramping weights,
  and50/50 ramping/resilience. The last two probe physical-objective corners, despite greater model
  extrapolation. p01969 is an intentional one-slot calibration remeasurement; other points must be
  novel relative to the offline dataset. All50 points are distinct from one another.
- Candidate pool: all1500 existing method proposals, the named anchors/controls,3600 Dirichlet
  perturbations around64 low-R measured centers plus the four exploitation anchors (concentrations
  30/100/300, alpha floor .05), and600 global product-simplex Dirichlet(.6) coverage draws.
- One unchanged packaged stack model and the four existing local judge models score the pool through
  `sdm predict`. No surrogate is retrained. Predictions/error proxies are not simulator measurements.

IDs r01-001..030 are exploitation,031..045 exploration,046..050 controls. All50 slots, including
controls and the deliberate known-row repeat, count against the planned round budget. The packet is
not an official submission or an implementation of the project's future shared online log schema.
