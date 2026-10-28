# Member D — mechanism analysis and experiment design

## Mission
Understand how the controller responds to the weights and turn that into priors for B and C, and
choose the online strategy by simulating it offline before spending real budget.

## You own
`src/sdm/replay.py` (create it), `analysis/mechanism/`, `experiments/replay-*.toml` if needed.

## Ask A first
Anything that changes how runs are recorded or how `X, y` are assembled.

## Backlog — offline (until the week 8 lab)
- [ ] Mechanism notes (`analysis/mechanism/README.md`): ratio effects within a building, additivity
      across buildings, low sensitivity of building 2; translate each into a modelling prior for B
- [ ] Extend `analysis/coupling/study.py` (semi-synthetic truths, exact regret) into the online replayer;
      the truth × surrogate interaction there is the main open question for the evaluator probes
- [ ] Offline replayer `src/sdm/replay.py`: hold out a pool from the 2048 rows; each round a strategy
      (any registered model + optimizer) picks 50 rows from the pool, "measures" them from the table,
      retrains; report best-so-far per round. Use `sdm.registry.build_model` / `build_optimizer`
- [ ] Compare exploit / explore / probe ratios with the replayer; recommend round-1 settings
- [ ] Design round-1 probes (repeat the offline best to test determinism, scale a row ×2 to test
      renormalisation, baseline `weights.json`, training row `p01969`)

## Backlog — online (weeks 8–10)
- [ ] Review C's proposals each round against the mechanism notes
- [ ] Diagnose points that measure far worse than predicted (extrapolation, regime switch, model error)
- [ ] Decide when to switch from exploration to exploitation

## Acceptance criteria
- Replay results are reproducible (fixed seeds) and reported as best-so-far curves per strategy.
- Recommendations cite the replay or analysis that supports them.

## Interfaces
Consumes `Surrogate` / `Optimizer` through the registry; provides strategy settings to C and priors to B.

## Handoff
PR or note with the replay command, seeds, the curves and a one-paragraph recommendation.
