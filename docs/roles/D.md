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
- [ ] Offline replayer `src/sdm/replay.py`: hold out a pool from the 2048 rows; a policy (any registered
      model + optimizer) makes 20 sequential calls, each picking one row from the pool, "measuring" it from
      the table and retraining; report best-so-far per call (decision 19). Use `sdm.registry.build_model` /
      `build_optimizer`
- [ ] Compare exploit / explore settings for the 20-call policy with the replayer
- [ ] Renormalisation probe: a training row scaled ×2 answers it in one evaluation, and local development
      runs are free (decision 20), so it does not need one of the 20 graded calls. Never spend a graded call
      on a known value: baseline `weights.json` is 119.09, `p01969` is 115.76, offline submission 1 is 119.71

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
