# Solver reverse-engineering study

This directory records an isolated development study of the course evaluator. It is included to make the controller analysis and the boundary-search result reviewable.

The extracted runtime showed that the score is produced by eight independent 168-step simulations. At each step, `WeightedMyopicBatteryAgent` evaluates discrete battery and cooling actions, chooses the minimum weighted local objective, advances the CityLearn state, and the evaluator returns the mean of the eight district ramping sums. Because the action is discrete and the state is recursive, the map from weights to `R` is piecewise constant with action-switch boundaries.

The scripts in `scripts/` used a private local copy of the evaluator and extracted dependencies. The bytecode listings in `bytecode/` are the relevant controller, evaluation, runtime, and command-line modules. The large executable, bundled dependencies, private extraction cache, and usage counter are intentionally excluded.

## Verified result

The boundary search started from `R = 104.82363781426102` and produced a candidate with local exact score `104.80496856663376`. The candidate was then checked in one batch against an independent development copy of the original evaluator, which returned:

```text
[104.82363781426102, 104.80496856663376]
```

This is a verified candidate, not a proof of global optimality. The formal course evaluator was not used for this check. Full weights and weekly scores are in `evidence/evidence_boundary_20261112.json`.

## Reproduction notes

The private runtime requires Python 3.11. Set the local extraction layout expected by `scripts/load_bundle.py`, then run `test_sim.py` to reproduce the uniform-weight score before running the boundary search. The public repository does not contain `solver.exe` or the extracted Torch/CityLearn bundle.
