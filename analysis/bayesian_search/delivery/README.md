# Bayesian model exploration and unmeasured candidate handoff

Seventeen new model recipes plus three fresh references were evaluated with the unchanged frozen-v3.2 protocol (ten folds each). Four additional matched model-seed evaluations were run. No simulator calls or official submissions were made.

| Model | v3.2 (higher better) | MAE | Low-region MAE | Top1 regret |
|---|---:|---:|---:|---:|
| baseline-v32 | 66.45 | 6.92 | 11.97 | 4.51 |
| method-blend-stack | 74.39 | 4.70 | 6.39 | 1.82 |
| gp-additive | 75.12 | 4.65 | 6.06 | 2.12 |
| gp-ard | 83.15 | 3.34 | 3.88 | 2.00 |
| gp-ard-log | 82.01 | 3.37 | 3.94 | 2.60 |
| gp-tree-blend | 76.17 | 4.39 | 6.06 | 1.63 |

All lower-is-better columns measure historical held-out labels. Discovery folds are reused; they are not independent confirmation. The strongest GP is an additive Matern kernel with shared seven-objective ARD, a 10% district interaction term and training-only empirical-Bayes marginal likelihood selection. Optimization uses at most18 L-BFGS iterations. Some length/noise parameters reach declared bounds. Posterior standard deviations assume the kernel/noise model and are not validated confidence bounds for optimized points.

The user-reported first submission is the old challenger, rounded to eight decimals. Reported R is approximately119 versus a baseline approximately118; exact measurements and evaluator access are unavailable. It was not added as a training label. New additive GP predicts119.7093; ARD GP121.4889; freshly trained old stack111.6715. This is one post hoc external diagnostic, not proof of calibration.

The fixed equal simple-GP/tree blend improves three-seed average v3.2 from 74.8916 to 76.2998, but mean top1 regret increases from 1.3601 to 1.6343. The declared replacement gate passes: **False**. No operational controller is automatically replaced.

`NEXT_TEST_WEIGHTS.json` is the first novel point suggested for real measurement, source `gp-tree-blend-ei-r06`. It is search rank6, selected from sixty proposals by the predeclared worst panel mean delta versus p01969 plus separation. It is not an official submission and actual R is unknown. Means: additive GP 112.7272, ARD GP 113.5699, tree stack 117.6036, equal blend 115.1654. Nearest-data distance ratio is 0.3880. Panel members are dependent; agreement does not certify real improvement.

`shortlist/weights.json` contains five3x7 points. `candidate_comparison.csv` includes predictions, error proxies, distances and explicitly blank actual labels. Three overlapping90% subset refits and128 directions at raw perturbation radii.01/.03 are archived. The primary beats p01969 in both additive GP and tree central predictions in all three subsets. Tree predictions still shift by up to3.33 around radius.03; local stability is not guaranteed. Other four points are exploratory backups, with weaker subset agreement. Do not replace the primary based on this audit.

`best_observed_115.7648.json` is unchanged historical p01969. Its dataset label is115.7648192592. Check whether observed rows are eligible and whether the current evaluator uses identical conditions before treating it as a fallback.

Model, optimizer and test plugins are included. These model-path changes cross role B ownership because the user explicitly requested new models. No framework interface, frozen split, scorer, course baseline or training table was changed. Git provenance is supplemental because this computer lacks a system git executable; see release_provenance.json. The working branch is `feat/c-bayesian-study`, based on azq651c310. There was no remote push.

The ZIP is an incremental handoff, not a standalone repository. Apply its relative paths over the azq651c310 checkout, or use the complete local branch already prepared beside the archive. It includes the fitted ARD GP, changed plugins/configs and all canonical evaluation evidence. Keep the unchanged course data and framework from the base repository.

## Replay from this branch

```sh
uv run --no-editable sdm evaluate -c experiments/bayes-study-gp-ard.toml
uv run --no-editable sdm predict --model-run artifacts/bayesian_gp --weights analysis/bayesian_search/delivery/NEXT_TEST_WEIGHTS.json
uv run --no-editable sdm optimize -c experiments/bayes-search-gp-ard-ei.toml -n 10 --model-run artifacts/bayesian_gp
uv run --no-editable pytest -q
```

Python3.11 and the unchanged uv.lock are required. `--no-editable` avoids a Python3.11 Windows locale issue with UTF-8 editable path files in Chinese directories. On this machine uv is also callable with `python -m uv`. Full-fit GP training and the primary source search reproduced within rtol1e-9. The fitted GP is in artifacts/bayesian_gp; only load this owned, trusted pickle. Same-platform reproducibility passed; cross-platform reproducibility has not been tested.

The EI pool is an offline next-batch acquisition step. A genuine online Bayesian optimization loop requires measured feedback, which is not available here. The approximate119 report was never used as an exact observation. Before another official submission, request validation of the primary and, if allowed, p01969 under the same evaluator.
