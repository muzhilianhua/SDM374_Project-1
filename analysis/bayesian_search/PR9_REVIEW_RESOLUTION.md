# PR9 review resolution

This revision is based on main `6dd38eb65a3e104ea5a4ec46f87764bdb90b2176`, after PR7. The five Bayesian commits were replayed; main's role brief and reviewed framework/model/test fixes were preserved. README and C-role additions are deliberate. The rebased history excludes the regenerable GP pickle. The original PR head remains in a local backup ref.

| Required finding | Resolution | Evidence |
|---|---|---|
| P1: wrong base and 900-file historical diff | Replay the five commits onto main; preserve PR7 fixes; retarget PR9 to main | Only Bayesian additions plus deliberate README/role/.gitignore changes; all18 protected files match main |
| P1: stale measurements and solver-screening guidance | Generator reads confirmed team records, preserves historical evidence and emits decision23 guidance | Submission1=119.71206678869203; baseline W*=119.0916; bar R<115.76; no solver calls; no official submission2 selection |
| P2: whitespace-sensitive acquisition rewriting | Read committed per-acquisition TOML; parse and validate model/mode; reject mismatches before execution | Whitespace and mismatch regression tests; all8 fresh train/search runs record clean source2a63eab |
| P2: direct Git file reads | run.py, audit.py and finalize.py use sdm.runs.git_state | Real packed-ref and worktree regression tests; missing Git returns unknown instead of crashing |
| P2: ignored starts and misleading kernel metrics | Honor configured ARD starts, deduplicate identical pairs, record seven learned lengths and fitted noise separately from initial settings | Duplicate-start test; learned-metric test; full fit uses one configured start |

Additional earlier review findings are also addressed:

- EI rejects heuristic spreads, including GP/tree disagreement envelopes. The blend uses mean/pessimistic acquisition. Its old EI config is removed from current experiments and preserved only in historical evidence.
- A separate ExtraTrees veto is not used in generation or ranking. Its threshold is min(4.9, its submission1 gain); it is informed by only two measurements and is not a guarantee.
- Ray analysis from p01969 toward old ei-r10, r05 and r06 records raw-distance .02 through .15, actual distance and any simplex projection. The current lead is a research point, not the team's submission2.
- GP seed is explicitly documented as a no-op with zero stochastic restarts. Historical seed-audit variation is from the tree half only.
- Optimizer diagnostics record nit, success/status, evaluations and gradient norm. The retained18-iteration budget is not represented as universal convergence. This full refit terminated successfully at17 iterations.
- A fitted33.9MB pickle is no longer tracked or present in the rebased commit chain. Configuration, checksum and metadata support local retraining. Historical metrics/checksum records are archived separately.
- Study logs are UTF-8 text; original historical numerical evidence is preserved.

Validation:156 passed,1 skipped (GUI dependency absent),1 known frozen-PGD xfailed. The15 targeted Bayesian tests include all reproduced configuration/layout/start bugs. The fresh full-data GP matches all64 historical candidate/control predictions at rtol1e-9, maximum mean difference1.42e-14. Eighteen protected files match main byte for byte. Fresh configuration/source Git states are clean. Full test log and current run evidence are under delivery/current/.

No new frozen-v3.2 score is claimed:83.1496 remains the original complete historical evaluation. No course solver was called and no official submission was made. Model-owner B confirmation remains pending; the draft PR must be reviewed before merge. The prior reviewer's independent macOS refit is credited as separate evidence, not a new test performed here.

## Re-review 5768870836 (head311adf)

The reviewer confirmed the original five findings were resolved and identified two remaining
reproducibility failures. Both are addressed in source/config commit54c161d:

| Remaining finding | Resolution | Verification |
|---|---|---|
| Eleven shipped TOMLs request unsupported EI | Set all eleven to pessimistic; retain native raw-GP EI recipes | A parameterized test executes actual optimizer proposal logic for all26 shipped Bayesian TOMLs, including both trust-dirichlet audits; all37 review regression cases pass |
| Published non-editable commands cannot locate repository data | Use uv sync --locked and uv run --locked --no-sync with the supported editable layout; update generator, both generated guides and artifact regeneration instructions | Fresh ASCII-path Git worktree, new virtualenv, PYTHONPATH unset, verified editable distribution/root/data checksum; actual GP train and previously failing SVR optimize complete |

`analysis/bayesian_search/check_editable_replay.py` reproduces the isolated check with Git and uv
on PATH. On Windows/Python3.11, the checkout must use an ASCII path because locale decoding of
editable .pth files can fail with Chinese paths. This is documented; no shared root resolver was
changed. The temporary worktree is removed after text evidence is copied, without archiving binaries.

Clean-environment runs (both source54c161d, dirty=false):

- `20261104-220925-train-bayes-study-gp-ard`
- `20261104-221206-optimize-bayes-study-svr` (one feasible 3x7 proposal)

Command logs, editable/root probe, configs, metadata, metrics and SVR weights are archived in
`delivery/current/editable_replay/`. Latest full-suite counts are in `delivery/current/validation.json`;
the earlier156-test result above is retained as previous-round evidence. These smoke runs establish
CLI reproducibility for the two reported failures, not a new complete model evaluation or real R.
All18 protected files and PR7's existing framework/model implementations remain unchanged.
No solver calls, official submissions or merge; model-owner B confirmation remains pending.
