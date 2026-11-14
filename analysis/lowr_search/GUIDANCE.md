# Guidance for the online model and point-selection research

What the evaluator studies (this folder, `analysis/mechanism/` T20, `analysis/gp_assumptions/` T35) imply for
the surrogate (T12–T14), the batch policy (T15, T16), the policy comparison (T18, T19) and the pre-submit
check (T08). Written 2026-11-13 by 张子木 (A). Recommendations, not decisions: each owner decides.

## 0. Boundary

- **Allowed** (decision 27, the user's judgment of 2026-11-11, **not yet confirmed by the TA**): use local
  evaluator runs to choose models, policies and hyperparameters.
- **Not allowed** (decisions 19, 27, 29): any point, R value or weight measured locally in the submitted code,
  its configs, warm starts or exclusion lists.
- **Grey zone, open question** (`docs/decisions.md`, Open questions): a structural prior such as "the low
  region uses degradation + ramping with r ≈ 0.25" is in effect the location of the optimum. Recommendation:
  every prior the policy uses must be derivable from the 2048 training rows alone. `train_priors.py` derives
  the ones cited below; its output is `train_priors.log`.

## 1. Structure of the evaluator

| Structure | Evidence | Consequence |
|---|---|---|
| One basin, on the simplex boundary (minor weights small or zero), about 1.9 spacings from the training rows | E3; T35 | A first batch that stays near the data cannot reach it; a trend has to carry the search out |
| Additive over buildings; strictly additive at fine scale in the low region (error ≈ 0.03) | E1, E4, E10; T20; T35 | Change one building per candidate, combine the buildings' winners in the next call |
| Below ~2 % weight change R behaves like a random draw (per-building SD 0.3–0.7; plateaus 0.02–0.2 % of a weight wide) | E5, E9; `verify.log` | No surrogate can rank at that scale; locally only many samples help (batch minimum) |
| Minor weights act as switches; tiny weights 1e-5 to 3e-3 change R; below 1e-6 a weight equals zero | E8, E10, E11 | The candidate generator must produce exact zeros and tiny weights |
| One-coordinate scans found 0.103 that random perturbations had missed | E10, E12 | Coordinate scans are a move type of their own |

## 2. Surrogate (T12, T13, T14)

1. **Separate two jobs.** A *navigation* model carries the search toward the low region and is judged by the
   direction of its extrapolation. A *local ranking* model picks points nearby and is judged by held-out MAE
   and Spearman. T35 found that a trend term hurts local ranking; navigation may need exactly that term. They
   need not be the same model, and model selection should score each job separately.
2. **The training table supports a trend** (`train_priors.log`, Spearman with R, 3-building mean):
   ramping share −0.64 (more is better), minor-weight sum +0.30 (less is better), r = degradation /
   (degradation + ramping) +0.51 (lower is better; the 50 lowest rows have median r 0.16 / 0.39 / 0.25).
   The trend is not monotone all the way: no training row has a building with only two objectives, so the
   face itself is outside the data. A constant-mean GP reverts to the mean away from the data (T35: about 105
   predicted as 124). Give the navigation model a per-building trend mean in these share features.
3. **Features** (inside the plugin's `fit` / `predict`, AGENTS.md section 4): per building r, ramping share and
   log(minor weight / (degradation + ramping)) with weights below 1e-6 floored to 0. Raw shares cannot tell
   1e-4 from 1e-2.
4. **Uncertainty:** treat the sub-2 % variation as an irreducible but exploitable spread of about 0.3–0.7 per
   building. More samples in a good region lower the batch minimum. T14 should include this layer rather than
   only target a coverage band.
5. **Not worth the effort:** richer models of the sub-2 % structure. In black-box data it behaves like noise.

## 3. Point-selection policy (T15, T16, T18)

1. **Two more moves in the candidate generator.** `bayes-pool` draws Dirichlet points with concentration
   offset 0.03, so it almost never produces exact zeros or systematic 1e-5 … 1e-2 weights. Add: set one minor
   weight to zero; add one minor weight at a log-uniform 1e-5 … 1e-2; a log-grid scan of one weight.
2. **Block design per call.** Of 50 points, about 3 × 15 change one building each (the other two at the
   incumbent), and a few combine the buildings' best changes from the previous call. Additivity makes the
   combination almost exactly predictable, so each call advances three 7-dimensional problems at once.
3. **Phases.** Early calls navigate (model trend, coarse steps, trust radius 1–2 spacings toward the boundary).
   Middle calls do per-building block descent. Late calls exploit: 0.1–2 % perturbations and coordinate scans
   around the best point, as many distinct points per call as possible, aimed at the batch minimum.
4. **Trust-region rule (T16).** Below ~2 % "grow on success, shrink on failure" reacts to near-random signals.
   Do not shrink below that scale.

## 4. Pre-submit check (T08)

- Two more duplicate rules: scaling one building's 7 weights by a constant leaves R unchanged (already in
  `docs/solver-internals.md`), and weights below 1e-6 are the same as 0. Points equal under either rule are
  repeats and waste the budget.
- Send full precision. Rounding the best local point to 4 decimals raised R by 0.02, to 3 decimals by 0.14 (`verify.log`).

## 5. Comparing policies (T18, T19)

- Both replay truths (table lookup, semi-synthetic GBM) are smooth and have no plateaus or needles. They
  favour pure exploitation and cannot rank local search; T18 notes that the semi-synthetic arm does not separate
  any setting.
- Compare on the real evaluator instead (T19). With `pool.py`'s 10 parallel instances a full 20 × 50 run is
  about 1,000 evaluations, about 26 minutes at 0.65 evaluations per second; 3 policies × 3 seeds take about 4
  hours. This is what decision 27 allows; until the TA confirms it, label such results with that condition.
- Repeated tuning on one evaluator over-fits it, but it is the graded evaluator. The real risk is encoding the
  optimum's location in hyperparameters without noticing, hence the training-data rule in section 0.

## 6. What to expect

- The team's earlier local search, which did not start from the training rows, went below 115.68 at
  evaluation 388 and below 107 at evaluation 547 (`analysis/replay/evidence/eval_log.jsonl`).
- E6: all three buildings on degradation + ramping only at the best block's r gave about 107–108.
- So 107–110 within 1,000 points is a reasonable target; 104–105 needs the local moves of section 3 and many
  points. These figures come from local runs: use them to set targets, never inside the policy.
