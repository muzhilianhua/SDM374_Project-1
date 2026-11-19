# v7 hybrid policy: a 20-call real online search

## 1. Goal and background

Three earlier experiments set the context (v4 = PR 25, h5 = 15-call accelerated schedule,
h6 = multiscale pool):

- v4 needed 35 calls to go from 115.764819 to 102.143322; after 15 calls it had only reached
  103.664157, i.e. 88.8% of the total drop;
- h6 replaced v4's model pool with a mixed pool and shrank the random quota, lost the 0.9938 `model`
  jump at call 12, and after 20 calls was only 0.052 ahead of v4;
- h5 moved the scans earlier and added more combinations; it stalled on a high sub-plateau at
  107.134.

The goal of v7 was therefore narrow: **keep calls 1-11 (which find the basin) untouched, keep every
productive v4 channel inside the basin, and replace only the slots that produced nothing**, to see
whether the same 20 calls / 1000 points can reach a lower R.

## 2. Policy

### 2.1 Two phases

| Phase | Calls | Proposals come from |
|---|---|---|
| Frozen (find the basin) | 1-11 | `BlockDescentPolicyV4(ParamsV4(shift_bias=True, n_nav=3), seed=7)`, point-identical to PR 25 |
| Hybrid (converge inside it) | 12-20 | the full 50-point v4 batch first, then multiscale candidates in the slots that yielded nothing |

The frozen phase is deliberate: h5 showed that moving scans earlier steals `model` slots and removes
the large structural jumps. v7 does not touch any basin-finding logic, so it still receives the 0.9938
`model` jump at call 12.

### 2.2 Replacement from call 12

1. Ask v4's `propose` for the full 50-point batch; `model / scan / icombo / random / tiny / base` are
   all kept as they are.
2. Free slots in the order `fine -> zero` (at most 12) and fill them with three kinds of multiscale
   candidate:
   - `coarse`: one coordinate scaled multiplicatively, step in `{0.45, 0.6, 0.75, 1.35, 1.7, 2.2, 3.0}`,
     to cross step-like plateaus;
   - `block`: 2-3 coordinates scaled together, to shift a whole structural block;
   - `restart`: convex combination with a low-R training block or with this call's elite block.
3. Each building first gets one `coarse`, one `block` and one `restart`; remaining slots take a second
   `coarse`.
4. Every replacement is de-duplicated against the training rows, all points already evaluated and the
   kept v4 points (after weight-grid quantisation). A duplicate is dropped and the v4 point is kept, so
   a replacement can never displace a validated point.
5. Step adaptation: after three consecutive calls without improvement for a building, its block sigma
   widens from 0.22 to 0.35 and snaps back on the next improvement.
6. All randomness derives from `(seed=7, call_no)`; no clock, no external log.

### 2.3 Design basis

In v4's calls 12-20 the new bests came from `model` (1), `scan` (3), `icombo` (3), `random` (2),
`tiny` (1) and re-tests of the incumbent (`base`, 3). `fine`, `zero` and `combo` produced none of the
13 new bests. v7 therefore replaces at most 12 of those zero-yield slots per call (80 in total over
nine calls) instead of rearranging the whole model pool as h6 did.

## 3. Fixed conditions

| Item | Value |
|---|---|
| Seed | 7 |
| Points per call / calls | 50 / 20 |
| Training table | 2048 x 23, sha256 `877b5f8ee265245dd693148a5d7b11f28f48b426e43ac3ac1be8b7a934249023` |
| Solver | sha256 `293867b81dbf0c6194f97a57115487b9a39239d0c7dadbd51d749a203a077247` |
| Environment | Python 3.11.18 / numpy 1.24.4 / scipy 1.10.1 / scikit-learn 1.2.2 / joblib 1.6.0 |
| Policy files | `policy.py`, `policy_v4.py`, `policy_v5.py`, `policy_v6.py`, `policy_v7.py`, `surrogate.py` (sha256 in `run_config.json`) |
| External history | not used for proposals (`history_used_for_policy = false`) |

Full reproduction steps are in [README.md](README.md).

## 4. Results

- **Final best R = 102.8058786104**, call 20, point 25, `icombo`, building 1.
- Total drop below the best training row (115.764819): **12.9589406488**.
- End of the frozen phase (call 11): 105.107716483; the hybrid phase added **2.3018378722**.
- Acceptance checks: R<=102.3 **False**, R<=102.15 **False**, R<=102.1433 **False**.

### 4.1 Per call

| Call | Phase | Batch minimum | Best so far | Improvement |
|---:|---|---:|---:|---:|
| 1 | frozen | 114.358691 | 114.358691 | 1.406128 |
| 2 | frozen | 111.711461 | 111.711461 | 2.647230 |
| 3 | frozen | 108.563154 | 108.563154 | 3.148307 |
| 4 | frozen | 107.942734 | 107.942734 | 0.620420 |
| 5 | frozen | 107.258304 | 107.258304 | 0.684431 |
| 6 | frozen | 107.079156 | 107.079156 | 0.179148 |
| 7 | frozen | 106.541927 | 106.541927 | 0.537229 |
| 8 | frozen | 106.208036 | 106.208036 | 0.333891 |
| 9 | frozen | 105.489656 | 105.489656 | 0.718380 |
| 10 | frozen | 105.397526 | 105.397526 | 0.092130 |
| 11 | frozen | 105.107716 | 105.107716 | 0.289810 |
| 12 | hybrid | 104.113936 | 104.113936 | 0.993780 |
| 13 | hybrid | 103.881007 | 103.881007 | 0.232929 |
| 14 | hybrid | 103.836817 | 103.836817 | 0.044190 |
| 15 | hybrid | 103.664157 | 103.664157 | 0.172660 |
| 16 | hybrid | 103.367455 | 103.367455 | 0.296703 |
| 17 | hybrid | 103.242708 | 103.242708 | 0.124746 |
| 18 | hybrid | 103.242708 | 103.242708 | 0 |
| 19 | hybrid | 102.960023 | 102.960023 | 0.282686 |
| 20 | hybrid | 102.805879 | **102.805879** | 0.154144 |

### 4.2 Structure of the best point (3 x 7 normalised weights)

| Building | cost | carbon | peak | discomfort | degradation | ramping | resilience |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.257266 | 0.000006 | 0.161925 | 0.002024 | 0.074694 | 0.464591 | 0.039494 |
| 2 | 0.071205 | 0.076885 | 0 | 0 | 0.163062 | 0.583746 | 0.105103 |
| 3 | 0.144909 | 0.000044 | 0.253156 | 0.000394 | 0.187344 | 0.412846 | 0.001307 |

All three buildings are dominated by `ramping`. h5, by contrast, stalled at 107.134 with more cost in
buildings 2 and 3 (0.266 and 0.240) and almost no resilience there (0.0000 and 0.0026).

### 4.3 Figures

| Figure | Content |
|---|---|
| [01_R_curve.png](viz_explore_v7/01_R_curve.png) | v7 / v4 / h6 convergence, frozen and hybrid phases marked |
| [02_learning_curve.png](viz_explore_v7/02_learning_curve.png) | all 1000 real evaluations |
| [03_step_evolution.png](viz_explore_v7/03_step_evolution.png) | channel mix per call and the sigma adaptation |
| [04_structure.png](viz_explore_v7/04_structure.png) | weight structure of the incumbent over calls |

## 5. Comparison

### 5.1 Same budget (20 calls / 1000 points, seed 7)

| Policy | Best R | Difference to v4 |
|---|---:|---:|
| v4 (PR 25) | 103.275295 | - |
| h6 multiscale | 103.222939 | -0.052356 |
| **v7 hybrid** | **102.805879** | **-0.469416** |

### 5.2 Against v4 with a longer budget

These numbers come from the committed logs `runs/pr25_repro` (calls 1-20) and `runs/from_call20`
(calls 21-35); see [runs/PROVENANCE.md](runs/PROVENANCE.md).

| v4 call | Best R |
|---:|---:|
| 20 | 103.275295 |
| 21 | 102.516515 |
| 22 | 102.280570 |
| 35 | 102.143322 |

v7's 20-call result lies between v4's 20-call and 21-call results, so in R terms it saves less than
one call. It does not meet the 15-call goal: at call 15 v7 is still 103.664157, exactly v4's value,
because the frozen phase is shared.

### 5.3 Where the difference comes from

This is the part the first review asked to correct. Counting each point that lowered the running best
inside calls 12-20:

| Channel | v7 new bests | Improvement in R |
|---|---:|---:|
| model (v4 channel) | 1 | 0.984362 |
| scan (v4 channel) | 3 | 0.656711 |
| coarse (new) | 1 | 0.282686 |
| icombo (v4 channel) | 2 | 0.095057 |
| tiny (v4 channel) | 2 | 0.019577 |
| base (re-test of the incumbent) | 5 | 0.263443 |
| block (new) | 0 | 0 |
| restart (new) | 0 | 0 |

Reading of the -0.469416 against v4:

1. **Keeping v4's batch contributes exactly 0 relative to v4.** Calls 1-15 have the same batches and
   the same best values (103.664157 at call 15). This change only explains why v7 beats h6: h6 lost
   the 0.9938 `model` jump at call 12 and never recovered it.
2. The lead appears at call 16: v7 reaches 103.367455 where v4 reaches 103.539411, a gap of 0.171956.
   The cause is indirect - the replacement points changed the set of evaluated points, which changed
   the incumbent and the history that later `model` and `scan` proposals are built from.
3. The only direct contribution of a new channel is the `coarse` hit at call 19 (-0.282686,
   building 1). Together with the indirect divergence that accounts for the 0.469416:
   0.282686 direct + 0.186730 accumulated from call 16 onward.
4. `block` and `restart` produced no new best in 45 candidates (26 block, 19 restart).
5. The sigma adaptation fired only for building 3, which produced no new best in that phase, so the
   widening targeted the wrong building.

## 6. Audit

All ten checks in `runs/explore_v7_s7/audit_report.json` pass:

| Check | Result | Point |
|---|---|---|
| 1_training_source | PASS | 2048 rows, sha256 matches the manifest |
| 2_policy_no_external_answers | PASS | policy source contains no external answer or log read |
| 3_runtime_file_access | PASS | no forbidden access during proposal replay |
| 4_proposal_replay | PASS | every call replayed bit-identically when it ran |
| 5_no_reused_points | PASS | all 1000 points unique |
| 6_result_attribution_order | PASS | every R belongs to its call/seq |
| 7_seed_config_fixed | PASS | seed, call count and batch size fixed, no clock |
| 8_environment_rebuildable | PASS | Python and packages match `PINNED` |
| 9_freeze_and_solver | PASS | calls 1-11 match PR 25 as a multiset, solver chain check reused |
| 10_budget | PASS | one fresh copy, 20/20 calls |

`verify_v7.py replay` is stronger than check 4: it recomputes calls 12-20 from this log instead of
reading the flag written during the run.

## 7. Conclusions and limitations

**Conclusion (seed 7 only).** On the seed used during design, v7 moves the 20-call best from
103.275295 to 102.805879. The gain does not come from keeping the v4 batch - that part is worth zero
against v4 - but from one `coarse` hit at call 19 and from the indirect divergence that starts at call
16. The three experiments together (h5, h6, v7) suggest that this response surface falls mainly
through occasional whole-block hits; larger steps and finer search only collect small amounts.

**Limitations.**

- One seed, and it is the seed the policy was designed on. Seeds 2026 and 11 have not been run, so no
  expected gain over seeds can be quoted (decision 32).
- 20 calls / 1000 points is a hard budget; neither 102.3 nor 102.1433 is reached.
- `block` and `restart` produced about 180 candidates and no new best; the block-level construction
  does not hit the step-shaped notches yet.
- Step adaptation acts per building and fired on a building that was not the source of improvement.
- The evaluator is step-shaped: equal R values need not mean identical weight structure.

**Next direction.** Raise the hit rate of whole-block candidates (for example select structurally
different training blocks by support-set signature instead of perturbing the current incumbent),
rather than tuning step sizes further.
