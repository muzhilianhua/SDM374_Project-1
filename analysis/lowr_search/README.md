# How low can R go? Black-box search and mechanism probes on the local evaluator

> **Exploration only. Not an input to the online policy.** This folder queries a local copy of
> `solver.exe` without a call limit to map the R landscape and find the lowest reachable point. Its
> points, R values and weights must not enter the submitted online code, its configs, warm starts or
> exclusion lists: before its first call the online code uses only the 2048 training rows (decisions 19,
> 29). Decision 27 (the user's judgment of 2026-11-11, pending the TA) allows local evaluator runs only to
> choose models, policies and hyperparameters.
> The study resets `usage.count` only inside its own `workers/` hard-link copies, never in `solver/`.
> All queries are black-box: the embedded evaluator program was not unpacked or run outside the wrapper.

> E13–E22 (2026-11-15/16) also *read* the online rehearsal logs in `analysis/online_rehearsal/rehearsals/`
> to seed their searches. The direction is one-way: nothing from this folder flows back into the rehearsal
> policy, its configs or its runs.

Owner: 张子木 (A). Run on 2026-11-12/13 (E1–E12) and 2026-11-15/16 (E13–E22) on one Windows 11 PC
(i5-14600KF), `solver.exe` sha256 `293867b8…a077247`. E1–E12: 8,763 evaluations in `eval_log.jsonl`
(the original T36 evidence, unchanged); E13–E22: 13,788 evaluations in `eval_log_part2.jsonl`, through the
same black-box `pool.py` and outer `solver.exe`. Both come on top of the team's 6,568-call log
(`analysis/replay/evidence/eval_log.jsonl`).

## Result

**Lowest point found: R = 102.09589070483344** (E20, 2026-11-16, `best_point_E20.json`), 2.76 below the
team log's best (104.8517), against the training-table best 115.765 and offline submission 2 at 115.68.
Section 2b explains how it was reached; it replaced all three buildings' blocks of the first phase's best
point W* below.

| Building | cost | carbon | peak | discomfort | degradation | ramping | resilience |
|---|---|---|---|---|---|---|---|
| 1 | 0.31939 | 0.004401 | 0.000237 | 0.002421 | 0.09769 | 0.575861 | 0 |
| 2 | 0.03233 | 0.021296 | 0.880493 | 0.004158 | 0.019541 | 0.041859 | 0.000322 |
| 3 | 0.144667 | 0.000287 | 0.252734 | 0.000512 | 0.187032 | 0.412158 | 0.002610 |

Checks (`verify_e20.py`, results inside `best_point_E20.json`): two fresh uncached calls (alone, and in
the middle of a 3-set batch) reproduce it bit for bit; 5 or more decimals keep R, 4 give 102.2150, 3 give
102.7326.

**First phase (E1–E12): W\* with R = 104.3726098362822**, 0.479 below the team log's best. It stays in
`best_point.json` because E13–E15 use it as their baseline.

| Building | cost | carbon | peak | discomfort | degradation | ramping | resilience |
|---|---|---|---|---|---|---|---|
| 1 | 0.001229 | 0.059745 | 0 | 0.006142 | 0.249282 | 0.675060 | 0.008541 |
| 2 | 0 | 0.008513 | 0 | 0.000217 | 0.209794 | 0.651955 | 0.129521 |
| 3 | 0.000029 | 0 | 0.009883 | 0.000294 | 0.242174 | 0.725088 | 0.022532 |

Full precision in `best_point.json`. Checks (`verify.py` → `verify.log`, `verification.json`):

- Two fresh `solver.exe` calls (alone, and in the middle of a 3-set batch) reproduce the value bit for bit.
- Rounding the weights to 5 or more decimals keeps R; 4 decimals give 104.3932 (+0.021), 3 give 104.509.
- Multiplying every weight by 10^N(0, s): s = 1e-5 keeps R; s = 1e-4 (~0.02 %) keeps it in 83 % of draws;
  s = 1e-3 always lands on another plateau (median +0.03, worst +0.70); s = 1e-2 gives median +2.4.

How it was reached:

| Date | Step | Evaluations | R |
|---|---|---|---|
| 11-12 | FMS search (section 3) and diagnostics E1–E5 | 5,317 | 104.8517 → 104.4755 |
| 11-13 | Mechanism experiments E6–E9 (not a search) | 2,405 | unchanged, but 11 scan points of E8 lay lower |
| 11-13 | Fine scans and combinations of those points, E10 | 171 | → 104.3728 |
| 11-13 | Vanishing weights, E11 | 117 | → 104.3726 |
| 11-13 | Independent re-checks | 129 | — |
| 11-13 | One more round of coordinate scans from the best point, E12 | 624 | unchanged |

E6–E10 deltas are relative to the 11-12 best, 104.4755 (`fms_state_run4.json`).

## 1. Shape of the landscape (E1–E5)

| Experiment | Points | Finding |
|---|---|---|
| E1 `exp_additivity.py`: 5 × 5 × 5 factorial of distinct blocks per building | 125 | In the low region R is close to additive over buildings: R² 0.976, interaction residual SD 0.24 |
| E2 `exp_profiles.py`: one-coordinate profiles around the team best | 342 | Neighbouring r = F5/(F5+F6) steps of 0.01 differ by 0.5–1.0 (sawtooth); building 1's resilience weight has no effect on 0–0.3 |
| E3 `exp_scan.py`: per building 220 basin samples and 80 sparse samples over the whole simplex | 864 | No second basin: the best global block is still +3.59 / +1.00 / +3.84 (buildings 1/2/3) |
| E4 `exp_recomb.py`: 7 × 7 × 7 combinations of near-best blocks | 328 | At fine scale buildings are strictly additive (prediction error +0.033 ± 0.030); no combination beats the best |
| E5 `exp_box.py`: 250 uniform samples per building in the best block's neighbourhood | 750 | Median +1.93 / +1.00 / +1.28, SD 0.70 / 0.29 / 0.59; the best blocks sit below the 0.4 % quantile of their neighbourhood |

So: one basin (degradation + ramping dominate, r ≈ 0.24–0.27, other weights small or zero); R ≈ const +
g₁(W₁) + g₂(W₂) + g₃(W₃); inside the basin, moving all weights by ~2 % already looks like a random draw.
The low points are needles, not valley floors.

## 2. Mechanism experiments (E6–E12)

### E6 `exp_mech.py`: baselines and knock-outs (206 points)

| All three buildings use | R |
|---|---|
| degradation only (battery never moves) | 172.63 |
| cost / carbon / peak / discomfort only | 173.5–176.4 |
| ramping only | 117.80 |
| the 11-12 best without its minor weights (degradation + ramping only) | 107.86 |
| the 11-12 best | 104.48 |

On the two-objective face, r between 0.30 and 0.50 switches the battery off step by step; above about
0.5 R no longer depends on r. A two-objective reading of the controller formulas in
`docs/solver-internals.md` (branch `docs/solver-internals`) puts the thresholds between 1/3 and 1/2,
roughly as measured. 66 archived blocks whose *estimated* delta was negative were re-measured against
the best point: none was lower.

### E7 `exp_regimes.py`: is there a division of labour between buildings? (109 new points)

Each building takes one of five very different controllers (best block, no minor weights, ramping only,
r = 0.40, battery idle); all 125 combinations.

- None is below the best. Interactions (measured minus the sum of single-building effects) are negative in
  92 of 112 multi-building combinations and grow with distance from good control: −12.9 when all three
  idle (16 % of the additive sum).
- The buildings are complements: idling building 1 costs 41.9 when the others are at their best, 30.0
  when they idle too. A building that smooths *less* never helped.

### E9 `exp_rscan.py`: fine structure of r (993 points), figure `mech_rscan.png`

r from 0.15 to 0.50 in steps of 0.002, one building at a time, with and without the minor weights.

- The best r is not the edge of the switch-off band: it sits in a local dip 0.02–0.04 wide and about 1 deep
  inside the 0.20–0.30 basin, 0.05–0.13 before R starts to climb.
- Along r the residual around an 11-point running median has SD 0.07–0.28 and lag-1 correlation 0.4–0.7;
  the correlation length is about 0.004–0.01.
- Buildings 1 and 3's minor weights help only near their own r; building 2's help across 0.15–0.35.

### E8 `exp_dose.py`: what the minor weights do (1,097 points), figures `mech_dose.png`, `mech_pair.png`

m = weight / (degradation + ramping). One weight varied from the best block; one weight added to the
block without minor weights; and a 10 × 10 grid over each building's two most valuable minor weights.

- Every minor weight acts like a switch: no effect below a threshold, useful in one window, harmful above.
  Repeatable windows: building 1 carbon (m 0.06–0.07, narrow), building 2 resilience (~0.15), building 3
  peak (0.007–0.03) and resilience (0.01–0.02).
- One weight per building recovers 0.41 + 1.61 + 0.42 = 2.44 of the 3.53 that the minor weights are worth
  at the best point; the rest comes from narrow combinations (building 1's discomfort helps only with
  carbon at its current value).
- Pairs are complements or additive, never substitutes. Combinations do not transfer between buildings:
  resilience does nothing in building 1, peak does little in buildings 1 and 2.
- Large weights always hurt (m = 1: cost and carbon +7 to +34).
- **Tiny weights on unused objectives lower R**: building 1 cost at m 3e-4 to 2e-3 (best −0.082),
  building 2 discomfort 2.2e-4 → 2.6e-4 (−0.015).

### E10 `exp_followup.py`: following those points (171 points)

At this size a weight only decides hours where the main objectives nearly tie, so R moves in a few wide
steps. Building 1 cost: 0 below m = 2.6e-4, −0.018 up to 1.1e-3, −0.082 up to 1.7e-3, then −0.026,
−0.014, −0.011, and +0.44 above 2.4e-3. The per-building best moves add up exactly (−0.0976 predicted,
−0.0975 measured); a tiny cost weight in building 3 helps only after its discomfort weight was added.
Result: 104.3728.

### E11 `exp_tiny.py`: is a vanishing weight different from zero? (117 points)

Each of the 15 minor weights replaced by m = 0, 1e-13, 1e-10, 1e-8, 1e-6, 1e-5, 3e-5, 1e-4: for every weight
R at 1e-13 … 1e-6 equals R at 0 bit for bit. The first changes appear between 1e-6 and 1e-5. So tiny
weights do not break exact ties (no lexicographic regime); they change choices in near-ties of order
1e-5 to 1e-3. Building 3 cost at 3e-5 gave the final −0.0002.

### E12 `exp_coord.py`: one more round from the best point (624 points)

All 18 coordinates (5 minor weights per building on 0 and a 22-point log grid 1e-6 … 3e-3 plus factors
0.6–1.6 of the current value; r ± 0.01 in steps of 0.0005): nothing lower. 16 coordinates at best tie;
building 2's r and resilience and building 3's resilience only get worse. **The best point is a
coordinate-wise local minimum at this resolution.**

## 2b. Other valleys, and crossing with the online rehearsal (E13–E22, 2026-11-15/16)

W* is a coordinate-wise local minimum, but not the only low region. The rehearsal runs of
`analysis/online_rehearsal/` stop in places that the first phase never sampled. Their blocks, refined,
reach lower valleys. All deltas are relative to the baseline named in each step.

| Step | Evaluations | R |
|---|---|---|
| E13–E15: transplant rehearsal blocks into W*, dose scans, per-building descent | 3,653 | 104.3726 → 103.8239 |
| E16–E17: retune buildings 1 and 2 at the new point, combine | 1,017 | → 103.2473 |
| E18: systematic per-region discovery around that point | 4,142 | unchanged |
| E19: cross with the v4 seed 7 online best point, refine | 1,223 | → 102.6609 |
| E20: cross with the other 9 rehearsal runs, refine | 2,818 | → **102.0959** |
| E21: shape of building 2's new valley | 607 | unchanged |
| E22: transplant that form into buildings 1 and 3 | 328 | unchanged |

### E13–E15 `exp_alt.py`, `exp_alt_dose.py`, `exp_alt_descent.py`: a second valley for building 3

- **E13 (29 points).** For each building and each "regime" (the objectives that hold at least 10 % of a
  block), the best block among the 60 best points of the 7 v1–v3 rehearsal runs was put into W*, with
  the other two buildings unchanged. An unrefined building-3 block with discomfort 0.90 was only +0.057
  above W*'s refined block. Random blocks inside W*'s basin are +0.40 to +0.93 (E3).
- **E14 (106 points).** The dominant weight was scaled by 0 to 100. Discomfort and peak blocks are
  insensitive from about 0.03× to 100×: once large enough, the objective acts like a constraint.
  Ramping-led blocks belong to W*'s family. Additivity holds only near W*: the same block measured in its
  rehearsal context differs by 0.4–1.1.
- **E15 (3,518 points).** Per-building coordinate descent of 7 seeds (others at W*), up to 6 rounds.
  Three different building-3 seeds fell below W*: −0.55, −0.50 and −0.31. The best block is cost 0.153,
  peak 0.154, discomfort 0.091, degradation 0.185, ramping 0.416 (r = 0.31), whereas W*'s block is almost
  pure degradation + ramping. **R = 103.8239** (`best_point_E15.json`). This refutes E3's "one basin":
  E3 compared unrefined global samples with a refined W*.

### E16–E17 `exp_retune.py`, `exp_retune2.py`: buildings interact across valleys

- **E16 (802 points).** A building-1 cost + ramping block (r ≈ 0.13) was +0.28 in W*'s context but
  −0.57 once building 3 sat in its new valley. **Additivity between buildings (error 0.03, section 1)
  holds only inside one valley; across valleys the interaction reaches about 0.85.**
- **E17 (215 points).** Measured combinations of the best blocks, then one more scan per building:
  **R = 103.2473** (`best_point_E17.json`), a coordinate-wise local minimum.

### E18 `exp_phase2.py`: systematic discovery (4,142 points)

500 sparse or one-dominant blocks per building around the E17 point, grouped into 87–101 regions per
building; the 5 best other regions per building refined for 2 rounds; 27 combinations measured. Nothing
went below 103.2473: the closest candidates were +0.21 (building 2) and +0.48 (building 1), still
descending when the budget ended. The conclusion is "no deeper valley that is easy to reach at this
budget", not "no other valley".

### E19 `exp_e19.py`: crossing with the v4 seed 7 online best point (1,223 points)

The v4 seed 7 rehearsal (compliant: training rows and its own 1,000 points only) ended at S = 103.2753,
close to the E17 point B but with different blocks. The 2×2×2 per-building combinations of B and S were
measured first:

| Combination (buildings 1/2/3 from B or S) | R |
|---|---|
| **BBS** | **102.8700** |
| BSS | 103.0066 |
| SBS | 103.1606 |
| BBB (= B) | 103.2473 |
| SSS (= S, re-measured identically) | 103.2753 |
| BSB / SBB / SSB | 103.49 / 103.64 / 103.84 |

S's building-3 block (peak 0.25, discomfort ≈ 0) beats E15's (peak 0.15, discomfort 0.09) in every
context. Then block-coordinate cycles: building 3 −0.175, building 1 −0.034, and a third cycle with no
improvement. **R = 102.6609** (`best_point_E19.json`, verified by `verify_e19.py`).

### E20 `exp_e20.py`: crossing with the other 9 rehearsal runs (2,818 points)

1. Each run's online best point replaced every non-empty subset of the E19 point's buildings (63 points).
   Only three swaps improved, all of them building 2: v2_s11 −0.319, v4_s11 −0.224, v4_s2026 −0.095.
2. The six single-building swaps within +0.6 were descended for up to 3 rounds. v2_s11's building 2
   reached −0.528, and five building-2 seeds of different origin ended below the base.
3. Two block-coordinate cycles: **R = 102.0959** (`best_point_E20.json`, `verify_e20.py`), a
   coordinate-wise local minimum.

The improvement is entirely building 2 moving to a peak-dominated valley: peak 0.88, small degradation
and ramping (r ≈ 0.32). It used to be ramping 0.65 + resilience 0.13. The block came from v2_s11, the
second-worst rehearsal run (online best 107.49): **how good a run is overall says nothing about how good
its single-building blocks are.**

### E21 `exp_e21.py`, `exp_e21b.py`, `exp_e21c.py`: shape of building 2's valley (607 points)

Building 2 was written as the peak share p, the battery share q = (deg + ramp)/(1 − p), and
r = deg/(deg + ramp), with the other minor weights kept in the E20 point's proportions.

- **R does not depend on p for p = 0.78–0.97** (identical to all digits). p = 0.70 gives +0.31. Only the
  ratios among the non-peak weights matter. `exp_e21.py` was therefore stopped during its seed descent:
  its four seeds differed only in p. Its grid is in `E21_coarse_log.json`.
- On a 17 × 17 fine grid in (r, q), no point beats the E20 point. The low values lie in a thin diagonal
  trough along which ramping, relative to the minor weights, is nearly constant. **Ramping is the needle
  coordinate; degradation is much looser.**
- On a ramping × degradation scan (steps of 0.2 % and 2.5 %) nothing is lower. R is **terraced**: it is
  identical over runs of grid cells, then jumps. This is consistent with discrete battery switching.

### E22 `exp_e22.py`, `exp_e22b.py`: the peak-dominated form in buildings 1 and 3 (328 points)

The seeds were peak-dominated rehearsal blocks plus a grid (p = 0.88, r 0.02–0.45, q 0.4–0.95). Every
point was at least +3.05 worse; nothing was descended. Each building has its own valley form: building 1
uses cost + ramping, building 2 is peak-dominated, and building 3 uses cost + peak + ramping.

### What this adds

- The low region is a set of per-building valleys whose combination must itself be searched. Buildings
  are additive inside a valley, but not across valleys.
- Inside a valley the landscape is terraced needles. Further drops come from jumping to another valley,
  not from finer tuning.
- Compliant online runs are an efficient way to discover valleys: they visit combinations that 15,000
  unrestricted evaluations missed. Crossing their best blocks costs 8–63 points plus refinement.
- None of this is an input to the online policy (see the note at the top). For policy design under
  decision 27, the transferable lesson is a move type: per-building block swaps and combinations between
  good points.

## 3. The search strategy (FMS, `fms.py`)

Factorised multi-scale search: one building changes per candidate (the other two at the incumbent, so
R − R* is that block's effect); multiplicative moves on one weight or all weights at seven scales;
toggles, crossovers between elites, one elite per plateau; adaptive operator probabilities; optional
tree-surrogate pre-screen, integer-ratio "snap" moves and side funnels. Four runs (2,841 evaluations)
went 104.78 → 104.4755.

Rejected ideas: recombination between buildings (E4), integer ratios to trigger the tie-break rule (0/131
hits), surrogate pre-screening (0/180), side funnels (0/1,080), a division of labour between buildings
(E7), the best r as an edge (E9), moving one building's useful minor weights to another (E8), a
lexicographic regime of vanishing weights (E11).

The main lesson: FMS's random "toggle a weight on/off" move (86 tries, no hit) missed what one-coordinate
systematic scans found in ~720 evaluations (−0.103). The stall at 104.4755 was partly a blind spot of the
move set, not only the landscape. `fms.py` does not yet have such a deterministic scan step.

## 4. What this means for the project

- For the surrogate study (T35, `analysis/gp_assumptions/`): below R ≈ 106 the landscape is a set of
  needles on plateaus 0.02–0.2 % of a weight wide, and the useful structure includes weights of order 1e-4.
  No smooth surrogate fitted to 2048 rows can resolve that; reaching ~104 is a local-search question.
- For the online policy (T15, T18), under decision 27 only as a design consideration, never as data:
  coordinate scans of small weights are a cheap move type that random perturbations miss.
- Larger drops would need hour-by-hour knowledge of the controller (which hours nearly tie, battery
  trajectories), i.e. a white-box replay of the evaluator. That was not done; it needs the user's or the
  TA's decision first.

Concrete recommendations for the surrogate, the batch policy, the pre-submit check and the policy
comparison are in [GUIDANCE.md](GUIDANCE.md); `train_priors.py` derives the priors it cites from the
training table alone.

## Files and reproduction

| File | Content |
|---|---|
| `pool.py` | Parallel, cached, append-only evaluator access (10 hard-linked `solver.exe` copies under `workers/`) |
| `qos.py` | Keeps the pool's own solver processes off Windows' E-core-only background scheduling (2.5× slower otherwise on this hybrid CPU); changes no system setting |
| `common.py`, `fms.py` | Block parameterisation and the FMS search (state in `fms_state.json`, git-ignored; a file `STOP` ends a run) |
| `exp_*.py`, `an_*.py` | Experiments E1–E12 and their analyses; logs `*.log`, results `E*_*.json`, `E*.npy` |
| `exp_alt*.py`, `exp_retune*.py`, `exp_phase2.py`, `exp_e19.py` … `exp_e22b.py` | Experiments E13–E22 (section 2b); logs `E1[3-9]*.log`, `E2*.log`, results `E1[3-9]_*.json`, `E2*_*.json` |
| `best_point_E15.json`, `_E17`, `_E19`, `_E20` | Best point after each step of section 2b; `verify_e19.py`, `verify_e20.py` re-check the last two |
| `eval_log_part2.jsonl` | Every E13–E22 evaluation, same format as `eval_log.jsonl`. To re-run those experiments from the cache, append it to `eval_log.jsonl` in a working copy (the pool reads only `eval_log.jsonl`) |
| `fms_progress.jsonl`, `fms_run*.log` | Per-generation records of the four FMS runs |
| `fms_state_run4.json` | Search state after run 4 (best 104.4755), the baseline of E6–E10 |
| `eval_log.jsonl` | Every evaluation of this study (tag, weights, R, worker, call time) |
| `verify.py`, `report.py` | Independent re-check of the best point; summary, `best_point.json`, `convergence.png` |
| `provenance.json` | Hashes, versions and machine |

```powershell
cd analysis/lowr_search          # every script expects this working directory
$env:PYTHONUTF8 = 1
uv run python exp_mech.py        # any experiment; points already in eval_log.jsonl are not re-evaluated
uv run python fms.py --fresh --minutes 45 --surrogate 0 --niche 12 --run-name runN
uv run python verify.py; uv run python report.py
uv run python an_mech.py --plot  # E8/E9 tables and the mech_*.png figures
```

`solver.exe` must be in `solver/` (decision 25). E3–E5 read `fms_state.json` as it was when they ran
(states after runs 1 and 3, not committed); re-running them now uses the current state and will evaluate
new points. `t_fms_dry.py` runs the search loop on a synthetic function without calling the solver.

## Limits

- One evaluator version, one machine. The first phase's "one basin" (E3) was refuted by E15.
- E13–E22 were developed outside the repository (`Project-1/lowR_research`) and copied here; only the
  paths to the rehearsal logs changed (`provenance.json`, `part2`).
- Operator hit rates are not a controlled comparison.
- The explanations in section 2 (battery reserve, near-ties, switch-off thresholds) are consistent with the
  black-box data; no hourly trajectory was observed.
- E8's dose curves were measured at the best block's r; E9 shows that their windows depend on r.
