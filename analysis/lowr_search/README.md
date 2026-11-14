# How low can R go? Black-box search and mechanism probes on the local evaluator

> **Exploration only. Not an input to the online policy.** This folder queries a local copy of
> `solver.exe` without a call limit to map the R landscape and find the lowest reachable point. Its
> points, R values and weights must not enter the submitted online code, its configs, warm starts or
> exclusion lists: before its first call the online code uses only the 2048 training rows (decisions 19,
> 29). Decision 27 (the user's judgment of 2026-11-11, pending the TA) allows local evaluator runs only to
> choose models, policies and hyperparameters.
> The study resets `usage.count` only inside its own `workers/` hard-link copies, never in `solver/`.
> All queries are black-box: the embedded evaluator program was not unpacked or run outside the wrapper.

Owner: 张子木 (A). Run on 2026-11-12/13 on one Windows 11 PC (i5-14600KF), `solver.exe` sha256
`293867b8…a077247`. 8,763 evaluations in `eval_log.jsonl`, on top of the team's 6,568-call log
(`analysis/replay/evidence/eval_log.jsonl`).

## Result

**Lowest point found: R = 104.3726098362822**, 0.479 below the team log's best (104.8517), against the
training-table best 115.765 and offline submission 2 at 115.68.

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

- One evaluator version, one machine. "One basin" rests on ~3,000 global samples, not a proof.
- Operator hit rates are not a controlled comparison.
- The explanations in section 2 (battery reserve, near-ties, switch-off thresholds) are consistent with the
  black-box data; no hourly trajectory was observed.
- E8's dose curves were measured at the best block's r; E9 shows that their windows depend on r.
