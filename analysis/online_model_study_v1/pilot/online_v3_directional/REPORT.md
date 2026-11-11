# Online V3 directional-roaming comparison

## Purpose and protocol

V3 changes only the shared candidate-generation method from V2, following the user's offline-roaming reference. The five models, independent own-feedback trajectories, minimum-predicted-mean selection rule, 3 seeds, 10 rounds, and 300-point pool size are retained.

- Anchor: `p01969`, the best row in the initial 2,048-row dataset.
- Direction bank: regenerate the offline-final study's 20,000 unit directions using its seed 2026 and construction (Gaussian vectors, mean removed within each building, L2 normalized). Remove directions with cosine >=0.5 to either of the two known failed directions. This leaves 19,483 geometrically eligible directions; 19,446 can take the maximum online step 0.05 without leaving the simplex.
- Offline preference: candidate groups draw from the union of direction IDs in offline `scan_top.csv` and eligible `finalists.csv` (17 directions, including consensus direction 14840 and submitted direction 15713). This shortlist is mixed with the wider screened direction bank; it is a prior preference, not assumed ground truth.
- Roaming steps: 0.01, 0.02, 0.03, 0.04, 0.05. Each pool has exactly 60 candidates at each step.
- Exclusions: initial training rows, all points already in the solver database, duplicate points in other V3 pools, two known failed-direction cones, and the previously submitted point `p01969 + 0.05*u[15713]` (reported external R=115.68; excluded from re-query).
- Candidate pool seeds: 5301, 5302, 5303. All five models share each replicate/round pool and receive only their own selected-point feedback.

## Results

| Model | Mean final best R ± SD (3 replicates) | Best across replicates | Replicates reaching R ≤ 115.68 | Feedback points R ≤ 115.68 | Mean fit time per round |
|---|---:|---:|---:|---:|---:|
| HistGB | **113.915 ± 0.090** | 113.812 | 3/3 | 15 | 4.04 s |
| ridge2 | 114.163 ± 0.350 | 113.949 | 3/3 | 17 | **0.29 s** |
| local-GP | 114.238 ± 0.464 | 113.788 | 3/3 | 14 | 47.98 s |
| XGBoost | 114.290 ± 0.827 | **113.444** | 3/3 | 13 | 2.09 s |
| boosted-consensus | 114.434 ± 0.511 | 113.982 | 3/3 | 9 | 6.11 s |

HistGB had the lowest mean final best R and the smallest between-seed variation. XGBoost produced the single lowest R in V3 (113.444), but was less consistent across the three seeds. ridge2, which failed to improve in V2, now reached the target in all three directional-pool runs. This is strong evidence that model rankings depend on candidate generation. In V2 the average-best ranking favored local-GP; in V3 HistGB led and ridge2 was competitive.

The offline-consensus direction 14840 at step 0.05 was selected by local-GP in replicate 1 and measured R=114.211. This is a useful real-R check of one offline finalist, while the 17-direction shortlist as a whole still needs more solver coverage.

V3's comparison among the five models is paired and fair because they saw the same pool within each round. V2-to-V3 comparisons are descriptive only: candidate generation changed from a global-plus-multiscale-Dirichlet pool to a directional pool concentrated around `p01969`. These results do not establish that one model is intrinsically best under every candidate strategy.

## Candidate-source availability audit

The 30 shared pools contained 9,000 candidate appearances: 8,918 broad screened directions (99.09%) and 82 offline-shortlist candidates (0.91%). Across 150 model selections, 139 came from the broad set (92.67%) and 11 from the shortlist (7.33%). Thus shortlist candidates were selected about 8 times as often as their pool share would suggest; raw selection counts alone (139 vs. 11) hide this availability difference.

Selection by model: local-GP chose 9 shortlist candidates out of 30 selections, boosted-consensus chose 2, while HistGB, XGBoost, and ridge2 chose none. The shortlist-selected points had a minimum observed R of 114.211 (6/11 selected events at R <= 115.68); broad-direction selections included the V3 minimum R=113.444 (62/139 events at R <= 115.68).

This is evidence of model-specific preference for shortlist candidates, especially for local-GP, not proof that shortlist directions have better true R. Only selected points were solver-scored, the shortlist sample is small, and directions/steps were not evaluated in a balanced randomized design. V3.1 should track both candidate availability and selection rates by source, and allocate enough candidates to each source for meaningful solver evidence.
## Cost and saved artifacts

- 150 model selections collapsed to 140 unique solver evaluations; 30 solver calls took 1,281.8 seconds (21.4 minutes).
- The canonical solver database now includes all 140 V3 weights/results, with no duplicates or overlap with the initial 2,048 rows.
- `model_summary.csv`: aggregate comparison; `replicate_comparison.csv`: per-seed results; `trajectories_all.csv`: labeled step-by-step model feedback; each replicate folder stores all 30 shared pools, solver call records, and resumable checkpoints.
- Candidate source and generation metadata are in each `replicate_*/candidate_pools/manifest.json`; the offline direction shortlist provenance is in `offline_direction_reference.json`.

## Limits

Three replicates are still a small sample. The V3 pool explores one anchor and short steps up to 0.05, so results answer which surrogate works best under this roaming setup. They do not test wider or dynamically re-anchored search. The offline calibration itself was not a guarantee of improvement; V3 obtains new solver labels for the selected weights and should be treated as the actual online evidence.

