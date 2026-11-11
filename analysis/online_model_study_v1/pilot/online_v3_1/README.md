# Online V3.1 — planning notes

**Status:** protocol design in progress; no V3.1 solver experiment has run yet.

## Agreed model-comparison frame

- Compare four models: `local-gp`, `histgb`, `xgboost`, and `boosted-consensus`; Ridge2 is excluded from the next main comparison based on its prior results.
- Run 10 rounds per replicate. Every model sees the same candidate pool within a round, selects one point by minimum predicted mean, and receives only its own solver feedback. Duplicate selections are solved once.
- Local-GP fit time is reported separately and is not a model-quality penalty.
- The current discussion favors a more exploratory candidate generator, including multiple promising anchors and feasible steps beyond V3's 0.05 ceiling, to seek lower-R points. Exact pool composition and step schedule remain to be finalized.
- Because this changes candidate generation, V3's three replicates remain a historical baseline and should not be pooled with V3.1 results as if they were one unchanged protocol.

## V3 finding that informs V3.1: shortlist availability vs. selection

V3 had 30 pools × 300 candidates = 9,000 candidate appearances:

| Source | Candidate appearances | Share of pools | Model selections | Share of selections |
|---|---:|---:|---:|---:|
| Broad screened directions | 8,918 | 99.09% | 139 | 92.67% |
| Offline shortlist | 82 | 0.91% | 11 | 7.33% |

The shortlist was selected about **8× its availability share**. By model, Local-GP selected 9 shortlist points out of 30 selections, Boosted-consensus selected 2, and HistGB, XGBoost, and Ridge2 selected none. The absolute count 139 vs. 11 should not be read as broad directions being preferred, because broad candidates made up over 99% of the pool.

The lowest V3 result still came from a broad screened direction (XGBoost, R=113.444). A shortlist direction was also useful: Local-GP selected offline consensus direction 14840 at step 0.05 and solver measured R=114.211. These observations show shortlist attraction/utility for some model trajectories, but do not establish that shortlist directions have better true R: only model-selected points were solved, and just 11 shortlist selections were observed.

## V3.1 logging requirements

Record candidate counts and selection counts by source, anchor, and step; report availability-adjusted selection rates alongside solver outcomes. Keep candidate pools shared across models. If the pool source proportions change, interpret raw source selection counts against those proportions. Treat V3 results as context, not pooled replicates, once the V3.1 generator changes.

## References

- V3 full report and candidate-source audit: [`../online_v3_directional/REPORT.md`](../online_v3_directional/REPORT.md).
- V3 trajectories: [`../online_v3_directional/trajectories_all.csv`](../online_v3_directional/trajectories_all.csv).
- V3 candidate pools and per-pool manifests: [`../online_v3_directional/replicate_01/candidate_pools/`](../online_v3_directional/replicate_01/candidate_pools/), with equivalent folders for replicates 02–03.
