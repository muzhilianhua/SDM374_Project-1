# Immutable dataset partition

Protocol: `sdm374.same8.splits.2x5x3.frozen.v1`. This package fixes data and sample membership only;
it contains no model selection, scoring objective or model-training policy.

The source has 2048 unique `candidate_id` records, 21 raw weight features and one `ramping` target.
Feature order is building 1, 2, 3, each with cost, carbon, peak, discomfort, degradation, ramping,
resilience. Each building's seven nonnegative weights must sum to one. The minimum recorded target
is about 115.76 (`p01969`). A surrogate prediction is not a new measurement.

| Item | Frozen value |
| --- | --- |
| Data SHA256 | `877b5f8ee265245dd693148a5d7b11f28f48b426e43ac3ac1be8b7a934249023` |
| ZIP SHA256 | `91fbd29f05d64ff2470e8b8a15ade78b7f234b234bf32f03a67fb4ca5d15a09d` |
| Outer generator | RepeatedKFold, 5 folds, 2 repeats, seed 20261024 |
| Inner generator | KFold, 3 folds, shuffle; seed 20261024 + repeat_index*5 + fold_index + 1 |
| Ordering | Ascending lexical candidate_id; indices start at zero |
| Outer sizes | Train 1638/1639; test 410/409 |
| Inner sizes | Train 1092/1093; validation 546/547 |

Use the explicit ID lists in `split_manifest.json` as the authority. The generator documents the
origin and supports verification; do not regenerate or replace the membership. These are random,
count-balanced folds, **without target stratification**. Equal fold sizes do not guarantee identical
target distributions or identical counts below any target threshold.

For one outer fold, hold its test samples aside. Fit preprocessing and model parameters only on
the allowed training samples. When tuning, split that outer training subset into the three fixed
inner folds, select the configuration using their mean validation score, then refit on the outer
training subset and evaluate once on the full outer test subset. The inner folds never contain that
outer test set. Filtering or weighting training samples must never filter test samples.

Within each repeat, every record is tested exactly once; across two repeats, every record is tested
twice. There are 4096 prediction rows per model but only 2048 distinct records. The ten folds overlap
in training membership and do not represent ten independent datasets.

For this handoff, hyperparameters were already fixed, so no inner search was run. The delivered
deployment weights then fit all 2048 records. Those final weights have no untouched test subset
inside this dataset; evaluate their workflow using the separately trained outer-fold counterparts.

Extract the frozen ZIP and run `python split_api.py` for checksum and membership validation. The
helper is standard-library-only. `load_package` and `iter_nested_splits` supply ID-aligned indices;
inner indices are relative to the corresponding outer training list. Do not rewrite the original
CSV, change IDs or silently repair a checksum failure. A future data change requires a new release
identity; it must not overwrite this frozen package.
