# Reproducible surrogate evaluation evidence

This directory preserves the saved outer-test predictions behind member B's 11-group v3.2 review
and the three selected full-data configurations. No training or hyperparameter search is performed.
It requires the `sdm.frozen_v32` implementation in PR #3 before the tests can run on main.

| Evidence | Content |
| --- | --- |
| `evidence/selection_11_groups/` | 11 groups, 110 outer folds, 45056 held-out predictions |
| `evidence/full_data_configs/` | 3 configurations, 30 outer folds, 12288 held-out predictions |
| [DATA_FREEZE.md](DATA_FREEZE.md) | Meaning and membership rules of the frozen partition |
| [SCORING_V3_2.md](SCORING_V3_2.md) | Approved score formulas and tolerance comparisons |
| [MODEL_SELECTION.md](MODEL_SELECTION.md) | Rankings, selected configurations and limitations |
| `rescore.py` | Rescore both saved datasets with the team's frozen-v32 implementation |

The shared input releases are [`../../frozen_data/`](../../frozen_data/) and
[`../../evaluation_v3_2/`](../../evaluation_v3_2/); no duplicate copies or model binaries are included.
After PR #3 is merged, use the team Python 3.11 environment:

```bash
uv sync --group gui
uv run python analysis/surrogate_release/rescore.py
uv run --group gui pytest -q
```

`tests/test_surrogate_release.py` locks all 14 recorded scores. Historical trimmed leaders scored
74.72, 74.50 and 74.21; removing trimming changed the full-data configurations to about 72.34?72.47.
The supplied full-config predictions come from separately fitted outer-training counterparts, not
from testing all-data weights on their own training rows. Their fitting history is provider-declared.
The reused folds are descriptive evidence, not a fresh independent final test or proof for new W.
Original provenance remains unchanged, including dirty-source flags and original environment paths.
