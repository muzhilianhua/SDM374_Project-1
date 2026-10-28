"""Member B's recorded v3.2 scores must be reproducible from the saved predictions with sdm.frozen_v32."""
import importlib.util

import pytest

from sdm.paths import REPO_ROOT

spec = importlib.util.spec_from_file_location("rescore", REPO_ROOT / "analysis" / "surrogate_release" / "rescore.py")
rescore = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rescore)

EXPECTED = {
    "selection_11_groups": {
        "peer_low90_xgb": 74.71671187267329, "peer_low80_histgb": 74.4974831653557,
        "peer_low60_histgb": 74.21427429317455, "peer_low90_histgb": 73.72628543046746,
        "peer_tiered_histgb": 73.26484022071187, "original_xgb_raw": 73.03836435765433,
        "original_hist_gradient_boosting": 72.6545842829437, "original_xgb_weighted": 72.17501576085706,
        "original_extra_trees": 68.43628591094652, "original_kernel_ridge": 67.01088745263193,
        "original_quadratic_ridge": 66.19342344422833,
    },
    "full_data_configs": {
        "histgb_low60_full": 72.47331740957708, "histgb_low80_full": 72.34651197825697,
        "xgb_w20x3_full": 72.3372956198732,
    },
}


@pytest.mark.parametrize("source", sorted(EXPECTED))
def test_recorded_scores_reproduce(source):
    assert rescore.rescore(source) == pytest.approx(EXPECTED[source], rel=1e-12)
