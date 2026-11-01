"""The selection CLI must run directly and export numerically valid simplex weights."""
import importlib.util
import subprocess
import sys

import numpy as np

from sdm.paths import REPO_ROOT
from sdm.simplex import check_weights, sample

SCRIPT = REPO_ROOT / "analysis/offline_submission/select_candidates.py"


def test_selection_script_starts_without_shadowing_standard_library():
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--proposals" in result.stdout


def test_export_rounding_preserves_the_product_simplex():
    spec = importlib.util.spec_from_file_location("submission_selection", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    points = sample(100, np.random.default_rng(2026))
    rounded = module.rounded_weights(points)
    check_weights(rounded)
    assert np.max(np.abs(rounded - points)) < 4e-8
