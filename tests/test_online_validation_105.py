"""Expose analysis/online_validation_105/test_validation.py to the repository test suite.

The validation tests live next to the run so they can import ``run_online`` directly. This
wrapper puts that directory on ``sys.path`` and loads the module, so ``uv run pytest`` from
the repository root discovers ``ValidationTests`` without duplicating the file.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis" / "online_validation_105"
sys.path.insert(0, str(ANALYSIS_DIR))

_SPEC = importlib.util.spec_from_file_location("online_validation_105.test_validation",
                                               ANALYSIS_DIR / "test_validation.py")
test_validation = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(test_validation)

ValidationTests = test_validation.ValidationTests
