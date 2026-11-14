"""Input validation reconstructed from the embedded Python CLI; use ../run.py."""

import argparse
import json
import math
import multiprocessing
import os
import sys
from pathlib import Path

VERSION = "same8-20261021.1"

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"


def parse_weights(text: str) -> list[list[float]]:
    value = json.loads(text)
    if isinstance(value, dict):
        if set(value) != {"weights"}:
            raise ValueError("JSON object must have exactly one key: weights")
        value = value["weights"]

    if not isinstance(value, list):
        raise ValueError("Expected 21 numbers or a 3 x 7 matrix")

    if len(value) == 21:
        rows = [value[i:i + 7] for i in range(0, 21, 7)]
    elif len(value) == 3 and all(isinstance(row, list) and len(row) == 7 for row in value):
        rows = value
    else:
        raise ValueError("Expected 21 numbers or a 3 x 7 matrix")

    for row in rows:
        if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in row):
            raise ValueError("Weights must be finite, nonnegative numbers; booleans are invalid")
        if not any(x > 0 for x in row):
            raise ValueError("Each building row needs at least one positive weight")
    return rows
