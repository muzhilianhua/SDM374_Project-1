"""Shared helpers: ratio parameterisation of one building's 7 weights.

A block is written  w ∝ [m1, m2, m3, m4, r, 1-r, m7]  with r = F5/(F5+F6) and
m_k = w_k/(F5+F6) for the five minor objectives (cost, carbon, peak, discomfort,
resilience). The evaluator renormalises each building, so only ratios matter.
Also the file helpers every script uses (paths relative to this folder, files closed by `with`).
"""
import json
from pathlib import Path

import numpy as np

NAMES = ["cost", "carbon", "peak", "discomfort", "degradation", "ramping", "resilience"]
MINOR = [0, 1, 2, 3, 6]


def block(r, m):
    w = np.zeros(7)
    w[4], w[5] = r, 1.0 - r
    w[MINOR] = m
    return w / w.sum()


def to_ratio(w):
    w = np.asarray(w, float)
    s = max(w[4] + w[5], 1e-12)       # a block may put all its weight on minor objectives
    return w[4] / s, w[MINOR] / s


def incumbent(log_W, log_R):
    i = int(np.nanargmin(log_R))
    return log_W[i].reshape(3, 7).copy(), float(log_R[i])


def team_best():
    return np.array(read_json(REPLAY / "best_weights_phaseE5.json")["weights_by_building"])


# ---- files: every path is resolved against this folder, so scripts run from any working directory ----
HERE = Path(__file__).resolve().parent
REPLAY = HERE.parent / "replay" / "evidence"        # the team's 6,568-call log and its best weights


def here(name) -> Path:
    """A path inside this study folder (absolute paths pass through)."""
    return HERE / name


def read_json(name):
    with open(here(name), encoding="utf-8") as fh:
        return json.load(fh)


def read_jsonl(name) -> list:
    with open(here(name), encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def write_json(name, obj, indent=None) -> None:
    with open(here(name), "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent)
