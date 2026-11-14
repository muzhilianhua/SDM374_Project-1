"""Priors for the online model and policy that follow from the 2048 training rows alone.

GUIDANCE.md recommends that every prior the online policy uses be derivable from the training table
(decisions 19, 27, 29). This script derives the ones it cites; it reads only `dataset_same8/train.csv`
and no evaluator data.

    uv run python analysis/lowr_search/train_priors.py      # from the repository root
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
df = pd.read_csv(ROOT / "dataset_same8" / "train.csv")
W = df.iloc[:, 1:22].to_numpy(float).reshape(-1, 3, 7)
y = df["ramping"].to_numpy(float)
W = W / W.sum(2, keepdims=True)

deg, ramp = W[:, :, 4], W[:, :, 5]
r = deg / np.maximum(deg + ramp, 1e-12)
minor = W[:, :, [0, 1, 2, 3, 6]].sum(2)
features = {"ramping share": ramp, "minor-weight sum": minor, "r = degradation / (degradation + ramping)": r,
            "degradation share": deg}

print(f"{len(y)} training rows, best R {y.min():.3f}\n")
print("Spearman correlation with R (negative: larger feature, lower R)")
print(f"  {'feature':44s} {'3-building mean':>16s}   b1      b2      b3")
for name, f in features.items():
    per = [spearmanr(f[:, b], y)[0] for b in range(3)]
    print(f"  {name:44s} {spearmanr(f.mean(1), y)[0]:+16.3f}   " + "  ".join(f"{v:+.3f}" for v in per))

top = np.argsort(y)[:50]
print("\nmedians, 50 lowest rows vs all rows (b1, b2, b3)")
for name, f in features.items():
    print(f"  {name:44s} {np.round(np.median(f[top], 0), 2)}  vs  {np.round(np.median(f, 0), 2)}")

print("\nsupport of the training rows")
print(f"  share of weights below 0.01: {(W < 0.01).mean():.3f}; below 1e-6: {(W < 1e-6).mean():.4f}")
print(f"  smallest minor-weight sum per building: {np.round(minor.min(0), 3)}; largest ramping share: {np.round(ramp.max(0), 2)}")
print(f"  rows in which some building uses at most two objectives (> 1e-6): {int(((W > 1e-6).sum(2) <= 2).any(1).sum())}")
