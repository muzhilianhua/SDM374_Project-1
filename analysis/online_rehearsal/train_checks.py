"""Design checks that use the 2048 training rows only (no evaluator data).

1. Additivity over buildings: 5-fold CV of HistGradientBoosting with and without per-building interaction
   constraints (an additive model cannot use cross-building interactions).
2. Features: raw shares vs the policy's ratio features (r, log deg+ramp share, log minor / (deg+ramp)).
3. The constants TrainPriors derives (tiny-weight range, shift-target probabilities).

Output: train_checks.log
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_validate

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from policy import TrainPriors  # noqa: E402
from rehearse import load_training  # noqa: E402
from surrogate import features  # noqa: E402


def main():
    X, y = load_training()
    F = features(X)
    kf = KFold(5, shuffle=True, random_state=0)
    groups = [list(range(b * 7, b * 7 + 7)) for b in range(3)]
    lines = [f"2048 training rows, best R {y.min():.3f}", "",
             "5-fold CV (KFold shuffle, random_state 0), HistGradientBoosting max_iter 600, lr 0.05",
             f"{'model':10s} {'inputs':16s} {'R2':>7s} {'MAE':>7s}"]
    for name, kw in (("full", {}), ("additive", {"interaction_cst": groups})):
        for label, Z in (("raw shares", X), ("ratio features", F)):
            m = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, random_state=0, **kw)
            cv = cross_validate(m, Z, y, cv=kf, scoring=("r2", "neg_mean_absolute_error"))
            lines.append(f"{name:10s} {label:16s} {cv['test_r2'].mean():7.4f} "
                         f"{-cv['test_neg_mean_absolute_error'].mean():7.3f}")
    pr = TrainPriors.from_training(X, y)
    np.set_printoptions(precision=3, suppress=True)
    lines += ["", f"tiny-weight range (0.1 % and 10 % quantiles of weights >= 1e-6): "
                  f"{pr.tiny_lo:.3g} .. {pr.tiny_hi:.3g}",
              "shift-target probabilities (v2), rows b1..b3, columns cost carbon peak discomfort "
              "degradation ramping resilience:", str(pr.favour)]
    text = "\n".join(lines) + "\n"
    (HERE / "train_checks.log").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
