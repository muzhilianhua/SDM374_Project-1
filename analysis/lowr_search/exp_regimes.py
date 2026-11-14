"""E7: cross-regime factorial (5 x 5 x 5 = 125 points).

Inside the basin R is additive over buildings (E1, E4), but the single-building "battery idle" penalties
(41.9 + 11.6 + 27.5 = 81.0) exceed the joint one (68.2), so far from the basin the buildings interact.
Each building takes one of five controllers that differ a lot:

  inc    its incumbent block
  pure   degradation + ramping only at the incumbent's r (minor weights removed)
  ramp   ramping only
  half   degradation + ramping at r = 0.40 (battery partly switched off)
  idle   degradation only (battery never moves)

Question: how large are the interactions, what sign do they have, and is there any combination in which
a building that smooths *less* gives a lower district R than the additive expectation suggests?
"""
import itertools
import json
import time

import numpy as np

from common import block, to_ratio
from pool import SolverPool

LEVELS = ["inc", "pure", "ramp", "half", "idle"]
state = json.load(open("fms_state_run4.json", encoding="utf-8"))
inc = np.array(state["inc"]); inc = inc / inc.sum(1, keepdims=True); inc_R = state["inc_R"]
Z = np.zeros(5)


def level_block(b, name):
    r0, _ = to_ratio(inc[b])
    return {"inc": inc[b], "pure": block(r0, Z), "ramp": block(0.0, Z), "half": block(0.40, Z), "idle": block(1.0, Z)}[name]


combos = list(itertools.product(range(5), repeat=3))
X = np.array([np.concatenate([level_block(b, LEVELS[c[b]]) for b in range(3)]) for c in combos])
pool = SolverPool(n_workers=10)
t0 = time.time()
y = pool.evaluate(X, tag="E7_regime")
print(f"{pool.n_solver_evals} new evaluations in {time.time() - t0:.0f}s; incumbent {inc_R:.4f}")
T = np.full((5, 5, 5), np.nan)
for c, r in zip(combos, y):
    T[c] = r
D = T - T[0, 0, 0]
main = [D[:, 0, 0], D[0, :, 0], D[0, 0, :]]
print("\nsingle-building effects (others at the incumbent)")
print("        " + "".join(f"{l:>9s}" for l in LEVELS))
for b in range(3):
    print(f"  b{b + 1}    " + "".join(f"{v:+9.3f}" for v in main[b]))

add = main[0][:, None, None] + main[1][None, :, None] + main[2][None, None, :]
inter = D - add
print("\npairwise interaction = R - (incumbent + the two single-building effects), third building at the incumbent")
for (i, j) in [(0, 1), (0, 2), (1, 2)]:
    sl = [0, 0, 0]; sl[i] = slice(None); sl[j] = slice(None)
    M = inter[tuple(sl)]
    print(f"  b{i + 1} (rows) x b{j + 1} (columns)      " + "".join(f"{l:>9s}" for l in LEVELS))
    for a in range(5):
        print(f"     {LEVELS[a]:>5s}                    " + "".join(f"{v:+9.3f}" for v in M[a]))

print("\nall 125 combinations: interaction as a share of the additive penalty")
m = add > 1e-9
print(f"  interaction: min {inter.min():+.3f}, median {np.median(inter[m]):+.3f}, max {inter.max():+.3f};"
      f"  share of additive penalty: median {np.median(inter[m] / add[m]):+.3f}, min {np.min(inter[m] / add[m]):+.3f}, max {np.max(inter[m] / add[m]):+.3f}")
print(f"  combinations below the incumbent: {int((D < -1e-9).sum())}; positive interactions: {int((inter > 1e-6).sum())}, negative: {int((inter < -1e-6).sum())}")
order = np.argsort(D.ravel())[:8]
print("  lowest combinations:")
for o in order:
    c = np.unravel_index(o, D.shape)
    print(f"     {[LEVELS[k] for k in c]}  R - incumbent = {D[c]:+.3f}  (additive {add[c]:+.3f}, interaction {inter[c]:+.3f})")

print("\nvalue of one building's control, depending on what the other two do (penalty of switching it to idle)")
for b in range(3):
    others = [x for x in range(3) if x != b]
    rows = []
    for a in range(5):
        for c in range(5):
            idx_ctrl = [0, 0, 0]; idx_ctrl[others[0]] = a; idx_ctrl[others[1]] = c
            idx_idle = list(idx_ctrl); idx_idle[b] = 4
            rows.append((LEVELS[a], LEVELS[c], T[tuple(idx_idle)] - T[tuple(idx_ctrl)]))
    v = np.array([r[2] for r in rows])
    lo, hi = int(np.argmin(v)), int(np.argmax(v))
    print(f"  b{b + 1}: from {v.min():.2f} (others {rows[lo][0]}/{rows[lo][1]}) to {v.max():.2f} (others {rows[hi][0]}/{rows[hi][1]});"
          f" others at incumbent {rows[0][2]:.2f}, others idle {rows[-1][2]:.2f}")

json.dump({"levels": LEVELS, "incumbent_R": inc_R, "R": T.tolist()}, open("E7_regimes.json", "w"), indent=1)
