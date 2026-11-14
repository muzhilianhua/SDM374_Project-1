"""E5: unbiased sample of each building's sweet-spot box around the incumbent block.
  r ~ U(r0 - 0.03, r0 + 0.03); active minor ratios x 10^U(-0.25, 0.25);
  inactive minors switched on with p = 0.25 at 10^U(-3, -1.5).
One building varies per row, so delta = R - R_incumbent is that block's effect. The sample says how
extreme the incumbent block is among the cells of its own neighbourhood, and gives new entrances."""
import sys, time
import numpy as np
from pool import SolverPool
from common import block, to_ratio
from common import read_json

N = int(sys.argv[1]) if len(sys.argv) > 1 else 250
state = read_json('fms_state.json')
inc = np.array(state['inc']); inc_R = state['inc_R']
rng = np.random.default_rng(55)
X, meta = [], []
for b in range(3):
    r0, m0 = to_ratio(inc[b])
    for _ in range(N):
        m = np.where(m0 > 0, m0 * 10 ** rng.uniform(-0.25, 0.25, 5), np.where(rng.random(5) < 0.25, 10 ** rng.uniform(-3, -1.5, 5), 0.0))
        W = inc.copy(); W[b] = block(r0 + rng.uniform(-0.03, 0.03), m); X.append(W.ravel()); meta.append(b)
pool = SolverPool(n_workers=10)
t = time.time(); y = pool.evaluate(np.array(X), tag='E5_box'); print(f'{pool.n_solver_evals} new evals in {time.time() - t:.0f}s; incumbent {inc_R:.6f}')
meta = np.array(meta); X = np.array(X).reshape(-1, 3, 7); np.set_printoptions(precision=4, suppress=True, linewidth=200)
for b in range(3):
    d = y[meta == b] - inc_R; o = np.argsort(d)
    print(f'b{b+1}: delta quantiles (min, 1%, 5%, 10%, 25%, 50%, 75%) {np.round(np.percentile(d, [0, 1, 5, 10, 25, 50, 75]), 3)}  below incumbent: {(d < -1e-9).sum()}  within 0.1: {(d < 0.1).sum()}  SD {d.std():.3f}')
    for i in o[:3]: print('     ', round(float(d[i]), 4), X[meta == b][i, b])
