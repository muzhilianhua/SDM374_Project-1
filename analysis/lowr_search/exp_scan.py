"""E3: per-building scan with the other two blocks at the incumbent.
  region part: r in [0.16, 0.36], each minor weight on with p=0.5 at 10^U(-3, -0.5) x (F5+F6)
  global part: sparse random blocks anywhere on the simplex (is there a second basin?)
Each row varies one building only, so R - R_incumbent is that block's effect directly."""
import json, sys, time
import numpy as np
from pool import SolverPool
from common import block

P = int(sys.argv[1]) if len(sys.argv) > 1 else 10
N_REG, N_GLOB = 220, 80
state = json.load(open('fms_state.json', encoding='utf-8'))
inc = np.array(state['inc']); inc_R = state['inc_R']
rng = np.random.default_rng(31)
X, meta = [], []
for b in range(3):
    for _ in range(N_REG):
        m = np.where(rng.random(5) < 0.5, 10 ** rng.uniform(-3, -0.5, 5), 0.0)
        W = inc.copy(); W[b] = block(rng.uniform(0.16, 0.36), m); X.append(W.ravel()); meta.append((b, 'region'))
    for _ in range(N_GLOB):
        s = rng.integers(1, 5); sup = rng.choice(7, size=s, replace=False)
        w = np.zeros(7); w[sup] = rng.dirichlet(np.ones(s))
        W = inc.copy(); W[b] = w / w.sum(); X.append(W.ravel()); meta.append((b, 'global'))
pool = SolverPool(n_workers=P)
t = time.time(); y = pool.evaluate(np.array(X), tag='E3_scan'); secs = time.time() - t
print(f'{pool.n_solver_evals} new evals in {secs:.0f}s -> {pool.n_solver_evals / max(secs, 1e-9):.2f} evals/s; incumbent {inc_R:.6f}')
X = np.array(X).reshape(-1, 3, 7); np.set_printoptions(precision=4, suppress=True, linewidth=200)
for b in range(3):
    for kind in ('region', 'global'):
        m = np.array([mb == b and k == kind for mb, k in meta])
        d = y[m] - inc_R; o = np.argsort(d)
        print(f'b{b+1} {kind:6s} n={m.sum()} delta quantiles (min/5%/25%/50%) {np.round(np.percentile(d, [0, 5, 25, 50]), 3)}')
        for i in o[:4]: print('     ', round(float(d[i]), 4), X[m][i, b])
