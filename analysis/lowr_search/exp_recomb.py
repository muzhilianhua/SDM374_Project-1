"""E4: interaction lottery. Cross product of the K best distinct blocks per building
(by delta in the search archive). Under exact additivity no combination could beat the
incumbent; the residual of an additive fit measures the fine-scale interaction."""
import itertools, sys, time
import numpy as np
from pool import SolverPool
from common import HERE, read_json

K = int(sys.argv[1]) if len(sys.argv) > 1 else 7
state = read_json('fms_state.json')
inc_R = state['inc_R']
sets, deltas = [], []
for b in range(3):
    ent = sorted(state['arch'][b], key=lambda e: e['delta'])
    chosen, seen = [], []
    for e in ent:
        if e['delta'] < -1e-9 or any(abs(e['delta'] - d) < 1e-7 for d in seen):   # one block per plateau
            continue
        chosen.append(np.array(e['w'])); seen.append(e['delta'])
        if len(chosen) == K: break
    sets.append(chosen); deltas.append(seen)
    print(f'b{b+1} deltas of the chosen blocks:', np.round(seen, 4))
X = np.array([np.concatenate(c) for c in itertools.product(*sets)])
pool = SolverPool(n_workers=10)
t = time.time(); y = pool.evaluate(X, tag='E4_recomb'); print(f'{pool.n_solver_evals} new evals in {time.time() - t:.0f}s')
T = y.reshape(K, K, K)
mu = T.mean(); a = T.mean((1, 2)) - mu; b_ = T.mean((0, 2)) - mu; c = T.mean((0, 1)) - mu
res = T - (mu + a[:, None, None] + b_[None, :, None] + c[None, None, :])
print(f'incumbent {inc_R:.6f}; cross product min {T.min():.6f} at {np.unravel_index(T.argmin(), T.shape)} (gain {inc_R - T.min():+.4f})')
print('additive R2', round(1 - res.var() / T.var(), 4), 'residual SD', round(res.std(), 4), 'min residual', round(res.min(), 4))
pred = inc_R + np.add.outer(np.add.outer(deltas[0], deltas[1]), deltas[2])     # additive prediction from single-block deltas
err = T - pred
print('error of the single-block additive prediction: mean', round(err.mean(), 4), 'SD', round(err.std(), 4), 'min', round(err.min(), 4))
print('combinations below the incumbent:', int((T < inc_R - 1e-9).sum()), 'of', T.size)
np.save(HERE / 'E4_recomb.npy', T)
