"""E1: is R additive over buildings in the low region?  5 x 5 x 5 factorial of
per-building weight vectors taken from the team's log (R < 108), chosen by
farthest-point sampling so the variants differ. Doubles as a throughput test."""
import itertools, sys, time
import numpy as np
from pool import SolverPool, normalize
from common import HERE, read_jsonl, write_json

P = int(sys.argv[1]) if len(sys.argv) > 1 else 8
rows = read_jsonl('../replay/evidence/eval_log.jsonl')
W = normalize([r['weights'] for r in rows]).reshape(-1, 3, 7)
R = np.array([r['R'] for r in rows])
good = np.where(R < 108)[0]
good = good[np.argsort(R[good])]
variants = []
for b in range(3):
    cand = np.unique(np.round(W[good, b], 10), axis=0)
    chosen = [np.round(W[good[0], b], 10)]            # the best point's block first
    while len(chosen) < 5:
        d = np.min([np.abs(cand - c).sum(1) for c in chosen], axis=0)
        chosen.append(cand[np.argmax(d)])
    variants.append(np.array(chosen))
    print('building', b + 1); print(np.round(variants[-1], 4))
combos = list(itertools.product(range(5), repeat=3))
X = np.array([np.concatenate([variants[0][i], variants[1][j], variants[2][k]]) for i, j, k in combos])
pool = SolverPool(n_workers=P)
t = time.time(); y = pool.evaluate(X, tag='E1_factorial'); secs = time.time() - t
print(f'P={P}: {pool.n_solver_evals} new evals in {secs:.0f}s -> {pool.n_solver_evals / max(secs, 1e-9):.2f} evals/s')
T = y.reshape(5, 5, 5)
mu = T.mean(); a = T.mean((1, 2)) - mu; b_ = T.mean((0, 2)) - mu; c = T.mean((0, 1)) - mu
fit = mu + a[:, None, None] + b_[None, :, None] + c[None, None, :]
res = T - fit
print('R range', T.min().round(3), T.max().round(3), 'total SD', T.std().round(3))
print('main effects b1', a.round(3), 'b2', b_.round(3), 'b3', c.round(3))
print('additive R2', 1 - res.var() / T.var(), 'residual SD', res.std().round(4), 'max |res|', np.abs(res).max().round(3))
i, j, k = np.unravel_index(T.argmin(), T.shape); ia, ja, ka = a.argmin(), b_.argmin(), c.argmin()
print('true argmin', (i, j, k), T.min(), '| additive argmin', (ia, ja, ka), T[ia, ja, ka])
np.save(HERE / 'E1_factorial.npy', T); write_json('E1_variants.json', [v.tolist() for v in variants])
