"""E2: one-dimensional profiles around the incumbent, one coordinate at a time.
Minor weights on a log scale (including exact 0), r on a linear scale."""
import sys, time, json
import numpy as np
from pool import SolverPool
from common import block, to_ratio, team_best, NAMES, MINOR

P = int(sys.argv[1]) if len(sys.argv) > 1 else 10
base = team_best()
LEVELS = [0, 1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3]
RS = np.round(np.arange(0.10, 0.455, 0.01), 4)
X, meta = [], []
for b in range(3):
    r0, m0 = to_ratio(base[b])
    for j, k in enumerate(MINOR):
        for v in LEVELS:
            m = m0.copy(); m[j] = v
            W = base.copy(); W[b] = block(r0, m); X.append(W.ravel()); meta.append((b, NAMES[k], v))
    for r in RS:
        W = base.copy(); W[b] = block(float(r), m0); X.append(W.ravel()); meta.append((b, 'r', float(r)))
pool = SolverPool(n_workers=P)
t = time.time(); y = pool.evaluate(np.array(X), tag='E2_profiles'); secs = time.time() - t
print(f'P={P}: {pool.n_solver_evals} new evals in {secs:.0f}s -> {pool.n_solver_evals / max(secs, 1e-9):.2f} evals/s')
out = {}
for (b, name, v), r in zip(meta, y):
    out.setdefault(f'b{b+1}:{name}', []).append((v, round(float(r), 4)))
for k, v in out.items():
    b = int(k[1]) - 1; r0, m0 = to_ratio(base[b])
    cur = r0 if k.endswith(':r') else m0[[NAMES[i] for i in MINOR].index(k.split(':')[1])]
    best = min(v, key=lambda t: t[1])
    print(f'{k:16s} incumbent value {cur:.5f} | best {best} | ' + ' '.join(f'{a:g}:{c:.2f}' for a, c in v))
json.dump(out, open('E2_profiles.json', 'w'))
