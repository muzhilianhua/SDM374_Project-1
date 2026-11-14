"""Dry run of the search loop on a synthetic piecewise-constant additive function (no solver)."""
import json, tempfile, pathlib, numpy as np
import fms, pool as poolmod
tmp = pathlib.Path(tempfile.mkdtemp())
fms.STATE = tmp / 's.json'; fms.PROGRESS = tmp / 'p.jsonl'

class Fake:
    n_solver_evals = 0
    def __init__(self): self.cache = {}
    def evaluate(self, W, tag=''):
        W = poolmod.normalize(W).reshape(-1, 3, 7); out = []
        for w in W:
            k = w.tobytes()
            if k not in self.cache:
                self.n_solver_evals += 1
                g = sum(np.floor(40 * abs(w[b, 4] / (w[b, 4] + w[b, 5] + 1e-12) - 0.2 - 0.03 * b)) / 40 + 0.3 * np.floor(200 * w[b, [0, 1, 2, 3, 6]].sum()) / 200 for b in range(3))
                self.cache[k] = 100 + 10 * g + 0.2 * np.sin(1e4 * w.sum(0)[5])
            out.append(self.cache[k])
        return np.array(out)

s = fms.Search(Fake()); s.init_from_logs(); s.save()
r0 = s.pool.evaluate(s.inc.ravel()[None])[0]; s.inc_R = float(r0)
for e in sum([list(a.values()) for a in s.arch], []): e['ctx'] = -1
for g in range(8): s.generation(40, 6, 6)
s2 = fms.Search(Fake()); s2.load(); print('reload ok', s2.gen, s2.inc_R, [len(a) for a in s2.arch])
print(dict(zip(fms.OPS, s.op_probs().round(3).tolist())))
