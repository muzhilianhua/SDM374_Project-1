import csv, json, time, numpy as np
from pool import SolverPool
rows = list(csv.DictReader(open('../SDM374_Project-1/dataset_same8/train.csv')))
r = next(x for x in rows if x['candidate_id'] == 'p01969')
cols = [c for c in rows[0] if c not in ('candidate_id', 'ramping')]
p = [float(r[c]) for c in cols]
best = json.load(open('../replay/evidence/best_weights_phaseE5.json', encoding='utf-8'))['weights_flat']
pool = SolverPool(n_workers=1)
t = time.time(); R = pool.evaluate([p, best], tag='sanity'); print(R.tolist(), 'expected', float(r['ramping']), 104.85173345124349, 'secs', round(time.time() - t, 1))
