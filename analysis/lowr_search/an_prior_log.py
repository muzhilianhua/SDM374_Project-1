import json, numpy as np, collections
from pool import normalize
np.set_printoptions(linewidth=250, precision=4, suppress=True)
rows = [json.loads(l) for l in open('../replay/evidence/eval_log.jsonl', encoding='utf-8')]
W = normalize([r['weights'] for r in rows]).reshape(-1, 3, 7); R = np.array([r['R'] for r in rows]); ph = np.array([r['phase'] for r in rows])
for p in ['vE1_b1', 'vE1_b2', 'vE1_b3', 'vE3_G', 'vE3_A', 'vertex', 'v5A_b1','vE2_A1']:
    m = np.where(ph == p)[0]
    print('==', p, len(m))
    for b in range(3):
        u = np.unique(np.round(W[m, b], 6), axis=0)
        print('  b%d distinct blocks %d' % (b + 1, len(u)), '' if len(u) > 3 else u.tolist())
    o = m[np.argsort(R[m])[:4]]
    for i in o: print('   ', round(R[i], 3), W[i, 0], W[i, 1], W[i, 2])
# how do global samples look: size of support / min weight stats of vE3_G
m = ph == 'vE3_G'
print('vE3_G support sizes', collections.Counter((W[m] > 1e-9).sum(2).ravel().tolist()))
print('vE3_G R quantiles', np.percentile(R[m], [0, 5, 25, 50, 75]).round(2))
# share of F5+F6 vs R across all log
s56 = W[:, :, 4] + W[:, :, 5]
for lo, hi in [(0, .2), (.2, .5), (.5, .8), (.8, .9), (.9, .97), (.97, 1.01)]:
    mm = (s56.min(1) >= lo) & (s56.min(1) < hi)
    if mm.sum(): print(f'min-building F5+F6 share in [{lo},{hi}): n={mm.sum():5d} minR={R[mm].min():.3f} medR={np.median(R[mm]):.2f}')
# minor-weight magnitudes explored among R<106
m = R < 106
minor = W[m][:, :, [0, 1, 2, 3, 6]]
pos = minor[minor > 0]
print('minor weights among R<106: n', pos.size, 'quantiles', np.percentile(pos, [0, 5, 25, 50, 75, 95, 100]).round(5))
r = W[m][:, :, 4] / (W[m][:, :, 4] + W[m][:, :, 5])
print('r=F5/(F5+F6) among R<106 per building: min/med/max', r.min(0).round(3), np.median(r, 0).round(3), r.max(0).round(3))
