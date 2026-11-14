"""How deep is the incumbent block inside its own neighbourhood box? Archive deltas by distance class."""
import numpy as np
from common import read_json
state = read_json('fms_state.json')
inc = np.array(state['inc'])
np.set_printoptions(precision=4, suppress=True, linewidth=200)
for b in range(3):
    W = np.array([e['w'] for e in state['arch'][b]]); d = np.array([e['delta'] for e in state['arch'][b]])
    s = W[:, 4] + W[:, 5]; ok = s > 0.3
    ratio = W[ok] / s[ok, None]; d = d[ok]
    r0 = inc[b] / (inc[b, 4] + inc[b, 5])
    rel = np.abs(ratio - r0)
    # box: r within +-0.03 and every minor ratio within max(0.02, 60 %) of the incumbent's value
    box = (np.abs(ratio[:, 4] - r0[4]) < 0.03) & np.all(rel[:, [0, 1, 2, 3, 6]] <= np.maximum(0.02, 0.6 * r0[[0, 1, 2, 3, 6]]), axis=1)
    db = d[box]; u = np.unique(np.round(db, 6))
    print(f'b{b+1}: incumbent ratios {np.round(r0, 4)}; blocks in box {box.sum()} ({len(u)} distinct cells)')
    print('    delta quantiles in box (min, 2%, 10%, 25%, 50%, 75%):', np.round(np.percentile(u, [0, 2, 10, 25, 50, 75]), 3),
          ' cells within 0.05/0.1/0.2:', (u < 0.05).sum(), (u < 0.1).sum(), (u < 0.2).sum())
