"""Read the E3 scan: which minor objectives help each building, and at what magnitude and r."""
import numpy as np
from pool import load_log
from common import to_ratio, NAMES, MINOR
from common import read_json
W, R, tags = load_log()
m = np.array([t == 'E3_scan' for t in tags]); W, R = W[m].reshape(-1, 3, 7), R[m]
state = read_json('fms_state.json'); inc = np.array(state['inc'])
# incumbent at scan time = most common block per building
np.set_printoptions(precision=3, suppress=True, linewidth=200)
for b in range(3):
    others = [x for x in range(3) if x != b]
    keys = [tuple(np.round(w, 12)) for w in W[:, b]]
    vals, counts = np.unique(np.round(W[:, b], 12), axis=0, return_counts=True)
    base = vals[counts.argmax()]
    mb = np.abs(W[:, b] - base).sum(1) > 1e-9                 # rows where building b was varied
    Wb, Rb = W[mb, b], R[mb]
    reg = (Wb[:, 4] > 0) & (Wb[:, 5] > 0) & (Wb[:, 4] / (Wb[:, 4] + Wb[:, 5]) < 0.37) & (Wb[:, 4] / (Wb[:, 4] + Wb[:, 5]) > 0.15) & ((Wb[:, 4] + Wb[:, 5]) > 0.5)
    r = Wb[reg, 4] / (Wb[reg, 4] + Wb[reg, 5]); M = Wb[reg][:, MINOR] / (Wb[reg, 4] + Wb[reg, 5])[:, None]; y = Rb[reg]
    print(f'\n== building {b+1}: region rows {reg.sum()}, R quantiles (min/10%/50%) {np.round(np.percentile(y, [0, 10, 50]), 2)}; global rows {(~reg).sum()} min {Rb[~reg].min():.2f} median {np.median(Rb[~reg]):.1f}')
    print('  median R by r bin:', ' '.join(f'[{lo:.2f},{lo+0.04:.2f}):{np.median(y[(r >= lo) & (r < lo + 0.04)]):.2f}' for lo in np.arange(0.16, 0.36, 0.04)))
    for j, k in enumerate(MINOR):
        off = M[:, j] == 0
        bins = [(1e-3, 1e-2), (1e-2, 3e-2), (3e-2, 1e-1), (1e-1, 0.32)]
        txt = ' '.join(f'[{lo:g},{hi:g}):{np.median(y[(M[:, j] >= lo) & (M[:, j] < hi)]):.2f}' for lo, hi in bins)
        print(f'  {NAMES[k]:11s} off:{np.median(y[off]):.2f}  on by magnitude {txt}')
