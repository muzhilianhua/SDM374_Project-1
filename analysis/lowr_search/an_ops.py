"""Outcome of each move type around the 104.5027 incumbent (run 2: no adoption, so every delta is
relative to the same point). Shows the correlation length of the landscape: how far a move can go
before the result is no better than a random block of the basin."""
import sys, numpy as np
from common import read_json
state = read_json(sys.argv[1] if len(sys.argv) > 1 else 'fms_state.json')
rows = {}
for b in range(3):
    for e in state['arch'][b]:
        if e['ctx'] == state['ctx'] and e['op'] not in ('seed',) and not e['op'].startswith('fms_'):
            rows.setdefault(e['op'], []).append(e['delta'])
order = ['scale0.001', 'scale0.003', 'scale0.01', 'scale0.03', 'scale0.1', 'scale0.3', 'scale1', 'multi0.002', 'multi0.01', 'multi0.05', 'multi0.2',
         'snap1000', 'snap400', 'snap100', 'snap40', 'toggle', 'cross', 'transplant', 'sur', 'global', 'funnel']
print(f"{'move':12s} {'n':>4s} {'same cell':>9s} {'<+0.05':>7s} {'<+0.2':>6s} {'median':>8s} {'best':>8s}")
for op in order:
    if op in rows:
        d = np.array(rows[op])
        print(f"{op:12s} {len(d):4d} {(np.abs(d) < 1e-9).mean():9.2f} {(d < 0.05).mean():7.2f} {(d < 0.2).mean():6.2f} {np.median(d):8.3f} {d.min():8.4f}")
