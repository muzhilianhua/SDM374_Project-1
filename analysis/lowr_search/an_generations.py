"""Delta distribution of the candidates of the last generations in fms_progress.jsonl (by evaluation order)."""
import numpy as np
from pool import load_log
from common import read_jsonl
W, R, tags = load_log()
prog = read_jsonl('fms_progress.jsonl')
tags = np.array(tags)
for g in prog[-4:]:
    prefix = g["run"] if (tags == f"{g['run']}_g{g['gen']}").any() else "fms"   # runs 1-3 were tagged "fms_g<gen>"
    idx = np.where(tags == f"{prefix}_g{g['gen']}")[0]
    idx = idx[-g['new_evals']:] if g['new_evals'] < len(idx) else idx      # run 1 used the same tags earlier
    d = R[idx] - (g['R_best'] + g['gain'])
    print(f"gen {g['gen']:2d} n={len(idx)} same plateau {(np.abs(d) < 1e-9).mean():.2f}  d<0: {(d < -1e-9).sum()}  0<d<0.02: {((d > 1e-9) & (d < 0.02)).sum()}  "
          f"0.02..0.1: {((d >= 0.02) & (d < 0.1)).sum()}  0.1..0.5: {((d >= 0.1) & (d < 0.5)).sum()}  >0.5: {(d >= 0.5).sum()}  smallest positive {d[d > 1e-9].min():.4f}")
