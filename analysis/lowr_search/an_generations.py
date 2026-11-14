"""Delta distribution of the candidates of the last generations in fms_progress.jsonl (by evaluation order)."""
import json, numpy as np
from pool import load_log
W, R, tags = load_log()
prog = [json.loads(l) for l in open('fms_progress.jsonl', encoding='utf-8')]
tags = np.array(tags)
for g in prog[-4:]:
    idx = np.where(tags == f"fms_g{g['gen']}")[0]
    idx = idx[-g['new_evals']:] if g['new_evals'] < len(idx) else idx      # run 1 used the same tags earlier
    d = R[idx] - (g['R_best'] + g['gain'])
    print(f"gen {g['gen']:2d} n={len(idx)} same plateau {(np.abs(d) < 1e-9).mean():.2f}  d<0: {(d < -1e-9).sum()}  0<d<0.02: {((d > 1e-9) & (d < 0.02)).sum()}  "
          f"0.02..0.1: {((d >= 0.02) & (d < 0.1)).sum()}  0.1..0.5: {((d >= 0.1) & (d < 0.5)).sum()}  >0.5: {(d >= 0.5).sum()}  smallest positive {d[d > 1e-9].min():.4f}")
