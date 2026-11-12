"""Continue the extracted-runtime boundary experiment from its best candidate."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='1'
import argparse, json, time, warnings
from pathlib import Path
import numpy as np
from boundary_search import setup, simulate, generate, score_candidate, worker_init
warnings.filterwarnings('ignore', category=FutureWarning)

BEST = np.array([
 [0.0003855940280853773,0.045,0,0,0.25760783455547365,0.687006571416441,0.01],
 [3.01522978528055e-05,0.010000000000000002,0,0.00036522105768565166,0.21029005469622036,0.6492697850133666,0.13004478693487453],
 [0,0,0.010203889879216995,0,0.24635139398101458,0.7322436592497638,0.011201056890004769]
])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--n',type=int,default=12); ap.add_argument('--workers',type=int,default=3); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    started=time.perf_counter(); setup(); anchor=simulate(BEST,capture=True)
    assert abs(anchor['R']-104.80496856663376)<1e-7
    candidates, raw=generate(BEST,anchor['records'],a.n)
    payload={'kind':'extracted-runtime boundary continuation','formal_solver_calls':0,'anchor_R':anchor['R'],'anchor_weights':BEST.tolist(),'anchor_weekly_R':anchor['weekly_R'],'raw_boundaries':raw,'results':[],'best':{'R':anchor['R'],'weights':BEST.tolist(),'weekly_R':anchor['weekly_R']}}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf8')
    print(f'anchor R={anchor["R"]:.12f}; boundaries={raw}; testing={len(candidates)}',flush=True)
    from concurrent.futures import ProcessPoolExecutor,as_completed
    from multiprocessing import get_context
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=get_context('spawn'),initializer=worker_init) as pool:
        futures=[pool.submit(score_candidate,c) for c in candidates]
        for i,f in enumerate(as_completed(futures),1):
            r=f.result(); payload['results'].append(r)
            if r['R']<payload['best']['R']: payload['best']=r
            a.out.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf8')
            print(f'{i}/{len(candidates)} R={r["R"]:.12f} best={payload["best"]["R"]:.12f}',flush=True)
    payload['results'].sort(key=lambda x:x['R']); payload['elapsed_seconds']=time.perf_counter()-started
    a.out.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'best':payload['best'],'output':str(a.out)},indent=2),flush=True)
if __name__=='__main__': main()
