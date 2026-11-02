"""Matched seed check of the discovered fixed blend; seeds do not create new held-out observations."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import time
from sdm.cli import execute

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'analysis/bayesian_search/delivery/seed_audit'


def evaluate(item):
    name,seed=item
    source=ROOT/('experiments/bayes-study-gp-tree-blend.toml' if name=='gp-tree-blend' else 'experiments/method-blend-stack.toml')
    text=source.read_text().replace('seed = 2026',f'seed = {seed}').replace('seed=2026',f'seed={seed}')
    config=ROOT/f'experiments/bayes-audit-{name}-{seed}.toml'
    config.write_text(text,encoding='utf-8')
    start=time.monotonic()
    run=execute('evaluate',config)
    return dict(model=name,seed=seed,run_id=run.id,seconds=time.monotonic()-start,**run.read_json('metrics.json'))


def main():
    OUT.mkdir(exist_ok=True)
    (OUT/'plan.json').write_text(json.dumps(dict(seeds=[2027,2028],
        fixed_blend_weight=.5,retune_after_audit=False,unchanged_frozen_v32=True,
        gate='3-seed mean score improves; mean top1 regret does not increase; global/local MAE <=1.05x reference at each seed',
        caveat='Reused folds; seeds assess algorithm sensitivity, not independent confirmation',simulator_calls=0),indent=2))
    if os.environ.get('PREPARE_ONLY')=='1':return
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(evaluate,(name,seed)) for seed in [2027,2028]
              for name in ['gp-tree-blend','method-blend-stack']]
        for job in as_completed(jobs):
            row=job.result();rows.append(row)
            (OUT/'evaluation.json').write_text(json.dumps(rows,indent=2))
            print(row['model'],row['seed'],row['v32_score'],row['v32_top1_regret'],flush=True)


if __name__=='__main__':main()
