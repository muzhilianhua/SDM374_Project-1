"""Declared adaptive extension after additive GP, unlike isotropic GP, matched the old stack."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time
from sdm.cli import execute

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'analysis/bayesian_search/delivery/extension'


def run(name):
    t=time.monotonic()
    result=execute('evaluate',ROOT/f'experiments/bayes-study-{name}.toml')
    return dict(model=name,run_id=result.id,seconds=time.monotonic()-t,**result.read_json('metrics.json'))


def main():
    OUT.mkdir(exist_ok=True)
    configs={
        'gp-ard':'name = "bayes-search"\n[model.params]\nfamily = "gp"\nkernel = "additive-ard"\nlengths = [0.5]\nnoises = [0.03]\n',
        'gp-ard-log':'name = "bayes-search"\n[model.params]\nfamily = "gp"\nkernel = "additive-ard"\nlengths = [0.5]\nnoises = [0.03]\ntarget = "log"\n',
        'gp-tree-blend':'name = "gp-tree-blend"\n[model.params]\n',
    }
    plan=dict(adaptive_extension=True,reason='Isotropic GP weak; additive GP 75.12 comparable to existing stack',
        methods=['shared-seven-objective ARD additive GP, raw/log','fixed equal additive GP / existing stack blend'],
        selection='empirical-Bayes marginal likelihood entirely within allowed training data; max18 L-BFGS iterations',
        blend_weights='fixed .5/.5 before examining blended outer score; no outer-fitted weights',
        caveat='Reused outer folds; exploratory discovery, not independent confirmation',simulator_calls=0)
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    for name,block in configs.items():
        text=('[data]\ndataset="dataset_same8"\n[model]\n'+block+'seed=2026\n'
              '[optimizer]\nname="bayes-pool"\n[optimizer.params]\nacquisition="ei"\nsamples=6000\nmax_distance_ratio=0.5\nseed=2026\n'
              '[evaluate]\nprotocol="frozen-v32"\ntop_k=50\n')
        (ROOT/f'experiments/bayes-study-{name}.toml').write_text(text,encoding='utf-8')
    if os.environ.get('PREPARE_ONLY')=='1':
        return
    rows=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        jobs={pool.submit(run,name):name for name in configs}
        for job in as_completed(jobs):
            try:
                row=job.result(); rows.append(row)
                (OUT/'evaluation.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
                print(row['model'],row['run_id'],row['v32_score'],row['v32_top1_regret'],row['seconds'],flush=True)
            except Exception as error:
                print('ERROR',jobs[job],repr(error),flush=True)


if __name__=='__main__':
    main()
