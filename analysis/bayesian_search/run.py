"""Run a predeclared model sweep using the canonical SDM experiment executor."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import time
from sdm.runs import git_state

# Set these before importing numpy/sklearn or spawning workers.
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"analysis/bayesian_search/delivery"
RECIPES = {
    "gp-matern": dict(family="gp", kernel="matern"),
    "gp-matern-log": dict(family="gp", kernel="matern", target="log"),
    "gp-sqrt": dict(family="gp", kernel="matern", transform="sqrt"),
    "gp-sqrt-log": dict(family="gp", kernel="matern", transform="sqrt", target="log"),
    "gp-additive": dict(family="gp", kernel="additive"),
    "gp-additive-log": dict(family="gp", kernel="additive", target="log"),
    "gp-rbf": dict(family="gp", kernel="rbf"),
    "gp-rq-log": dict(family="gp", kernel="rq", target="log"),
    "gp-trend": dict(family="gp", kernel="matern", trend=True),
    "bayes-quadratic": dict(family="bayesian-ridge"),
    "ard-quadratic": dict(family="ard"),
    "svr": dict(family="svr"),
    "svr-log": dict(family="svr", target="log"),
    "svr-sqrt-log": dict(family="svr", target="log", transform="sqrt"),
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def prepare():
    for name, params in RECIPES.items():
        text = '[data]\ndataset = "dataset_same8"\n\n[model]\nname = "bayes-search"\n[model.params]\nseed = 2026\n'
        for key, value in params.items():
            text += f'{key} = {json.dumps(value)}\n'
        acquisition = ('ei' if params['family']=='gp' and params.get('target','raw')=='raw'
                       and not params.get('trend',False) else 'pessimistic')
        text += ('\n[optimizer]\nname = "bayes-pool"\n[optimizer.params]\n'
                 f'acquisition = "{acquisition}"\nsamples = 6000\nmax_distance_ratio = 0.5\nseed = 2026\n'
                 '\n[evaluate]\nprotocol = "frozen-v32"\ntop_k = 50\n')
        (ROOT/f'experiments/bayes-study-{name}.toml').write_text(text, encoding="utf-8")
    write_json(OUT/'plan.json', dict(recipes=RECIPES, seed=2026,
        protocol='unchanged frozen-v32; 10 complete outer folds',
        gp_selection='training-only marginal likelihood over 3 lengths x 2 noises',
        svr_selection='3-fold training-only low-weighted MAE; inner training-only scaling',
        promotion='discovery only; use matched folds, regret and low-region error; no measured improvement claim',
        confirmed_team_records=json.loads((ROOT/'analysis/bayesian_search/team_records.json').read_text()),
        gp_seed_note='No random restarts; seed does not vary the GP fit',
        simulator_calls=0))


def provenance():
    from sdm.data import load_dataset
    state = git_state()
    commit = state['commit']
    files = [ROOT/'src/sdm/models/bayes_search.py', ROOT/'src/sdm/optimizers/bayes_pool.py', Path(__file__)]
    write_json(OUT/'provenance.json', dict(commit=commit, git=state, python=platform.python_version(),
        packages={k:version(k) for k in ['numpy','pandas','scipy','scikit-learn','xgboost','catboost']},
        dataset_sha256=load_dataset(ROOT/'dataset_same8').sha256, seed=2026,
        files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        note='CLI run git fields can be null if git executable is unavailable; this records the actual Git HEAD.'))


def evaluate(name):
    from sdm.cli import execute
    started = time.monotonic()
    path = ROOT/(f'experiments/bayes-study-{name}.toml' if name in RECIPES else f'experiments/{name}.toml')
    run = execute('evaluate', path)
    return dict(model=name, run_id=run.id, seconds=time.monotonic()-started, **run.read_json('metrics.json'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare','evaluate'])
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase == 'prepare':
        prepare()
        return
    provenance()
    rows = []
    names = list(RECIPES)+['baseline-v32','xgboost-v32','method-blend-stack']
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(evaluate, name): name for name in names}
        for job in as_completed(jobs):
            name = jobs[job]
            try:
                row = job.result()
                rows.append(row)
                write_json(OUT/'evaluation.json', rows)
                print(name, row['run_id'], 'score', round(row['v32_score'],4),
                      'regret',round(row['v32_top1_regret'],4), 'seconds',round(row['seconds'],1),flush=True)
            except Exception as error:
                write_json(OUT/f'error-{name}.json', dict(model=name,error=repr(error)))
                print('ERROR',name,repr(error),flush=True)
    import pandas as pd
    pd.DataFrame(rows).sort_values('v32_score',ascending=False).to_csv(OUT/'model_comparison.csv',index=False)


if __name__ == '__main__':
    main()
