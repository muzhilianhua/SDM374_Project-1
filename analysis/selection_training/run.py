"""Fresh CLI evidence for the prespecified selection-conditioned training study."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from sdm.paths import REPO_ROOT
from sdm.runs import git_state, resolve_run

BASE = REPO_ROOT/'analysis/selection_training'
RESULTS = BASE/'results'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def selected_measured(run):
    frame = pd.read_csv(run.path/'predictions.csv')
    selected = pd.read_csv(run.path/'fold_scores.csv')[['split_id','top1_candidate_id']]
    return selected.merge(frame,left_on=['split_id','top1_candidate_id'],
        right_on=['split_id','candidate_id'],validate='one_to_one').actual.mean()


def main(phase):
    RESULTS.mkdir(parents=True, exist_ok=True)
    if git_state()['dirty']:
        raise ValueError('Commit model/configuration sources before evidence')
    path = RESULTS/'runs.json'
    manifest = json.loads(path.read_text()) if path.exists() else {}
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')

    def job(task):
        label, kind, config, options = task
        if label in manifest:
            print('Reuse '+label+': '+manifest[label]['run'], flush=True)
            return label, manifest[label]
        command = [str(Path(sys.executable).with_name('sdm')), kind, '-c',
            str(REPO_ROOT/'experiments'/config), '--name', 'selection-study-'+label, *options]
        print('Start '+label, flush=True)
        started = time.monotonic()
        result = subprocess.run(command, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
        (RESULTS/(label+'.log')).write_text(result.stdout+'\n'+result.stderr)
        if result.returncode:
            raise RuntimeError(label+' failed; see archived log')
        run = resolve_run(next(line[5:] for line in result.stdout.splitlines() if line.startswith('run: ')))
        record = {'run': run.id, 'command': command, 'seconds': time.monotonic()-started,
            'metrics': run.read_json('metrics.json'), 'meta': run.read_json('meta.json'),
            'thread_environment': {k: env[k] for k in
                ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}}
        if kind == 'evaluate':
            record['selected_actual_R'] = float(selected_measured(run))
        print('Done '+label+f" ({record['seconds']:.1f}s): "+run.id, flush=True)
        return label, record

    if phase == 'evaluate':
        tasks = [('baseline', 'evaluate', 'baseline-v32.toml', [])]
        tasks += [(f'{name}-{seed}', 'evaluate', f'selection-{name}-{seed}.toml', [])
                  for seed in (2026, 2027, 2028) for name in ('reference', 'corrected')]
    elif phase == 'fit':
        tasks = [(name+'-fit', 'train', f'selection-{name}-2026.toml', [])
                 for name in ('reference', 'corrected')]
    elif phase == 'search':
        tasks = [(f'{name}-search-{seed}', 'optimize', f'selection-search-{seed}.toml',
                  ['-n', '50', '--model-run', manifest[name+'-fit']['run']])
                 for name in ('reference', 'corrected') for seed in (2040, 2041)]
    elif phase == 'replay':
        # Refit the complete trained procedure through the public CLI, not just a pickle replay.
        label = 'corrected-fit'
        command = [str(Path(sys.executable).with_name('sdm')), 'reproduce', manifest[label]['run']]
        result = subprocess.run(command, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
        (RESULTS/'reproduce.log').write_text(result.stdout+'\n'+result.stderr)
        if result.returncode:
            raise RuntimeError('full fit reproduction failed')
        write(RESULTS/'reproduce.json', {'command': command, 'returncode': result.returncode,
                                       'output': result.stdout, 'git': git_state()})
        print(result.stdout, flush=True)
        return
    else:
        raise ValueError(phase)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(job, task) for task in tasks]
        for future in futures:
            label, record = future.result()
            manifest[label] = record
            write(path, manifest)
    if phase == 'evaluate':
        refs = [manifest[f'reference-{s}'] for s in (2026, 2027, 2028)]
        news = [manifest[f'corrected-{s}'] for s in (2026, 2027, 2028)]
        avg = lambda rows, k: float(np.mean([r['metrics'][k] for r in rows]))
        gates = {'mean_v32_improves': avg(news, 'v32_score') > avg(refs, 'v32_score'),
            'mae_each_seed': all(n['metrics']['mae'] <= 1.05*r['metrics']['mae'] for r,n in zip(refs,news)),
            'local_mae_each_seed': all(n['metrics']['v32_local_mae'] <= 1.05*r['metrics']['v32_local_mae']
                                      for r,n in zip(refs,news)),
            'mean_regret_not_worse': avg(news, 'v32_top1_regret') <= avg(refs, 'v32_top1_regret'),
            'mean_selected_actual_R_not_worse': np.mean([r['selected_actual_R'] for r in news])
                                                  <= np.mean([r['selected_actual_R'] for r in refs])}
        gates = {key:bool(value) for key,value in gates.items()}
        write(RESULTS/'decision.json', {'gates': gates, 'replacement_gate_passed': all(gates.values()),
            'reference_mean_score': avg(refs,'v32_score'), 'corrected_mean_score': avg(news,'v32_score'),
            'working_recipe': 'corrected' if all(gates.values()) else 'reference',
            'operational_W_changed': False, 'first_test_priority_changed': False, 'simulator_calls': 0})
        print(json.dumps(json.loads((RESULTS/'decision.json').read_text()), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('evaluate', 'fit', 'search', 'replay'))
    main(parser.parse_args().phase)
