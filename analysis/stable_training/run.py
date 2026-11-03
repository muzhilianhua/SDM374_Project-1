"""Execute the frozen CLI experiment stages and preserve every result and failure."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from sdm.paths import REPO_ROOT
from sdm.runs import git_state, resolve_run

RESULTS = REPO_ROOT / 'analysis/stable_training/results/study'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main(phase):
    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest_path = RESULTS / 'runs.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if git_state()['dirty']:
        raise ValueError('Commit model and configuration sources before evidence runs')
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')

    def job(task):
        label, kind, config, options = task
        if label in manifest:
            print('Reuse ' + label + ': ' + manifest[label]['run'], flush=True)
            return label, manifest[label]
        command = [str(Path(sys.executable).with_name('sdm')), kind, '-c',
                   str(REPO_ROOT / 'experiments' / config), '--name', 'stable-study-' + label, *options]
        print('Start ' + label, flush=True)
        started = time.monotonic()
        result = subprocess.run(command, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
        (RESULTS / (label + '.log')).write_text(result.stdout + '\n' + result.stderr)
        if result.returncode:
            write(RESULTS / (label + '-failure.json'), {'command': command, 'returncode': result.returncode})
            raise RuntimeError(label + ' failed; see archived log')
        run_path = next(line.removeprefix('run: ') for line in result.stdout.splitlines()
                        if line.startswith('run: '))
        run = resolve_run(run_path)
        record = {'run': run.id, 'command': command, 'seconds': time.monotonic()-started,
                  'metrics': run.read_json('metrics.json'), 'meta': run.read_json('meta.json'),
                  'thread_environment': {key: env[key] for key in
                                         ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}}
        print('Done ' + label + f" ({record['seconds']:.1f}s): " + run.id, flush=True)
        return label, record

    if phase == 'discovery':
        tasks = [(name + '-2026', 'evaluate', f'stable-{name}-2026.toml', [])
                 for name in ('reference', 'mild', 'strong')]
    elif phase == 'audit':
        selected = json.loads((RESULTS / 'selection.json').read_text())['profile']
        tasks = [(name + '-' + str(seed), 'evaluate', f'stable-{name}-{seed}.toml', [])
                 for name in ('reference', selected) for seed in (2027, 2028)]
    elif phase == 'fit':
        selected = json.loads((RESULTS / 'selection.json').read_text())['profile']
        tasks = [(name + '-fit', 'train', f'stable-{name}-2026.toml', [])
                 for name in ('reference', selected)]
    elif phase == 'search':
        selected = json.loads((RESULTS / 'selection.json').read_text())['profile']
        tasks = [(f'{name}-{mode}-{seed}', 'optimize', f'stable-search-{mode}-{seed}.toml',
                  ['-n', '50', '--model-run', manifest[name + '-fit']['run']])
                 for name in ('reference', selected) for mode in ('central', 'neighbors')
                 for seed in (2026, 2027)]
    elif phase == 'calibration':
        tasks = [(f'honest-{seed}', 'evaluate', f'stable-honest-{seed}.toml', [])
                 for seed in (2026, 2027, 2028)]
        tasks.append(('honest-fit', 'train', 'stable-honest-2026.toml', []))
    elif phase == 'calibration-search':
        tasks = [('honest-neighbors-2026', 'optimize', 'stable-search-neighbors-2026.toml',
                  ['-n', '50', '--model-run', manifest['honest-fit']['run']])]
    else:
        raise ValueError(phase)
    # Distinct CLI run directories; at most two fits/searches occupy the machine at once.
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(job, task) for task in tasks]
        for future in futures:
            label, result = future.result()
            manifest[label] = result
            write(manifest_path, manifest)
    if phase == 'discovery':
        ref = manifest['reference-2026']['metrics']
        checked = []
        for profile in ('mild', 'strong'):
            m = manifest[profile + '-2026']['metrics']
            gates = {'score': m['v32_score'] >= ref['v32_score']-.5,
                     'mae': m['mae'] <= 1.05*ref['mae'],
                     'local_mae': m['v32_local_mae'] <= 1.05*ref['v32_local_mae'],
                     'regret': m['v32_top1_regret'] <= ref['v32_top1_regret']+.5,
                     'coverage': .55 <= m['std_coverage'] <= .80}
            checked.append({'profile': profile, 'score': m['v32_score'], 'gates': gates,
                            'eligible': all(gates.values())})
        eligible = [r for r in checked if r['eligible']]
        selected = max(eligible or checked, key=lambda r: r['score'])
        write(RESULTS / 'selection.json', {'profile': selected['profile'],
            'discovery_eligible': selected['eligible'], 'candidates': checked,
            'post_audit_reselection_allowed': False})
        print(json.dumps(json.loads((RESULTS / 'selection.json').read_text()), indent=2), flush=True)
    if phase == 'audit':
        selected = json.loads((RESULTS / 'selection.json').read_text())
        profile = selected['profile']
        refs = [manifest[f'reference-{s}']['metrics'] for s in (2026, 2027, 2028)]
        news = [manifest[f'{profile}-{s}']['metrics'] for s in (2026, 2027, 2028)]
        average = lambda rows, key: sum(r[key] for r in rows)/len(rows)
        gates = {'discovery': selected['discovery_eligible'],
                 'score': average(news, 'v32_score') > average(refs, 'v32_score'),
                 'mae': all(n['mae'] <= 1.05*r['mae'] for n, r in zip(news, refs)),
                 'regret': average(news, 'v32_top1_regret') <= average(refs, 'v32_top1_regret')+.5}
        write(RESULTS / 'audit_decision.json', {'profile': profile, 'gates': gates,
            'replacement_gate_passed': all(gates.values()),
            'reference_mean_score': average(refs, 'v32_score'),
            'new_mean_score': average(news, 'v32_score'),
            'operational_W_changed': False, 'simulator_calls': 0})
        print(json.dumps(json.loads((RESULTS / 'audit_decision.json').read_text()), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('discovery', 'audit', 'fit', 'search',
                                        'calibration', 'calibration-search'))
    main(parser.parse_args().phase)
