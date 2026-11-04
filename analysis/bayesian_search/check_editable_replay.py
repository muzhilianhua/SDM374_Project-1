"""Verify documented editable CLI commands in an isolated ASCII-path worktree.

Run with the repository environment. No course evaluator is called. Only textual
run evidence is copied back; fitted model binaries stay in the temporary checkout.
"""
from pathlib import Path
import json
import os
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'analysis/bayesian_search/delivery/current/editable_replay'


def main():
    env = os.environ.copy()
    for key in ('PYTHONPATH', 'PYTHONHOME', 'UV_PROJECT_ENVIRONMENT', 'UV_NO_EDITABLE', 'VIRTUAL_ENV'):
        env.pop(key, None)
    env.update(PYTHONUTF8='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    git, uv = shutil.which('git'), shutil.which('uv')
    if not git or not uv:
        raise RuntimeError('Git and uv must be on PATH')
    commit = subprocess.check_output([git, 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    base = Path(tempfile.mkdtemp(prefix='sdm374-editable-')).resolve()
    checkout = base / 'checkout'
    if not str(checkout).isascii():
        raise RuntimeError('Use an ASCII-only temporary directory')
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.run([git, 'worktree', 'add', '--detach', str(checkout), commit], cwd=ROOT, check=True)
    commands = []

    def run(name, args):
        commands.append(dict(name=name, command=args))
        with (OUT / (name + '.log')).open('wb') as log:
            subprocess.run(args, cwd=checkout, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print('PASS:', name, flush=True)

    try:
        run('sync', [uv, 'sync', '--locked'])
        probe = '''import importlib.metadata as m,json,os
from pathlib import Path
import sdm
from sdm.paths import REPO_ROOT
from sdm.data import load_dataset
from sdm.runs import git_state
assert 'PYTHONPATH' not in os.environ
assert REPO_ROOT.resolve()==Path.cwd().resolve(),(REPO_ROOT,Path.cwd())
assert Path(sdm.__file__).resolve().is_relative_to(Path.cwd()/'src')
direct=json.loads(m.distribution('sdm374').read_text('direct_url.json'))
assert direct['dir_info']['editable'] is True
dataset=load_dataset(REPO_ROOT/'dataset_same8')
assert len(dataset.y)==2048
assert git_state()['dirty'] is False
print(json.dumps(dict(root=str(REPO_ROOT),editable=True,pythonpath_present=False,rows=len(dataset.y),dataset_sha256=dataset.sha256,git=git_state()),indent=2))
'''
        run('editable_probe', [uv, 'run', '--locked', '--no-sync', 'python', '-c', probe])
        run('gp_train', [uv, 'run', '--locked', '--no-sync', 'sdm', 'train', '-c', 'experiments/bayes-study-gp-ard.toml'])
        run('svr_optimize', [uv, 'run', '--locked', '--no-sync', 'sdm', 'optimize', '-c', 'experiments/bayes-study-svr.toml', '-n', '1'])
        evidence = []
        for source in sorted((checkout / 'runs').iterdir()):
            if not source.is_dir():
                continue
            dest = OUT / source.name
            dest.mkdir(exist_ok=True)
            copied = []
            for path in source.iterdir():
                if path.suffix in ('.json', '.toml', '.csv', '.txt'):
                    shutil.copy2(path, dest / path.name)
                    copied.append(path.name)
            evidence.append(dict(run_id=source.name,files=copied))
        (OUT / 'verification.json').write_text(json.dumps(dict(source_commit=commit,commands=commands,
            evidence=evidence,editable=True,pythonpath_present=False,solver_calls=0), indent=2)+'\n', encoding='utf-8')
    finally:
        # Check the exact resolved target before Git recursively removes the worktree.
        if checkout.resolve().parent != base or checkout.name != 'checkout':
            raise RuntimeError('Unexpected cleanup target')
        subprocess.run([git, 'worktree', 'remove', '--force', str(checkout)], cwd=ROOT, check=True)
        base.rmdir()


if __name__ == '__main__':
    main()
