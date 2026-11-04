"""Regression checks for PR9's reproduced review findings."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import json
from dataclasses import replace
import numpy as np
import pytest
from sdm.models.bayes_search import BayesianSearchModel, LimitedOptimizer
from sdm.models.gp_tree_blend import GPTreeBlend
from sdm.optimizers.bayes_pool import BayesianPool
from sdm import runs
from sdm import registry
from sdm.config import load_config

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('path',sorted((ROOT/'experiments').glob('bayes-*.toml')),ids=lambda p:p.stem)
def test_every_shipped_bayes_acquisition_is_accepted(path,dataset):
    """Exercise the real capability guard, without fitting every alternative model."""
    config=load_config(path)
    model=registry.build_model(config.model.name,config.model.params)
    optimizer=registry.build_optimizer(config.optimizer.name,config.optimizer.params)
    if config.optimizer.name=='bayes-pool':
        optimizer.params=replace(optimizer.params,samples=120)
    elif config.optimizer.name=='trust-dirichlet':
        optimizer.params=replace(optimizer.params,iterations=1,samples_per_center=4)
    else:
        raise AssertionError(f'Uncovered optimizer: {config.optimizer.name}')
    model.predict=lambda X:(np.full(len(X),110.),np.full(len(X),2.))
    proposed=optimizer.propose(model,dataset.X[:60],dataset.y[:60],n=1)
    from sdm.simplex import check_weights
    assert check_weights(proposed).shape==(1,21)


def module(name):
    spec=importlib.util.spec_from_file_location('bayes_review_'+name,ROOT/f'analysis/bayesian_search/{name}.py')
    item=importlib.util.module_from_spec(spec);spec.loader.exec_module(item)
    return item


def test_ard_honors_starts_and_deduplicates(monkeypatch,dataset):
    starts=[]
    def optimizer(self,objective,theta,bounds):
        starts.append(theta.copy())
        return theta,objective(theta)[0]
    monkeypatch.setattr(LimitedOptimizer,'__call__',optimizer)
    p=BayesianSearchModel.Params(kernel='additive-ard',lengths=(.2,.2,.5),noises=(.03,.03))
    model=BayesianSearchModel(p).fit(dataset.X[:30],dataset.y[:30])
    assert len(starts)==model.metrics['kernel_fits']==2
    np.testing.assert_allclose(np.exp(starts[0][:7]),.2)
    np.testing.assert_allclose(np.exp(starts[1][:7]),.5)


def test_ard_reports_fitted_values_and_termination(dataset):
    p=BayesianSearchModel.Params(kernel='additive-ard',lengths=(.5,),noises=(.03,))
    model=BayesianSearchModel(p).fit(dataset.X[:35],dataset.y[:35])
    kernel=model.estimator_.kernel_
    np.testing.assert_allclose([model.metrics[f'length_{i}'] for i in range(7)],kernel.k1.length_scale)
    assert model.metrics['noise']==kernel.k2.noise_level
    assert model.metrics['initial_noise']==.03
    assert 0<=model.metrics['optimizer_nit']<=18
    assert isinstance(model.metrics['optimizer_success'],bool)
    assert model.metrics['gp_seed_affects_fit'] is False
    assert np.isfinite(model.metrics['optimizer_gradient_inf_norm'])


def test_gp_seed_does_not_create_variability(dataset):
    kwargs=dict(lengths=(.5,),noises=(.03,))
    a=BayesianSearchModel(BayesianSearchModel.Params(seed=1,**kwargs)).fit(dataset.X[:30],dataset.y[:30])
    b=BayesianSearchModel(BayesianSearchModel.Params(seed=2,**kwargs)).fit(dataset.X[:30],dataset.y[:30])
    np.testing.assert_array_equal(a.predict(dataset.X[30:40])[0],b.predict(dataset.X[30:40])[0])


def test_ei_rejects_blend_disagreement(dataset):
    optimizer=BayesianPool(BayesianPool.Params(acquisition='ei'))
    with pytest.raises(ValueError,match='native Gaussian posterior'):
        optimizer.propose(GPTreeBlend(),dataset.X[:30],dataset.y[:30])


@pytest.mark.parametrize('spaces',[False,True])
def test_config_whitespace_and_mismatched_acquisition(tmp_path,spaces):
    study=module('candidates')
    path=tmp_path/'config.toml'
    source=(ROOT/'experiments/bayes-search-gp-ard-mean.toml').read_text()
    if spaces:source=source.replace('acquisition="mean"','acquisition = "mean"')
    path.write_text(source)
    assert study.validate_search_config(path,'gp-ard','mean').optimizer.params['acquisition']=='mean'
    assert path.read_text()==source
    with pytest.raises(ValueError,match='Acquisition mismatch'):
        study.validate_search_config(path,'gp-ard','ei')


def test_external_veto_can_reject_best_gp():
    study=module('candidates')
    keep=study.eligible_by_veto(np.array([6.28,4.34,4.90]),4.9)
    np.testing.assert_array_equal(keep,[False,True,True])


def test_line_pool_records_simplex_projection(dataset):
    from sdm.simplex import check_weights
    study=module('candidates');anchor=dataset.X[0]
    rows=study.line_pool(anchor,{'target':dataset.X[1]},radii=[.02,.15])
    check_weights([r[1] for r in rows])
    assert rows[0][2]==.02 and rows[1][2]==.15
    assert all(0<r[3]<=r[2]+1e-9 for r in rows)


@pytest.mark.skipif(shutil.which('git') is None,reason='real Git needed for layout regression')
@pytest.mark.parametrize('layout',['packed','worktree'])
def test_git_provenance_for_real_layouts(tmp_path,monkeypatch,layout):
    def git(*args,cwd):return subprocess.run(['git',*args],cwd=cwd,check=True,capture_output=True,text=True).stdout.strip()
    source=tmp_path/'source';source.mkdir()
    git('init',cwd=source);git('config','user.name','Test',cwd=source);git('config','user.email','test@example.invalid',cwd=source)
    (source/'src').mkdir();(source/'src/a.py').write_text('x=1\n')
    git('add','src/a.py',cwd=source);git('commit','-m','test',cwd=source)
    expected=git('rev-parse','HEAD',cwd=source)
    if layout=='packed':
        git('pack-refs','--all',cwd=source);target=source
    else:
        target=tmp_path/'worktree';git('worktree','add','-b','other',str(target),cwd=source)
        assert (target/'.git').is_file()
    monkeypatch.setattr(runs,'REPO_ROOT',target)
    assert runs.git_state()==dict(commit=expected,dirty=False)


def test_generated_guidance_uses_confirmed_results_and_decision23(tmp_path,monkeypatch):
    script=module('finalize')
    out=tmp_path/'delivery';out.mkdir();(out/'extension').mkdir()
    (out/'README.md').write_text('Historical approximate119/baseline118\n')
    rows=[dict(model=name,v32_score=80,mae=4,v32_top1_regret=2) for name in ['method-blend-stack','gp-ard','gp-tree-blend']]
    (out/'evaluation.json').write_text(json.dumps(rows));(out/'extension/evaluation.json').write_text('[]')
    monkeypatch.setattr(script,'OUT',out);monkeypatch.setattr(script,'CURRENT',out/'current')
    root=tmp_path/'checkout';root.mkdir()
    monkeypatch.setattr(script,'ROOT',root)
    monkeypatch.setattr(script,'git_state',lambda:dict(commit='test',dirty=False))
    script.main();text=(out/'README.md').read_text()
    assert '119.71206678869203' in text and '119.0916' in text and 'R < 115.76' in text
    assert 'Do not use RampingSolver.exe' in text
    assert 'No official submission2 is selected here' in text
    assert 'There was no remote push' not in text
    assert (out/'historical/README_2026-11-02.md').read_text().startswith('Historical')
    assert 'uv sync --locked' in text
    assert 'uv run --locked --no-sync sdm train' in text
    assert 'uv run --no-editable' not in text
    artifact=(root/'artifacts/bayesian_gp/README.md').read_text()
    assert 'uv run --locked --no-sync sdm train' in artifact
    assert 'uv run --no-editable' not in artifact
