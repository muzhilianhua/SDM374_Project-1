"""Build and verify the clearly labelled GitHub handoff without changing source artifacts."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd

from sdm.data import load_dataset, sha256_of
from sdm.paths import REPO_ROOT
from sdm.runs import git_state, resolve_run
from sdm.simplex import FEATURES, METRICS, check_weights

STUDY = REPO_ROOT/'analysis/selection_training/delivery'
ONLINE = REPO_ROOT/'analysis/online_round1/delivery'
DEFAULT = REPO_ROOT/'SUBMISSION_READY'
ARCHIVE = REPO_ROOT/'artifacts/SUBMISSION_READY_2026-11-03.zip'
TEMPLATES = Path(__file__).with_name('templates')
REAL_P1 = 119.71206678869203
REAL_BASELINE = 119.0916
REAL_P01969 = 115.76481925917324
MEASUREMENT_SOURCE = ('https://github.com/muzhilianhua/SDM374_Project-1/blob/'
    '1bf9dbeadbd45ec3f51299b489d8893de1a71186/docs/decisions.md')
CANDIDATES = [
    ('P1', '01_FIRST_REAL_TEST', 'challenger', STUDY/'first_test_weights.json',
     'MEASURED, FAILED; do not resubmit', 'r01-004'),
    ('P2', '02_ORIGINAL_RECOMMENDATION', 'operational',
     REPO_ROOT/'analysis/offline_submission/submission/weights.json',
     'HISTORICAL ORIGINAL DEFAULT; not selected for submission 2', 'r01-003'),
    ('P3', '03_EXPLORATORY_ONLY', 'corrected-search-2040-r1', STUDY/'exploratory_weights.json',
     'EXPLORATORY ONLY; real result unknown', 'direction-probe p01'),
]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def markdown(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip()+'\n')


def checksum_map(out):
    return {str(p.relative_to(out)): sha256_of(p) for p in sorted(out.rglob('*'))
            if p.is_file() and p.name != 'CHECKSUMS.json'}


def known_pool_results():
    return {'r01-004': REAL_P1, 'r01-046': REAL_BASELINE, 'r01-047': REAL_P01969}


def validate_status_table(table):
    """Known points cannot become pending again when the historical pool is rebuilt."""
    known = known_pool_results()
    assert table.candidate_id.is_unique
    rows = table.set_index('candidate_id')
    for identifier, measured in known.items():
        assert np.isclose(rows.loc[identifier, 'measured_R'], measured, rtol=0, atol=1e-12)
        assert rows.loc[identifier, 'status'] == 'known_do_not_resubmit'
    unknown = rows.loc[~rows.index.isin(known)]
    assert unknown.measured_R.isna().all()
    assert (unknown.status == 'unmeasured_not_scheduled').all()


def verify(out, *, check_hashes=True):
    manifest = json.loads((out/'MANIFEST.json').read_text())
    if check_hashes:
        expected = json.loads((out/'CHECKSUMS.json').read_text())
        if checksum_map(out) != expected:
            raise ValueError('package file list or checksum changed')
    for entry in manifest['source_copies']:
        if sha256_of(out/entry['destination']) != entry['sha256']:
            raise ValueError('source bytes changed: '+entry['destination'])
    points = check_weights(np.vstack([check_weights(json.loads(
        (out/item['payload']).read_text())) for item in manifest['main_candidates']]))
    batches = {}
    for folder, count in [('05_ONLINE_ROUND1_50',50),('06_DIRECTION_PROBES_20',20)]:
        batch = check_weights(json.loads((out/folder/'weights.json').read_text()))
        assert len(batch) == count and len(np.unique(np.round(batch,9),axis=0)) == count
        individuals = sorted((out/folder/'weights').glob('*.json'))
        assert len(individuals) == count
        np.testing.assert_array_equal(check_weights(np.vstack([
            check_weights(json.loads(p.read_text())) for p in individuals])), batch)
        batches[folder] = batch
    np.testing.assert_array_equal(points[0], batches['05_ONLINE_ROUND1_50'][3])
    np.testing.assert_array_equal(points[1], batches['05_ONLINE_ROUND1_50'][2])
    np.testing.assert_array_equal(points[2], batches['06_DIRECTION_PROBES_20'][0])
    pending = pd.read_csv(out/'05_ONLINE_ROUND1_50/pending_results.csv',float_precision='round_trip')
    validate_status_table(pending)
    np.testing.assert_array_equal(pending[list(FEATURES)].to_numpy(),batches['05_ONLINE_ROUND1_50'])
    csv = pd.read_csv(out/'05_ONLINE_ROUND1_50/weights_w0_w20.csv',float_precision='round_trip')
    np.testing.assert_array_equal(csv[[f'w{i}' for i in range(21)]].to_numpy(),batches['05_ONLINE_ROUND1_50'])
    probes = pd.read_csv(out/'06_DIRECTION_PROBES_20/manifest.csv')
    assert probes.candidate_measured_R.isna().all() and not probes.new_candidate_actual_R_known.any()
    ds = load_dataset(REPO_ROOT/'dataset_same8')
    assert manifest['dataset_sha256'] == ds.sha256
    np.testing.assert_array_equal(batches['05_ONLINE_ROUND1_50'][45],
        check_weights(json.loads((REPO_ROOT/'weights.json').read_text()))[0])
    row = ds.ids.index('p01969')
    np.testing.assert_array_equal(batches['05_ONLINE_ROUND1_50'][46], ds.X[row])
    assert ds.y[row] == REAL_P01969
    observations = json.loads((out/'KNOWN_OBSERVATIONS.json').read_text())['observations']
    assert {r['alias']:r['measured_R'] for r in observations} == known_pool_results()
    policy = manifest['online_policy']
    assert policy['max_calls'] == 20 and policy['points_per_call'] == 1 and policy['sequential']
    assert policy['initial_training_rows'] == 2048 and policy['exclude_known_points']
    env = os.environ.copy()
    env.pop('PYTHONPATH',None)
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    replay = []
    for item in manifest['main_candidates']:
        command = [str(Path(sys.executable).with_name('sdm')),'predict','--model-run',
            str(out/'04_RECOMMENDED_MODEL'),'--weights',str(out/item['payload'])]
        result = subprocess.run(command,cwd=REPO_ROOT,env=env,capture_output=True,text=True,check=True)
        values = json.loads(result.stdout)
        np.testing.assert_allclose(values['predicted_mean'],[item['predicted_R']],atol=1e-10,rtol=0)
        np.testing.assert_allclose(values['predicted_std'],[item['heuristic_error_scale']],atol=1e-10,rtol=0)
        assert item['measured_R'] == (REAL_P1 if item['id'] == 'P1' else None)
        replay.append({'id':item['id'],'command':command,'prediction':values})
    report = {'source_copies_match':True,'main_single_candidates':3,
        'online_batch_size':50,'direction_probe_batch_size':20,'all_weights_feasible':True,
        'within_batch_distinct':True,'json_csv_and_individuals_agree':True,
        'cross_package_aliases_verified':True,'known_results_match_team_decisions':True,
        'unknown_measured_R_left_blank':True,'superseded_pool_not_scheduled':True,
        'cli_replay_matches':True,'cli_replays':replay,'simulator_calls':0}
    return report


def build(out):
    out.mkdir(parents=True, exist_ok=False)
    copies = []

    def copy(source, relative):
        target = out/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        copies.append({'source':str(source.relative_to(REPO_ROOT)),
            'destination':str(relative),'sha256':sha256_of(source)})

    central = pd.read_csv(STUDY/'candidates/central.csv').set_index('candidate')
    items = []
    for identifier,folder,name,source,role,alias in CANDIDATES:
        relative = Path(folder)/'weights.json'
        copy(source,relative)
        row = central.loc[name]
        item = {'id':identifier,'role':role,'payload':str(relative),'source_candidate':name,
            'source':str(source.relative_to(REPO_ROOT)),'source_sha256':sha256_of(source),
            'predicted_R':float(row.reference_mean),'heuristic_error_scale':float(row.reference_error),
            'distance_ratio':float(row.distance_ratio),'measured_R':REAL_P1 if identifier == 'P1' else None,'alias':alias,
            'real_result_source':MEASUREMENT_SOURCE if identifier == 'P1' else None,
            'submission_status':'measured_failed' if identifier == 'P1' else 'unmeasured_not_selected',
            'precision':'source bytes preserved; no new rounding'}
        items.append(item)
        markdown(out/folder/'README.md',f'''# {identifier} — {role}

**Payload: [weights.json](weights.json)** — one numeric 3x7 matrix, buildings 1/2/3.
Column order: cost,carbon,peak,discomfort,degradation,ramping,resilience.

Source: `{item['source']}`. Source candidate: `{name}`. Exact source bytes are preserved.
Recommended-model prediction: **{item['predicted_R']:.6f}**; heuristic error scale:
{item['heuristic_error_scale']:.6f}; distance ratio:{item['distance_ratio']:.6f}.
**Measured R: {str(item['measured_R']) + ' (FAILED; do not resubmit)' if item['measured_R'] is not None else 'UNKNOWN'}.**
Predictions are not simulator results. Error scales are not confidence bounds.
Also present as `{alias}`; these are aliases of the same controller, not separate discoveries.

P1 failed the real bar. P2 preserves the historical default; P2/P3 are not selected for submission 2.
P1 was optimistic by about 8.29 R; the other surrogate predictions carry the same risk.
Use [the package index](../README.md) for the complete decision and evaluation evidence.
''')
    for filename in ('model.pkl','config.toml','meta.json','metrics.json'):
        copy(STUDY/'reference_model'/filename,Path('04_RECOMMENDED_MODEL')/filename)
    model = resolve_run(str(STUDY/'reference_model')).load_model()
    predictor = model.predictor_
    params = {'registered_model':'honest-stack','outer_parameters':asdict(model.params),
        'ensemble':'three-fold training-only OOF convex blend',
        'fitted_blend_weights':dict(zip(('xgboost','catboost','histgb'),predictor.blend_weights.tolist())),
        'training_rows':2048,'input_features':list(FEATURES),'target':'ramping (lower is better)',
        'low_target_sample_weights':{'fraction':.2,'xgboost':3.,'catboost':3.,'histgb':1.},
        'error_proxy_is_heuristic':True,'members':{}}
    keys = {
        'xgboost':['n_estimators','learning_rate','max_depth','subsample','tree_method','objective','random_state','n_jobs'],
        'catboost':['iterations','learning_rate','depth','l2_leaf_reg','border_count','loss_function',
                    'bootstrap_type','bagging_temperature','random_strength','random_seed','thread_count'],
        'histgb':['max_iter','learning_rate','max_leaf_nodes','min_samples_leaf','l2_regularization','random_state']}
    for name,member in zip(keys,predictor.members):
        values = member.estimator.get_params()
        params['members'][name] = {k:values[k] for k in keys[name]}
    write(out/'04_RECOMMENDED_MODEL/parameters.json',params)
    write(out/'04_RECOMMENDED_MODEL/environment.json',{'python':sys.version.split()[0],
        'packages':{p:version(p) for p in ('numpy','pandas','scipy','scikit-learn','xgboost','catboost')},
        'dependency_lock':'../uv.lock in the repository root','requires_repository_source':True})
    markdown(out/'04_RECOMMENDED_MODEL/README.md','''# RECOMMENDED MODEL — honest-stack, seed 2026

**Restore model.pkl with `python3 analysis/submission_handoff/assets.py restore` from the repository root, then load it from this directory.** This is the freshly fitted recommended full-data model;
the selection-corrected experiment did not pass its replacement gate. The older published model
in artifacts/best_model is historical. The release asset provides the original fitted bytes; original config/meta/metrics, resolved
parameters and environment versions remain tracked here. Code and uv.lock are in the repository root.

| Member | Iterations | Learning rate | Main setting | Fitted blend weight |
|---|---:|---:|---|---:|
| XGBoost |600|.10|max_depth 3,subsample.9|.2920246172436632|
| CatBoost |600|.05|depth 5,l2_leaf_reg 5,border_count 128|.4437867721586836|
| HistGB |600|.05|max_leaf_nodes 15,min_samples_leaf 20|.2641886105976533|

Inputs are raw 21 weights in parameters.json order; 2048 measured rows, seed 2026. XGBoost/CatBoost
weight the lowest 20% training-target rows3x; HistGB uses uniform weights. Convex blend weights
use 3-fold training OOF predictions. Independent nested3-fold residual calibration uses 4 distance
bins and the .68 absolute-error quantile. The returned std is a heuristic error scale.

| Frozen-v32 comparison | Score (higher is better) | MAE | Top1 regret |
|---|---:|---:|---:|
| Recommended recipe, seed 2026 |75.1136584760|4.6932867464|1.0632429468|
| Recommended recipe, three-seed mean |75.3590513811|4.6980812519|1.1078203701|
| Selection-corrected experiment, three-seed mean |75.2920482885|4.6934024481|1.1078203701|

The binary is seed 2026, not an ensemble of three training seeds.
Source fit: `20261101-224249-train-selection-study-reference-fit`.
Reference evaluations: `20261101-223009-evaluate-selection-study-reference-2026`,
`20261101-223250-evaluate-selection-study-reference-2027`,
`20261101-223538-evaluate-selection-study-reference-2028`.
P1 was predicted at 111.418998 but measured 119.71206678869203; the model
remains an offline reference, with no demonstrated real improvement at its searched optimum. See
../07_EVIDENCE/model_evaluation.csv and source_runs.json for exact run IDs and metrics. The
frozen-v32 splits/scorer are unchanged. Reused offline folds do not establish independent SOTA
or a measured improvement for a new W. Three-region generalization remains unresolved.

From the repository root:

```sh
uv sync --locked --group gui
env -u PYTHONPATH uv run --locked --group gui sdm predict \\
  --model-run SUBMISSION_READY/04_RECOMMENDED_MODEL \\
  --weights SUBMISSION_READY/01_FIRST_REAL_TEST/weights.json
```

Run evaluate/train with experiments/selection-reference-2026.toml to refit from the unchanged
dataset. Training/optimization implementations live in src/sdm; this handoff adds no new model.
''')
    for filename in ('weights.json','weights_w0_w20.csv','candidates.csv','pending_results.csv',
                     'plan.csv','constraints.json','budget.json','validation.json'):
        name = 'historical_'+filename if filename in {'pending_results.csv','budget.json','constraints.json'} else filename
        copy(ONLINE/filename,Path('05_ONLINE_ROUND1_50')/name)
    pending = pd.read_csv(ONLINE/'pending_results.csv', float_precision='round_trip')
    pending['status'] = 'unmeasured_not_scheduled'
    pending['notes'] = 'Superseded idea pool; not scheduled for an evaluator call.'
    for candidate, value in known_pool_results().items():
        rows = pending.candidate_id == candidate
        pending.loc[rows, 'measured_R'] = value
        pending.loc[rows, 'status'] = 'known_do_not_resubmit'
        pending.loc[rows, 'notes'] = 'Known result; source: docs/decisions.md decisions 17-23 and Findings.'
    pending.to_csv(out/'05_ONLINE_ROUND1_50/pending_results.csv', index=False, float_format='%.17g')
    write(out/'05_ONLINE_ROUND1_50/current_policy.json', {
        'status':'superseded_idea_pool_only', 'max_sequential_calls':20, 'weights_per_call':1,
        'initial_training_rows':2048, 'excluded_known_ids':list(known_pool_results()),
        'exclude_all_training_rows_and_previous_calls':True, 'policy_implemented_here':False})
    for path in sorted((ONLINE/'weights').glob('*.json')):
        copy(path,Path('05_ONLINE_ROUND1_50/weights')/path.name)
    markdown(out/'05_ONLINE_ROUND1_50/README.md', (TEMPLATES/'online.md').read_text())
    for filename in ('weights.json','manifest.csv','summary.json','raw_weights.json'):
        copy(STUDY/'direction_probes'/filename,Path('06_DIRECTION_PROBES_20')/filename)
    for path in sorted((STUDY/'direction_probes/weights').glob('*.json')):
        copy(path,Path('06_DIRECTION_PROBES_20/weights')/path.name)
    markdown(out/'06_DIRECTION_PROBES_20/README.md', (TEMPLATES/'probes.md').read_text())
    evidence = {'evaluation.csv':'model_evaluation.csv','runs.json':'source_runs.json',
        'decision.json':'source_model_decision.json','recommendation.json':'historical_source_recommendation.json',
        'candidates/central.csv':'candidate_judges.csv','candidates/neighbors.csv':'fresh_neighborhood_audit.csv',
        'candidates/training_subset_summary.csv':'training_subset_summary.csv',
        'precision-old/summary.csv':'precision_old_summary.csv',
        'precision-new/summary.csv':'precision_new_summary.csv',
        'test_result.json':'source_training_test_result.json','verification_extended.json':'source_verification.json'}
    for source,target in evidence.items():
        copy(STUDY/source,Path('07_EVIDENCE')/target)
    markdown(out/'07_EVIDENCE/README.md', (TEMPLATES/'evidence.md').read_text())
    ds = load_dataset(REPO_ROOT/'dataset_same8')
    manifest = {'release':'2026-11-03-member-C-reviewed-handoff','build_git':git_state(),
        'builder':'analysis/submission_handoff/package.py','builder_sha256':sha256_of(Path(__file__)),
        'template_sha256':{p.name:sha256_of(p) for p in sorted(TEMPLATES.glob('*.md'))},
        'dataset_sha256':ds.sha256,'feature_order':list(FEATURES),'per_building_order':list(METRICS),
        'constraint':'finite,nonnegative,each building sums to1 (atol1e-9)',
        'recommended_model':'04_RECOMMENDED_MODEL/model.pkl',
        'source_model_run':'20261101-224249-train-selection-study-reference-fit',
        'main_candidates':items,'source_copies':copies,'simulator_calls_during_packaging':0,
        'team_offline_submissions_recorded':1,'known_observations':'KNOWN_OBSERVATIONS.json',
        'online_policy':{'max_calls':20,'points_per_call':1,'sequential':True,
                         'initial_training_rows':2048,'exclude_known_points':True},
        'online_pool_status':'superseded_idea_pool_only',
        'online_round1_count':50,'separate_direction_probe_count':20,'operational_W_replaced':False}
    observations = [
        {'id':'P1','alias':'r01-004','measured_R':REAL_P1,'status':'measured_failed',
         'weights_sha256':items[0]['source_sha256'],'source':MEASUREMENT_SOURCE},
        {'id':'baseline-W*','alias':'r01-046','measured_R':REAL_BASELINE,'status':'TA_confirmed',
         'weights_sha256':sha256_of(REPO_ROOT/'weights.json'),'source':MEASUREMENT_SOURCE},
        {'id':'p01969','alias':'r01-047','measured_R':REAL_P01969,'status':'training_row',
         'source':'dataset_same8/train.csv'}]
    write(out/'KNOWN_OBSERVATIONS.json', {'observations':observations,
        'imported_from_team_records':True,'new_evaluator_calls':0,'frozen_training_data_modified':False})
    write(out/'MANIFEST.json',manifest)
    markdown(out/'README.md', (TEMPLATES/'index.md').read_text())
    write(out/'VERIFICATION.json',verify(out,check_hashes=False))
    write(out/'CHECKSUMS.json',checksum_map(out))
    archive = ARCHIVE if out == DEFAULT else out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob('*')):
            if path.is_file():
                item = zipfile.ZipInfo(str(Path('SUBMISSION_READY')/path.relative_to(out)),(2026,11,3,0,0,0))
                item.compress_type = zipfile.ZIP_DEFLATED
                item.external_attr = 0o100644 << 16
                z.writestr(item,path.read_bytes())
    archive.with_suffix(archive.suffix+'.sha256').write_text(sha256_of(archive)+'  '+archive.name+'\n')
    print(json.dumps({'out':str(out),'files':len(checksum_map(out)),'archive':str(archive),
        'archive_sha256':sha256_of(archive),'verified':True},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('build','verify'))
    parser.add_argument('--out',type=Path,default=DEFAULT)
    args = parser.parse_args()
    if args.phase == 'build':
        build(args.out.resolve())
    else:
        print(json.dumps(verify(args.out.resolve()),indent=2))
