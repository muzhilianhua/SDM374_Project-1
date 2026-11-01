"""Score every existing method candidate and course baseline W with one unchanged CLI predictor."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
import tomllib

import numpy as np
import pandas as pd

from sdm.data import load_dataset, sha256_of
from sdm.evaluate import median_spacing, nearest_distances
from sdm.paths import REPO_ROOT
from sdm.runs import PACKAGES, git_state, resolve_run
from sdm.simplex import FEATURES, check_weights

JUDGE = '20261029-171251-train-method-blend-stack'
MANIFEST = REPO_ROOT/'analysis/method_expansion/delivery/discovery_manifest.json'


def read(path):
    def invalid(value):
        raise ValueError(f'Nonstandard JSON {value} in {path}')
    return json.loads(path.read_text(), parse_constant=invalid)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def predict(weights_path):
    command = [str(Path(sys.executable).with_name('sdm')), 'predict',
               '--model-run', JUDGE, '--weights', str(weights_path.resolve())]
    result = subprocess.run(command, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    return command, result.stdout, result.stderr


def tables(inputs, predictions):
    weights = check_weights(inputs['weights'])
    rows = pd.DataFrame(inputs['rows'])
    assert inputs['feature_order'] == list(FEATURES)
    assert len(rows) == len(weights) == len(predictions['predicted_mean']) == 1503
    assert rows.point_index.tolist() == list(range(len(rows)))
    assert len(predictions['predicted_std']) == len(rows)
    rows['judge_mean'] = predictions['predicted_mean']
    rows['judge_proxy'] = predictions['predicted_std']
    assert np.isfinite(rows[['judge_mean', 'judge_proxy']]).all().all()
    assert (rows.judge_proxy >= 0).all()
    rows['judge_mean_plus_proxy'] = rows.judge_mean + rows.judge_proxy
    baseline = rows[rows.label == 'course-baseline'].iloc[0]
    rows['delta_vs_baseline'] = rows.judge_mean - baseline.judge_mean
    rows['reduction_pct'] = -100*rows.delta_vs_baseline/baseline.judge_mean
    rows['risk_delta_vs_baseline'] = rows.judge_mean_plus_proxy-baseline.judge_mean_plus_proxy
    rows['below_baseline'] = rows.delta_vs_baseline < 0
    proposals = rows[rows.kind == 'proposal']
    assert (proposals.distance_ratio <= .750001).all()
    np.testing.assert_allclose(proposals.nearest_distance, proposals.source_nearest_distance,
                               rtol=1e-7, atol=1e-9)
    summaries = []
    for label, group in rows[rows.kind == 'proposal'].groupby('label', sort=True):
        assert len(group) == 100 and group.groupby('optimizer_seed').size().to_dict() == {2026:50,2027:50}
        assert all(sorted(g['rank']) == list(range(1,51)) for _,g in group.groupby('optimizer_seed'))
        for _,g in group.groupby('optimizer_seed'):
            original_order = g.sort_values('rank')
            assert (np.diff(original_order.source_mean + original_order.source_proxy) >= -1e-9).all()
        first = group[(group.optimizer_seed == 2026) & (group['rank'] == 1)].iloc[0]
        second = group[(group.optimizer_seed == 2027) & (group['rank'] == 1)].iloc[0]
        best = group.sort_values(['judge_mean', 'optimizer_seed', 'rank'], kind='stable').iloc[0]
        top_mean = (first.judge_mean+second.judge_mean)/2
        summaries.append({
            'label':label, 'candidate_count':len(group),
            'primary_run':first.source_run, 'primary_rank':1, 'primary_point_index':int(first.point_index),
            'primary_judge_mean':first.judge_mean, 'primary_judge_proxy':first.judge_proxy,
            'primary_delta':first.delta_vs_baseline, 'primary_reduction_pct':first.reduction_pct,
            'primary_risk_delta':first.risk_delta_vs_baseline, 'primary_distance_ratio':first.distance_ratio,
            'second_top1_run':second.source_run, 'second_top1_judge_mean':second.judge_mean,
            'second_top1_delta':second.delta_vs_baseline,
            'two_seed_top1_mean':top_mean, 'two_seed_top1_delta':top_mean-baseline.judge_mean,
            'best100_run':best.source_run, 'best100_rank':int(best['rank']),
            'best100_point_index':int(best.point_index), 'best100_judge_mean':best.judge_mean,
            'best100_judge_proxy':best.judge_proxy, 'best100_delta':best.delta_vs_baseline,
            'best100_reduction_pct':best.reduction_pct, 'best100_risk_delta':best.risk_delta_vs_baseline,
            'best100_distance_ratio':best.distance_ratio,
            'candidate_mean':group.judge_mean.mean(), 'candidate_median':group.judge_mean.median(),
            'candidate_max':group.judge_mean.max(), 'below_baseline_count':int(group.below_baseline.sum()),
            'below_baseline_fraction':group.below_baseline.mean(),
        })
    summary = pd.DataFrame(summaries).sort_values(['primary_judge_mean','label'], kind='stable').reset_index(drop=True)
    assert len(summary) == 15
    return rows, summary


def build(out):
    out.mkdir(parents=True, exist_ok=False)
    dataset = load_dataset(REPO_ROOT/'dataset_same8')
    cases = read(MANIFEST)['cases']
    assert len(cases) == 15
    paths = [MANIFEST, REPO_ROOT/'weights.json', REPO_ROOT/'analysis/offline_submission/submission/weights.json']
    evidence = {}

    def record(ref):
        run = resolve_run(ref)
        meta = run.read_json('meta.json')
        assert meta['dataset']['sha256'] == dataset.sha256 and not meta['git']['dirty']
        evidence[ref] = {'meta':meta, 'config':run.config_path.read_text(),
                         'metrics':run.read_json('metrics.json')}
        paths.extend([run.path/'meta.json',run.config_path,run.path/'metrics.json'])
        return run

    judge = record(JUDGE)
    paths.append(judge.path/'model.pkl')
    points, entries = [], []
    for label, path in [('course-baseline',REPO_ROOT/'weights.json'),
                        ('operational-recommendation',REPO_ROOT/'analysis/offline_submission/submission/weights.json')]:
        points.append(check_weights(read(path))[0])
        entries.append({'label':label, 'kind':'reference', 'source_run':str(path.relative_to(REPO_ROOT)),
                        'rank':0, 'optimizer_seed':0, 'measured_R':None})
    observed = int(np.argmin(dataset.y))
    points.append(dataset.X[observed])
    entries.append({'label':'best-observed-p01969', 'kind':'reference',
                    'source_run':str(dataset.ids[observed]), 'rank':0, 'optimizer_seed':0,
                    'measured_R':float(dataset.y[observed])})
    assert dataset.ids[observed] == 'p01969'
    for case in cases:
        for ref in case['optimize']:
            run = record(ref)
            config = tomllib.loads(run.config_path.read_text())
            seed = int(config['optimizer']['params']['seed'])
            proposals = run.read_json('proposals.json')
            assert len(proposals) == 50 and seed in (2026,2027)
            for rank, proposal in enumerate(proposals, 1):
                points.append(check_weights(proposal['weights'])[0])
                entries.append({'label':case['label'], 'kind':'proposal', 'source_run':ref,
                    'rank':rank, 'optimizer_seed':seed, 'measured_R':None,
                    'source_mean':proposal['predicted_mean'], 'source_proxy':proposal['predicted_std'],
                    'source_nearest_distance':proposal['nearest_train_distance']})
            paths.append(run.path/'proposals.json')
    weights = check_weights(np.asarray(points))
    spacing = median_spacing(dataset.X)
    distances = nearest_distances(weights, dataset.X)
    for i,entry in enumerate(entries):
        entry.update({'point_index':i, 'nearest_distance':float(distances[i]),
                      'distance_ratio':float(distances[i]/spacing)})
    inputs = {'judge_run':JUDGE, 'feature_order':list(FEATURES),
              'rows':entries, 'weights':weights.reshape(-1,3,7).tolist()}
    write(out/'inputs.json', inputs)
    # Keep an ordinary CLI-compatible weights file separate from metadata.
    (out/'weights.json').write_text(json.dumps(inputs['weights'],separators=(',',':'),allow_nan=False)+'\n')
    command, stdout, stderr = predict(out/'weights.json')
    (out/'cli_predictions.json').write_text(stdout)
    (out/'cli_stderr.txt').write_text(stderr)
    rows, summary = tables(inputs, read(out/'cli_predictions.json'))
    rows.to_csv(out/'candidate_comparison.csv',index=False)
    summary.to_csv(out/'method_summary.csv',index=False)
    rows[rows.kind=='reference'].to_csv(out/'reference_comparison.csv',index=False)
    selected = {row.label:inputs['weights'][int(row.best100_point_index)] for row in summary.itertuples()}
    write(out/'posthoc_best_weights.json',selected)
    selected = {row.label:inputs['weights'][int(row.primary_point_index)] for row in summary.itertuples()}
    write(out/'primary_weights.json',selected)
    write(out/'run_evidence.json',evidence)
    write(out/'provenance.json', {'git':git_state(), 'dataset_sha256':dataset.sha256,
        'python':platform.python_version(), 'packages':{p:version(p) for p in (*PACKAGES,'catboost')},
        'judge_run':JUDGE, 'judge_refit':False, 'model_seed':2026, 'optimizer_seeds':[2026,2027],
        'prediction_command':command, 'source_hashes':{
            str(p.relative_to(REPO_ROOT)):sha256_of(p) for p in paths},
        'raw_weights_unchanged':True, 'simulator_calls':0,
        'limitation':'Post-search surrogate comparison; the judge is not independent of candidate generation.'})
    print(summary[['label','primary_judge_mean','primary_delta','best100_judge_mean','best100_delta']].to_string(index=False))


def verify(out, replay):
    checksums = out/'checksums.json'
    if checksums.exists():
        for name,digest in read(checksums).items():
            assert sha256_of(out/name) == digest, name
    for path in out.glob('*.json'): read(path)
    inputs = read(out/'inputs.json')
    assert inputs['judge_run'] == JUDGE
    np.testing.assert_array_equal(check_weights(inputs['weights']), check_weights(read(out/'weights.json')))
    rows, summary = tables(inputs, read(out/'cli_predictions.json'))
    for actual,path in [(rows,'candidate_comparison.csv'), (summary,'method_summary.csv'),
                        (rows[rows.kind=='reference'],'reference_comparison.csv')]:
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), pd.read_csv(out/path).reset_index(drop=True),
                                      check_dtype=False, rtol=1e-12, atol=1e-12)
    for name,index in [('primary_weights.json','primary_point_index'), ('posthoc_best_weights.json','best100_point_index')]:
        exported = read(out/name)
        for row in summary.to_dict('records'):
            np.testing.assert_array_equal(check_weights(exported[row['label']]),
                check_weights(inputs['weights'][int(row[index])]))
    source_count = 0
    for run,evidence in read(out/'run_evidence.json').items():
        for key in ('meta','metrics'): assert isinstance(evidence[key],dict)
        if run != JUDGE:
            own = rows[rows.source_run==run].sort_values('rank')
            assert len(own)==50
            source_count += len(own)
    assert source_count==1500
    if (out/'baseline_weights.json').exists():
        np.testing.assert_array_equal(check_weights(read(out/'baseline_weights.json')),
                                      check_weights(read(REPO_ROOT/'weights.json')))
        assert (out/'operational_weights.json').read_bytes() == (
            REPO_ROOT/'analysis/offline_submission/submission/weights.json').read_bytes()
        for group,column in [('primary','primary_point_index'),('posthoc','best100_point_index')]:
            for row in summary.to_dict('records'):
                np.testing.assert_array_equal(check_weights(read(out/'exported_weights'/group/f"{row['label']}.json")),
                    check_weights(inputs['weights'][int(row[column])]))
    if replay:
        for ref,part in rows[rows.kind=='proposal'].groupby('source_run'):
            original = resolve_run(ref).read_json('proposals.json')
            for row in part.to_dict('records'):
                np.testing.assert_array_equal(check_weights(original[int(row['rank'])-1]['weights']),
                    check_weights(inputs['weights'][int(row['point_index'])]))
        _,stdout,_ = predict(out/'weights.json')
        actual=json.loads(stdout); expected=read(out/'cli_predictions.json')
        for key in expected: np.testing.assert_allclose(actual[key],expected[key],rtol=1e-9,atol=1e-9)
    print(json.dumps({'verified':True, 'candidate_count':1500, 'reference_count':3,
        'methods':15, 'same_judge':JUDGE, 'cli_replay':replay, 'simulator_calls':0},indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['build','verify'])
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--replay-cli',action='store_true')
    args = parser.parse_args()
    if args.phase=='build': build(args.out)
    else: verify(args.out,args.replay_cli)


if __name__=='__main__': main()
