"""Archive CLI retraining evidence and compare predictions without refitting or retuning."""
from __future__ import annotations

import argparse
import gzip
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from sdm.data import load_dataset, sha256_of
from sdm.paths import REPO_ROOT
from sdm.runs import git_state, resolve_run
from sdm.simplex import check_weights

OLD = {
    'ridge_eval': '20261029-125034-evaluate-baseline-v32',
    'xgb_eval': '20261029-160333-evaluate-soft-xgboost-step-2026',
    'stack_eval': '20261029-171154-evaluate-method-blend-stack',
    'stack_train': 'artifacts/best_model',
    'consensus_train': '20261029-132622-train-consensus-r075-2026',
    'consensus_search': '20261029-132648-optimize-consensus-r075-2026',
}
KEY_METRICS = ('v32_score', 'mae', 'v32_local_mae', 'v32_top1_regret')


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def table(frame):
    return '\n'.join(['| ' + ' | '.join(frame.columns) + ' |',
                      '|' + '|'.join(['---'] * len(frame.columns)) + '|'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |'
                      for row in frame.itertuples(index=False, name=None)])


def compare(new, out):
    out.mkdir(parents=True, exist_ok=False)
    ds = load_dataset(REPO_ROOT / 'dataset_same8')
    evidence, input_paths = {}, [Path(__file__), Path(__file__).with_name('README.md')]

    def record(label, ref):
        run = resolve_run(ref)
        meta = run.read_json('meta.json')
        assert meta['dataset']['sha256'] == ds.sha256
        assert not meta['git']['dirty']
        evidence[label] = {'reference': ref, 'meta': meta,
                           'metrics': run.read_json('metrics.json'),
                           'config': run.config_path.read_text()}
        input_paths.extend(run.path / name for name in
                           ('meta.json', 'metrics.json', 'config.toml', 'model.pkl',
                            'weights.json', 'predictions.csv') if (run.path / name).exists())
        return run

    old_runs = {k: record('old_' + k, v) for k, v in OLD.items()}
    new_runs = {k: record('new_' + k, v) for k, v in new.items()}
    metric_rows, fold_rows = [], []
    for label in ('ridge_eval', 'xgb_eval', 'stack_eval'):
        before = old_runs[label].read_json('metrics.json')
        after = new_runs[label].read_json('metrics.json')
        for metric in KEY_METRICS:
            metric_rows.append({'model': label, 'metric': metric, 'old': before[metric],
                                'new': after[metric], 'delta': after[metric] - before[metric],
                                'new_run': new[label], 'old_run': OLD[label]})
        keys = ['split_id', 'candidate_id']
        a = pd.read_csv(old_runs[label].path / 'predictions.csv').sort_values(keys).reset_index(drop=True)
        b = pd.read_csv(new_runs[label].path / 'predictions.csv').sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(a[keys], b[keys])
        np.testing.assert_array_equal(a.actual, b.actual)
        for column in ('predicted_mean', 'predicted_std'):
            fold_rows.append({'model': label, 'column': column, 'rows': len(a),
                              'max_abs_difference': float(np.max(abs(a[column] - b[column])))})
        b.to_csv(out / f'{label}-outer-predictions.csv.gz', index=False, compression='gzip')
    metric_frame = pd.DataFrame(metric_rows)
    metric_frame.to_csv(out / 'metric_comparison.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(out / 'outer_prediction_comparison.csv', index=False)

    original_inputs = REPO_ROOT / 'analysis/common_judge/delivery/inputs.json'
    challenger_path = REPO_ROOT / 'analysis/robust_search/delivery/challenger_weights.json'
    inputs = json.loads(original_inputs.read_text())
    old_pool = check_weights(inputs['weights'])
    challenger = check_weights(json.loads(challenger_path.read_text()))
    source_new = check_weights(new_runs['consensus_search'].read_json('weights.json'))
    source_old = check_weights(old_runs['consensus_search'].read_json('weights.json'))
    assert source_new.shape == source_old.shape == (50, 21)
    points = np.vstack([old_pool, challenger, source_new])
    assert len(points) == 1554
    input_paths.extend([original_inputs, challenger_path])
    payload = json.dumps(points.tolist(), separators=(',', ':'))
    with gzip.open(out / 'comparison-weights.json.gz', 'wt') as f:
        f.write(payload)
    predictions, commands = {}, []
    for name, run in [('old_stack', old_runs['stack_train']), ('new_stack', new_runs['stack_train']),
                      ('old_consensus', old_runs['consensus_train']),
                      ('new_consensus', new_runs['consensus_train'])]:
        cmd = [str(Path(sys.executable).with_name('sdm')), 'predict',
               '--model-run', str(run.path), '--weights', '-']
        result = subprocess.run(cmd, input=payload, capture_output=True, text=True,
                                cwd=REPO_ROOT, check=True)
        predictions[name] = json.loads(result.stdout)
        write(out / f'predictions-{name}.json', predictions[name])
        commands.append({'command': cmd, 'stdin_sha256': hashlib.sha256(payload.encode()).hexdigest(),
                         'stderr': result.stderr})
    checks = []
    for family in ('stack', 'consensus'):
        for column in ('predicted_mean', 'predicted_std'):
            a = np.asarray(predictions['old_' + family][column])
            b = np.asarray(predictions['new_' + family][column])
            checks.append({'model': family, 'column': column, 'points': len(a),
                           'max_abs_difference': float(np.max(abs(a - b)))})
    prediction_checks = pd.DataFrame(checks)
    prediction_checks.to_csv(out / 'prediction_comparison.csv', index=False)
    names = [''] * len(points)
    for i, row in enumerate(inputs['rows']):
        names[i] = f"{row['label']}:s{row['optimizer_seed']}:r{row['rank']}"
    names[0], names[1], names[2], names[1503] = 'baseline', 'incumbent', 'p01969', 'challenger'
    for i in range(50):
        names[1504 + i] = f'new-consensus-rank{i+1}'
    candidates = pd.DataFrame({'candidate': names, 'point_index': np.arange(len(points))})
    for label, values in predictions.items():
        candidates[label + '_mean'] = values['predicted_mean']
        candidates[label + '_proxy'] = values['predicted_std']
        candidates[label + '_risk'] = candidates[label + '_mean'] + candidates[label + '_proxy']
    candidates.to_csv(out / 'candidate_predictions.csv', index=False)
    selected = candidates[candidates.candidate.isin([
        'baseline', 'incumbent', 'p01969', 'challenger', 'blend-fixed:s2026:r1',
        'blend-fixed:s2027:r1', 'blend-fixed:s2027:r2', 'blend-stack:s2027:r2',
        'new-consensus-rank1'])]
    selected.to_csv(out / 'shortlist_predictions.csv', index=False)
    write(out / 'new_consensus_rank1.json', source_new[0].reshape(3, 7).tolist())
    write(out / 'new_consensus_proposals.json', new_runs['consensus_search'].read_json('proposals.json'))

    bundle = out / 'retrained_stack'
    bundle.mkdir()
    for name in ('model.pkl', 'config.toml', 'meta.json', 'metrics.json'):
        shutil.copyfile(new_runs['stack_train'].path / name, bundle / name)
    (bundle / 'README.md').write_text('''# Retrained selected stack

This is the new all-data fit, not an overwrite of artifacts/best_model. Validation belongs to
separately fitted outer-fold counterparts. Load through the existing CLI from repository root:

```sh
env -u PYTHONPATH uv run --locked --group gui sdm predict \\
  --model-run analysis/retrain_comparison/delivery/retrained_stack \\
  --weights analysis/robust_search/delivery/challenger_weights.json
```

The parent report provides exact run IDs, evaluation, comparisons and limitations.
''')
    # Verify that the transferable bundle loads independently of its source run directory.
    cmd = [str(Path(sys.executable).with_name('sdm')), 'predict',
           '--model-run', str(bundle), '--weights', str(challenger_path)]
    output = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=REPO_ROOT)
    bundle_prediction = json.loads(output.stdout)
    expected = predictions['new_stack']['predicted_mean'][1503]
    np.testing.assert_allclose(bundle_prediction['predicted_mean'], [expected], rtol=1e-12, atol=1e-12)
    commands.append({'command': cmd, 'stderr': output.stderr})
    verification = {
        'dataset_rows': len(ds.y), 'new_measured_rows': 0,
        'max_metric_abs_difference': float(metric_frame.delta.abs().max()),
        'max_outer_prediction_abs_difference': max(r['max_abs_difference'] for r in fold_rows),
        'max_candidate_prediction_abs_difference': float(prediction_checks.max_abs_difference.max()),
        'max_consensus_proposal_weight_abs_difference': float(np.max(abs(source_new - source_old))),
        'new_rank1_vs_old_export_max_abs_difference': float(np.max(abs(source_new[0] - challenger[0]))),
        'bundle_cli_prediction_matches': True,
        'old_model_sha256': sha256_of(old_runs['stack_train'].path / 'model.pkl'),
        'new_model_sha256': sha256_of(bundle / 'model.pkl'), 'simulator_calls': 0,
    }
    write(out / 'verification.json', verification)
    write(out / 'run_evidence.json', evidence)
    write(out / 'run_manifest.json', {'new': new, 'old': OLD})
    write(out / 'provenance.json', {'git': git_state(), 'dataset_sha256': ds.sha256,
                                  'packages': {p: version(p) for p in
                                      ('numpy', 'pandas', 'scikit-learn', 'xgboost', 'catboost')},
                                  'input_hashes': {str(p.resolve()): sha256_of(p) for p in input_paths},
                                  'prediction_commands': commands, 'simulator_calls': 0})
    display = metric_frame.pivot(index='model', columns='metric', values='new').reset_index()
    for column in KEY_METRICS:
        display[column] = display[column].map(lambda x: f'{x:.4f}')
    shortlist = selected[['candidate', 'old_stack_mean', 'new_stack_mean', 'new_consensus_risk']].copy()
    for col in shortlist.columns[1:]:
        shortlist[col] = shortlist[col].map(lambda x: f'{x:.4f}')
    report = f'''# Selected-model retraining and consensus search comparison

Repeated six CLI experiments on unchanged2,048 observations and frozen-v32. No new measured
labels, retuning, scorer changes or simulator queries. New run IDs are in run_manifest.json;
run_evidence.json preserves every config and metadata. The original published model is unchanged.

## Fresh model evaluation

{table(display)}

The maximum old/new metric difference is {verification['max_metric_abs_difference']:.12g};
the maximum aligned outer-prediction difference is
{verification['max_outer_prediction_abs_difference']:.12g}. Global MAE is the CLI `mae` metric.
These results reproduce earlier evidence; they are not an additional improvement or independent test.

## Fresh training and weight search

The stack all-data fit relearns the same three convex coefficients (see retrained_stack/metrics.json).
The consensus model is separately retrained and searched at risk1, radius0.75, seed2026,50 proposals.
Maximum old/new difference across all50 source proposals is
{verification['max_consensus_proposal_weight_abs_difference']:.12g}. New rank1 versus the old
eight-decimal exported challenger differs by at most
{verification['new_rank1_vs_old_export_max_abs_difference']:.12g}; distinguish export rounding
from a new strategy. The fresh stack and fresh consensus are compared with their old fitted
counterparts on1,554 inputs, maximum prediction/proxy difference
{verification['max_candidate_prediction_abs_difference']:.12g}.

{table(shortlist)}

All R values are surrogate predictions. Consensus risk is max(member mean+proxy), not actual R.
Because the data/settings are unchanged, retraining is expected to reproduce the old result.
This does not establish any new true-R gain or settle the remaining model disagreement.

## Deliverables

- retrained_stack/model.pkl: transferable fresh model bundle, CLI-load verification passed.
- new_consensus_rank1.json: raw-precision new first proposal; the prior operational W is unchanged.
- metric_comparison.csv, outer_prediction_comparison.csv: old/new validation evidence.
- candidate_predictions.csv: all1,554 inputs, old/new stack and consensus means and proxies.
- comparison-weights.json.gz, prediction JSONs: exact CLI batch inputs/outputs.
- verification.json, provenance.json, run_evidence.json: numerical checks and source traceability.

Source model hashes can differ even when predictions match, because serialized estimator metadata
may vary. Prediction agreement, rather than byte equality, is the relevant reproducibility check.
No new model is declared superior solely because it was fitted later.
'''
    (out / 'README.md').write_text(report)
    write(out / 'checksums.json', {str(p.relative_to(out)): sha256_of(p)
                                  for p in sorted(out.rglob('*')) if p.is_file()})
    print(json.dumps(verification, indent=2))
    print(table(display))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in OLD:
        parser.add_argument('--' + name.replace('_', '-'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    compare({k: getattr(args, k) for k in OLD}, args.out.resolve())
