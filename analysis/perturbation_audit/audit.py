"""Audit candidate-neighborhood stability using shared feasible perturbations and CLI judges."""
from __future__ import annotations

import argparse
import gzip
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd

from sdm.data import load_dataset, sha256_of
from sdm.paths import REPO_ROOT
from sdm.runs import git_state, resolve_run
from sdm.simplex import FEATURES, check_weights, project

JUDGES = {
    'stack': 'artifacts/best_model',
    'histgb': '20261029-125052-optimize-histgb-v32',
    'xgboost': '20261029-125052-optimize-xgboost-v32',
    'extratrees': '20261029-125052-optimize-extratrees-default',
}
CORE = ('stack', 'histgb', 'xgboost')
RADII = (.02, .05)
AUDIT_SEEDS = (2027, 2028, 2029)
SHORTLIST = {
    'challenger': ('consensus-r075-2026', 2026, 1),
    'fixed-2027-r1': ('blend-fixed', 2027, 1),
    'stack-2027-r2': ('blend-stack', 2027, 2),
    'incumbent': ('operational-recommendation', 0, 0),
    'cat-2026-r4': ('cat-rmse', 2026, 4),
    'fixed-2026-r1': ('blend-fixed', 2026, 1),
}
HTML_CLAIMS = {
    'challenger': (-6.76, -9.02, -9.67, -4.45, .82, -3.39, .62),
    'fixed-2027-r1': (-9.01, -7.36, -15.52, -3.30, .57, -2.53, .35),
    'stack-2027-r2': (-7.87, -7.35, -11.95, -3.41, .62, -2.90, .48),
    'incumbent': (-6.10, -7.84, -8.08, -3.47, .65, -2.47, .32),
    'cat-2026-r4': (-5.42, -5.83, -6.14, -1.97, .20, -.01, .02),
}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def directions(seed, n):
    raw = np.random.default_rng(seed).normal(size=(n, 3, 7))
    raw -= raw.mean(axis=2, keepdims=True)
    raw = raw.reshape(n, 21)
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


def perturb(center, unit, radius, method):
    """Return feasible neighbors; exact-project corrects the final, rather than nominal, radius."""
    center = check_weights(center)[0]
    if radius <= 0 or unit.ndim != 2 or unit.shape[1] != 21:
        raise ValueError('positive radius and (n,21) directions required')
    if method == 'clip-normalize':
        points = np.maximum(center + radius * unit, 0).reshape(-1, 3, 7)
        points /= points.sum(axis=2, keepdims=True)
        return check_weights(points)
    if method == 'project-nominal':
        return check_weights(project(center + radius * unit))
    if method != 'project-exact':
        raise ValueError(method)
    # A vertex's outward normal cone can project an entire ray back to the vertex.
    # Reverse only such stationary rays rather than inventing zero-length neighbors.
    unit = unit.copy()
    stationary = np.linalg.norm(project(center + radius * unit) - center, axis=1) < 1e-12
    unit[stationary] *= -1
    lo = np.zeros(len(unit))
    hi = np.full(len(unit), radius)
    for _ in range(24):
        distance = np.linalg.norm(project(center + hi[:, None] * unit) - center, axis=1)
        expand = distance < radius
        if not expand.any():
            break
        hi[expand] *= 2
    else:
        raise ValueError('direction cannot reach requested distance')
    for _ in range(42):
        mid = (lo + hi) / 2
        distance = np.linalg.norm(project(center + mid[:, None] * unit) - center, axis=1)
        small = distance < radius
        lo = np.where(small, mid, lo)
        hi = np.where(small, hi, mid)
    points = check_weights(project(center + ((lo + hi) / 2)[:, None] * unit))
    np.testing.assert_allclose(np.linalg.norm(points - center, axis=1), radius, atol=1e-10, rtol=0)
    return points


def worst_difference(predictions, reference):
    return np.max(np.asarray(predictions) - np.asarray(reference), axis=1)


def cli_predict(points, label, out, judges=CORE):
    payload = json.dumps(check_weights(points).tolist(), separators=(',', ':'))
    result = {}
    commands = []
    for name in judges:
        cmd = [str(Path(sys.executable).with_name('sdm')), 'predict',
               '--model-run', JUDGES[name], '--weights', '-']
        run = subprocess.run(cmd, input=payload, cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True)
        values = json.loads(run.stdout)
        mean = np.asarray(values['predicted_mean'])
        if mean.shape != (len(points),) or not np.isfinite(mean).all():
            raise ValueError(f'invalid predictions from {name}')
        result[name] = mean
        with gzip.open(out / f'{label}-{name}.json.gz', 'wt') as f:
            json.dump(values, f, separators=(',', ':'), allow_nan=False)
        commands.append({'judge': name, 'command': cmd,
                         'stdin_sha256': hashlib.sha256(payload.encode()).hexdigest(),
                         'stderr': run.stderr})
    with gzip.open(out / f'{label}-weights.json.gz', 'wt') as f:
        f.write(payload)
    return result, commands


def markdown(frame):
    # Avoid an extra table-formatting dependency.
    return '\n'.join(['| ' + ' | '.join(frame.columns) + ' |',
                      '|' + '|'.join(['---'] * len(frame.columns)) + '|'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |'
                      for row in frame.itertuples(index=False, name=None)])


def run(html, out):
    out.mkdir(parents=True, exist_ok=False)
    sources = [REPO_ROOT / 'analysis/common_judge/delivery/inputs.json',
               REPO_ROOT / 'analysis/robust_search/delivery/challenger_weights.json', html,
               Path(__file__), Path(__file__).with_name('README.md')]
    inputs = json.loads(sources[0].read_text())
    assert inputs['feature_order'] == list(FEATURES)
    points = check_weights(inputs['weights'])
    entries = pd.DataFrame(inputs['rows'])
    challenger = check_weights(json.loads(sources[1].read_text()))
    points = np.vstack([points, challenger])
    entries = pd.concat([entries, pd.DataFrame([{
        'label': 'consensus-r075-2026', 'kind': 'proposal',
        'source_run': '20261029-132648-optimize-consensus-r075-2026',
        'rank': 1, 'optimizer_seed': 2026, 'distance_ratio': .6381709229164247,
    }])], ignore_index=True)
    entries['pool_index'] = np.arange(len(points))
    assert len(points) == 1504
    reference_index = int(entries.index[entries.label == 'best-observed-p01969'][0])
    commands = []
    predicted, cmds = cli_predict(points, 'pool', out, judges=tuple(JUDGES))
    commands.extend(cmds)
    reference = np.array([predicted[name][reference_index] for name in CORE])
    for name in JUDGES:
        entries[f'{name}_mean'] = predicted[name]
        entries[f'{name}_delta'] = predicted[name] - predicted[name][reference_index]
    entries['worst_delta'] = entries[[f'{name}_delta' for name in CORE]].max(axis=1)
    index_names = {}
    for name, (label, seed, rank) in SHORTLIST.items():
        found = entries.index[(entries.label == label) & (entries.optimizer_seed == seed)
                              & (entries['rank'] == rank)].tolist()
        assert len(found) == 1, (name, found)
        index_names[found[0]] = name
    checks = []
    for index, name in index_names.items():
        if name not in HTML_CLAIMS:
            continue
        actual = entries.loc[index, [f'{x}_delta' for x in CORE]].to_numpy(dtype=float)
        expected = np.array(HTML_CLAIMS[name][:3])
        for judge, value, claim in zip(CORE, actual, expected):
            checks.append({'candidate': name, 'judge': judge, 'delta': value,
                           'html_delta': claim, 'abs_difference': abs(value - claim),
                           'matches_display_precision': bool(abs(value - claim) <= .006)})
    check_frame = pd.DataFrame(checks)
    check_frame.to_csv(out / 'html_single_point_checks.csv', index=False)
    reference_check = np.max(abs(reference - np.array([118.18, 116.82, 116.95]))) <= .006
    if not check_frame.matches_display_precision.all() or not reference_check:
        raise ValueError('Archived models do not reproduce HTML single-point figures; stop before perturbing')
    screened = entries.index[(entries.kind == 'proposal') & (entries.worst_delta < -6)].tolist()
    panel = sorted(set(screened) | set(index_names))
    for index in panel:
        index_names.setdefault(index, f'pool-{index:04d}')
    entries['candidate'] = entries.pool_index.map(index_names).fillna('')
    entries['in_perturbation_panel'] = entries.pool_index.isin(panel)
    entries.to_csv(out / 'pool_scores.csv', index=False)
    write(out / 'panel.json', [{'candidate': index_names[i], 'pool_index': i,
                              'source_run': entries.loc[i, 'source_run'],
                              'source_rank': int(entries.loc[i, 'rank']),
                              'weights': points[i].reshape(3, 7).tolist()} for i in panel])
    print(f'HTML single-point predictions match. Panel: {len(panel)} candidates.', flush=True)
    all_samples = []
    settings = [('project-nominal', 2026, 40), ('clip-normalize', 2026, 40)]
    settings += [('project-exact', seed, 512) for seed in AUDIT_SEEDS]
    for method, seed, count in settings:
        unit = directions(seed, count)
        neighbors, metadata = [], []
        for index in panel:
            for radius in RADII:
                nearby = perturb(points[index], unit, radius, method)
                neighbors.append(nearby)
                metadata.extend({'candidate': index_names[index], 'pool_index': index,
                                 'method': method, 'seed': seed, 'radius': radius,
                                 'direction': j, 'actual_distance': float(distance)}
                                for j, distance in enumerate(np.linalg.norm(nearby - points[index], axis=1)))
        batch = np.vstack(neighbors)
        label = f'{method}-{seed}'
        preds, cmds = cli_predict(batch, label, out)
        commands.extend(cmds)
        sample = pd.DataFrame(metadata)
        for name in CORE:
            sample[f'{name}_mean'] = preds[name]
            sample[f'{name}_delta'] = preds[name] - predicted[name][reference_index]
        sample['worst_delta'] = worst_difference(np.column_stack([preds[x] for x in CORE]), reference)
        sample['success'] = sample.worst_delta < -3
        sample.to_csv(out / f'samples-{label}.csv.gz', index=False, compression='gzip')
        all_samples.append(sample)
        print(f'Completed {label}: {len(sample)} feasible neighbors.', flush=True)
    samples = pd.concat(all_samples, ignore_index=True)
    group = ['candidate', 'method', 'radius', 'seed']
    per_seed = summarize(samples, group)
    per_seed.to_csv(out / 'per_seed_summary.csv', index=False)
    summary = summarize(samples, group[:-1])
    summary.to_csv(out / 'summary.csv', index=False)
    pairs = []
    for radius in RADII:
        subset = samples[(samples.method == 'project-exact') & (samples.radius == radius)]
        reference_samples = subset[subset.candidate == 'challenger'].set_index(['seed', 'direction'])
        for candidate in sorted(set(subset.candidate) - {'challenger'}):
            comparison = subset[subset.candidate == candidate].set_index(['seed', 'direction'])
            comparison = comparison.loc[reference_samples.index]
            delta = comparison.success.astype(float).to_numpy() - reference_samples.success.astype(float).to_numpy()
            se = delta.std(ddof=1) / np.sqrt(len(delta))
            pairs.append({'candidate': candidate, 'radius': radius, 'n': len(delta),
                          'success_rate_minus_challenger': float(delta.mean()),
                          'conditional_normal_ci_low': float(delta.mean() - 1.96 * se),
                          'conditional_normal_ci_high': float(delta.mean() + 1.96 * se),
                          'candidate_only_success': int((delta == 1).sum()),
                          'challenger_only_success': int((delta == -1).sum())})
    pd.DataFrame(pairs).to_csv(out / 'paired_comparisons.csv', index=False)
    html_comparisons = []
    for name, claims in HTML_CLAIMS.items():
        for radius, offset in [(.02, 3), (.05, 5)]:
            for method in ('project-nominal', 'clip-normalize', 'project-exact'):
                r = summary[(summary.candidate == name) & (summary.radius == radius)
                            & (summary.method == method)].iloc[0]
                html_comparisons.append({'candidate': name, 'radius': radius, 'method': method,
                                         'html_median_delta': claims[offset],
                                         'html_success_fraction': claims[offset + 1],
                                         'audit_median_delta': r.median_delta,
                                         'audit_success_fraction': r.success_fraction})
    pd.DataFrame(html_comparisons).to_csv(out / 'html_neighborhood_comparison.csv', index=False)
    dataset = load_dataset(REPO_ROOT / 'dataset_same8')
    model_metadata = {}
    for name, ref in JUDGES.items():
        run_folder = resolve_run(ref)
        sources.append(run_folder.path / 'model.pkl')
        meta = run_folder.read_json('meta.json')
        assert meta['dataset']['sha256'] == dataset.sha256
        model_metadata[name] = {'reference': ref, 'meta': meta,
                                'config': run_folder.config_path.read_text()}
    write(out / 'provenance.json', {
        'analysis_id': '20261101-candidate-perturbation-audit-v1', 'git': git_state(),
        'python': platform.python_version(), 'packages': {p: version(p) for p in
            ('numpy', 'pandas', 'scikit-learn', 'xgboost', 'catboost')},
        'dataset_sha256': dataset.sha256, 'input_hashes': {str(p.resolve()): sha256_of(p) for p in sources},
        'reference_id': 'p01969', 'reference_predictions': dict(zip(CORE, reference.tolist())),
        'judges': model_metadata, 'commands': commands,
        'panel_count': len(panel), 'screened_below_minus6': len(screened),
        'neighbor_count': len(samples), 'simulator_calls': 0,
        'exact_original_reproduction': False,
        'limitation': 'Original HTML scripts, samples and Nov1 fits unavailable; independent reconstruction only.',
    })
    main = summary[summary.method == 'project-exact'].copy()
    main['rank'] = main.groupby('radius').success_fraction.rank(method='min', ascending=False)
    main.to_csv(out / 'main_ranking.csv', index=False)
    shortlist = main[main.candidate.isin(SHORTLIST)].sort_values(['radius', 'rank'])
    display = shortlist[['candidate', 'radius', 'success_fraction', 'median_delta', 'rank']].copy()
    for column in ('success_fraction', 'median_delta'):
        display[column] = display[column].map(lambda v: f'{v:.4f}')
    report = f'''# Candidate perturbation audit

Independent audit of the supplied HTML; no retraining or real-evaluator calls.
Single-point differences for all five HTML shortlist candidates match displayed values using the
archived judges. This does not recover the missing original direction generator or model binaries.

Panel: {len(panel)} fixed candidates, including every original proposal with worst-judge delta<-6,
the HTML shortlist, and original seed2026 fixed-blend rank1. Neighbor count: {len(samples)}.
Main audit: three new direction seeds,512 shared directions each, actual Euclidean radii0.02/0.05.
Success means all three core judges predict R at least3 lower than their own p01969 prediction.
All reported improvements below are predicted differences, never actual-R estimates.

## Main shortlist results

{markdown(display)}

Rank is within the complete screened panel, not only this shortlist. Lower median delta is better;
higher success fraction is better. See main_ranking.csv for every candidate, per_seed_summary.csv
for seed sensitivity, and paired_comparisons.csv for paired direction differences.

## Scope of the conclusion

The HTML's82%/62% cannot be claimed as exactly reproduced without its original scripts/samples.
html_neighborhood_comparison.csv compares them with both reconstructed40-direction nominal-step
methods and the larger actual-distance audit. summary.csv reports achieved distances as well.
Three core judges are correlated: the stack contains HistGB/XGBoost-family members. Relative
prediction to a known reference does not remove unknown differential surrogate error.
ExtraTrees dissent is retained in pool_scores.csv; it is not silently used or removed from the
three-judge endpoint. Real baseline R119.0916 mentioned in HTML is not verified by this experiment.

Training-subsample stability and input-neighborhood stability are different checks. This audit
does not alter the prior operational-W replacement gate. It can support a prioritized real test,
but cannot establish the true improvement, a true-R margin of3, or global optimality.

## Reproducibility

provenance.json records committed source, HTML/model/data hashes, packages and every CLI command.
panel.json identifies exact source ranks and weights. Compressed weights, predictions and sample
tables preserve every evaluated neighbor. No frozen data, model, candidate or scorer is modified.
'''
    (out / 'README.md').write_text(report)
    write(out / 'checksums.json', {str(p.relative_to(out)): sha256_of(p)
                                  for p in sorted(out.iterdir()) if p.is_file()})
    print(markdown(display), flush=True)


def summarize(samples, group):
    records = []
    for keys, frame in samples.groupby(group, sort=True):
        records.append({**dict(zip(group, keys)), 'n': len(frame),
                        'success_fraction': float(frame.success.mean()),
                        'median_delta': float(frame.worst_delta.median()),
                        'q90_delta': float(frame.worst_delta.quantile(.9)),
                        'median_actual_distance': float(frame.actual_distance.median()),
                        'min_actual_distance': float(frame.actual_distance.min()),
                        'max_actual_distance': float(frame.actual_distance.max())})
    return pd.DataFrame(records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--html', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.html.resolve(), args.out.resolve())
