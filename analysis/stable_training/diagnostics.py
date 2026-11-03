"""Supplementary measured-region and fresh-direction checks, separate from frozen-v32."""
from __future__ import annotations

import argparse
import gzip
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from sdm.config import load_config
from sdm.data import load_dataset, sha256_of
from sdm.evaluate import median_spacing, nearest_distances
from sdm.paths import REPO_ROOT
from sdm.registry import build_model
from sdm.runs import git_state, resolve_run
from sdm.simplex import check_weights
from sdm.optimizers._neighborhood import exact_neighbors, neighborhood_scores, shared_directions

BASE = REPO_ROOT / 'analysis/stable_training'
RESULTS = BASE / 'results/study'
OLD = REPO_ROOT / 'analysis/candidate_table_revision/delivery'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def provenance(path, ds, extra=()):
    sources = [Path(__file__), BASE/'README.md', RESULTS/'runs.json',
               RESULTS/'selection.json', REPO_ROOT/'src/sdm/models/stable_stack.py',
               REPO_ROOT/'src/sdm/optimizers/_neighborhood.py', *extra]
    write(path, {'git': git_state(), 'dataset_sha256': ds.sha256,
        'packages': {p: version(p) for p in ('numpy', 'pandas', 'scikit-learn', 'catboost', 'xgboost')},
        'sources': {str(p): sha256_of(p) for p in sources}, 'simulator_calls': 0,
        'seeds': {'regions': 2031, 'subsample_audit': list(range(2032, 2037)),
                  'directions': [2037, 2038, 2039]}})


class Chunked:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        values = [self.model.predict(X[i:i+512]) for i in range(0, len(X), 512)]
        return tuple(np.concatenate([v[k] for v in values]) for k in (0, 1))


def region_checks():
    out = RESULTS/'regions'
    out.mkdir(exist_ok=True)
    ds = load_dataset(REPO_ROOT/'dataset_same8')
    selection = json.loads((RESULTS/'selection.json').read_text())
    profile = selection['profile']
    configs = {name: load_config(REPO_ROOT/f'experiments/stable-{name}-2026.toml')
               for name in ('reference', profile, 'honest')}
    provenance(out/'provenance.json', ds)
    regions = KMeans(n_clusters=3, n_init=10, random_state=2031).fit_predict(ds.X)
    pd.DataFrame({'candidate_id': ds.ids, 'region': regions}).to_csv(out/'regions.csv', index=False)
    rows, predictions = [], []
    unit = shared_directions(2031, 16)
    for region in range(3):
        train, test = np.flatnonzero(regions != region), np.flatnonzero(regions == region)
        ratio = nearest_distances(ds.X[test], ds.X[train]) / median_spacing(ds.X[train])
        for name, cfg in configs.items():
            print(f'Region {region}, fit {name}: {len(train)} train/{len(test)} test', flush=True)
            model = build_model(cfg.model.name, cfg.model.params).fit(ds.X[train], ds.y[train])
            predictor = Chunked(model)
            mean, scale = predictor.predict(ds.X[test])
            tail, parts = neighborhood_scores(predictor, ds.X[test], radii=(.02, .05),
                directions=unit, risk=1., quantile=.8)
            for j, index in enumerate(test):
                predictions.append({'region': region, 'model': name, 'candidate_id': ds.ids[index],
                    'actual': float(ds.y[index]), 'mean': float(mean[j]), 'error_scale': float(scale[j]),
                    'neighborhood_objective': float(tail[j]), 'distance_ratio': float(ratio[j])})
            local = np.argsort(ds.y[test], kind='stable')[:max(1, len(test)//5)]
            for support, eligible in (('all', np.arange(len(test))),
                                      ('ratio<=.75', np.flatnonzero(ratio <= .75))):
                for rule, scores in (('mean', mean), ('mean+error', mean+scale), ('neighborhood', tail)):
                    row = {'region': region, 'model': name, 'support': support, 'rule': rule,
                           'pool_size': len(eligible), 'holdout_rows': len(test),
                           'mae': float(np.mean(abs(ds.y[test]-mean))),
                           'low20_mae': float(np.mean(abs(ds.y[test][local]-mean[local]))),
                           'coverage': float(np.mean(abs(ds.y[test]-mean) <= scale)),
                           'optimistic_error_q90': float(np.quantile(ds.y[test]-mean, .9))}
                    if len(eligible):
                        best = eligible[np.argmin(scores[eligible])]
                        row.update({'selected_id': ds.ids[test[best]],
                            'selected_actual_R': float(ds.y[test[best]]),
                            'pool_best_R': float(ds.y[test[eligible]].min()),
                            'regret': float(ds.y[test[best]]-ds.y[test[eligible]].min())})
                    rows.append(row)
            pd.DataFrame(rows).to_csv(out/'selection.csv', index=False)
            pd.DataFrame(predictions).to_csv(out/'predictions.csv.gz', index=False)
            print(f'Region {region}, done {name}', flush=True)


def candidate_checks():
    out = RESULTS/'candidates'
    out.mkdir(exist_ok=True)
    ds = load_dataset(REPO_ROOT/'dataset_same8')
    runs = json.loads((RESULTS/'runs.json').read_text())
    profile = json.loads((RESULTS/'selection.json').read_text())['profile']
    names, weights, sources = [], [], []
    for path in sorted((OLD/'weights').glob('*.json')):
        names.append(path.stem)
        weights.append(check_weights(json.loads(path.read_text()))[0])
        sources.append(str(path))
    for label, record in runs.items():
        if '-central-' not in label and '-neighbors-' not in label:
            continue
        run = resolve_run(record['run'])
        points = check_weights(run.read_json('weights.json'))
        for rank in (1, 2):
            names.append(label+f'-r{rank}')
            weights.append(points[rank-1])
            sources.append(run.id)
    points = check_weights(np.array(weights))
    refs = {'reference': runs['reference-fit']['run'], profile: runs[profile+'-fit']['run'],
            'honest': runs['honest-fit']['run'],
            'histgb': '20261029-125052-optimize-histgb-v32',
            'xgboost': '20261029-125052-optimize-xgboost-v32',
            'extratrees': '20261029-125052-optimize-extratrees-default'}
    provenance(out/'provenance.json', ds, [OLD/'decision_manifest.json'])
    models = {name: resolve_run(ref).load_model() for name, ref in refs.items()}
    write(out/'inputs.json', {'names': names, 'sources': sources, 'weights': points.tolist(),
                             'model_runs': refs})
    reference_index = names.index('control-p01969')
    center_predictions = {name: Chunked(model).predict(points) for name, model in models.items()}
    central = []
    for i, name in enumerate(names):
        row = {'candidate': name, 'source': sources[i],
               'distance_ratio': float(nearest_distances(points[i:i+1], ds.X)[0]/median_spacing(ds.X))}
        for model_name, (mean, scale) in center_predictions.items():
            row[model_name+'_mean'] = float(mean[i])
            row[model_name+'_error'] = float(scale[i])
            row[model_name+'_delta'] = float(mean[i]-mean[reference_index])
        central.append(row)
    pd.DataFrame(central).to_csv(out/'central_predictions.csv', index=False)
    neighbor_rows = []
    for radius in (.02, .05):
        for seed in (2037, 2038, 2039):
            neighbors = exact_neighbors(points, shared_directions(seed, 128), radius)
            flat = neighbors.reshape(-1, 21)
            predictions = {name: Chunked(model).predict(flat) for name, model in models.items()}
            with gzip.open(out/f'neighbors-{radius}-{seed}.json.gz', 'wt') as f:
                json.dump({'weights': neighbors.tolist(),
                    'predictions': {k: {'mean': v[0].tolist(), 'error': v[1].tolist()}
                                    for k, v in predictions.items()}}, f, separators=(',', ':'))
            for i, name in enumerate(names):
                row = {'candidate': name, 'radius': radius, 'seed': seed}
                delta = []
                for judge in ('reference', 'histgb', 'xgboost'):
                    mean = predictions[judge][0].reshape(len(points), -1)[i]
                    delta.append(mean-center_predictions[judge][0][reference_index])
                worst = np.max(delta, axis=0)
                row['legacy_consensus_success'] = float(np.mean(worst < -3))
                row['legacy_worst_delta_median'] = float(np.median(worst))
                for judge in ('reference', profile, 'honest'):
                    mean, scale = predictions[judge]
                    mean = mean.reshape(len(points), -1)[i]
                    scale = scale.reshape(len(points), -1)[i]
                    row[judge+'_delta_median'] = float(np.median(mean-center_predictions[judge][0][reference_index]))
                    row[judge+'_acquisition_q80'] = float(np.quantile(mean+scale, .8))
                neighbor_rows.append(row)
            print(f'Fresh candidate directions: radius {radius}, seed {seed}', flush=True)
            pd.DataFrame(neighbor_rows).to_csv(out/'neighbor_summary.csv', index=False)
    # Five *new* training-subset audits. For the new recipe, residual calibration cannot affect
    # its mean: _fit_core computes exactly the final predictive fit without needless calibration.
    subset_rows = []
    cfgs = {n: load_config(REPO_ROOT/f'experiments/stable-{n}-2026.toml')
            for n in ('reference', profile)}
    for seed in range(2032, 2037):
        subset = np.sort(np.random.default_rng(seed).choice(len(ds.y), int(.9*len(ds.y)), replace=False))
        for name, cfg in cfgs.items():
            model = build_model(cfg.model.name, cfg.model.params)
            print(f'Training-subset audit {seed}: {name}', flush=True)
            if name == profile:
                core = model._fit_core(ds.X[subset], ds.y[subset], model.params.seed)
                mean = core.predict(points)[0]
            else:
                mean = model.fit(ds.X[subset], ds.y[subset]).predict(points)[0]
            for i, candidate in enumerate(names):
                subset_rows.append({'seed': seed, 'model': name, 'candidate': candidate,
                    'predicted_mean': float(mean[i]), 'delta_to_p01969': float(mean[i]-mean[reference_index]),
                    'delta_to_incumbent': float(mean[i]-mean[names.index('incumbent')])})
            write(out/f'subset-{seed}.json', {'indices': subset.tolist(),
                'seed': seed, 'training_rows': len(subset), 'calibration_skipped_for_new_mean_only': True})
            pd.DataFrame(subset_rows).to_csv(out/'training_subset_predictions.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('regions', 'candidates'))
    args = parser.parse_args()
    region_checks() if args.phase == 'regions' else candidate_checks()
