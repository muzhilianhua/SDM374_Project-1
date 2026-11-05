"""Experimental v3.3 frozen tasks and optimizer-independent scoring (stdlib only)."""
import csv
import hashlib
import json
import math
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, '../..'))
PROTOCOL = 'sdm374.offline-breakthrough.v3.3.experimental.1'
DATA_SHA256 = '877b5f8ee265245dd693148a5d7b11f28f48b426e43ac3ac1be8b7a934249023'
POLICY = {
    'protocol_id': PROTOCOL,
    'status': 'EXPERIMENTAL_NOT_A_REPLACEMENT_FOR_V32',
    'test_n': 410,
    'tail_hidden_ranks': [5, 10, 20],
    'tail_seeds': [3301, 3302],
    'region_anchor_ranks': [1, 3, 5],
    'region_mandatory_hidden_ranks': 5,
    'inner_folds': 3,
    'inner_seed': 3303,
    'family_weights': {'hidden_elite': 0.5, 'local_region': 0.5},
    'selection': 'minimum predicted_mean, exact ties by candidate_id',
    'score': '100 * (train_best - selected_actual) / (train_best - test_best)',
    'score_clipping': 'none: losses remain negative; attainable maximum 100',
    'diagnostic_cutoff': 132.8,
    'p1_usage': 'external known failure only; excluded from all frozen fits and scores',
}


def sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def read_json(path):
    with open(path, encoding='utf-8') as stream:
        return json.load(stream)


def write_json(path, value):
    with open(path, 'w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def read_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with open(path, 'w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_data():
    path = os.path.join(ROOT, 'dataset_same8/train.csv')
    if sha256(path) != DATA_SHA256:
        raise ValueError('Official training dataset checksum mismatch')
    rows = read_csv(path)
    features = ['b{}_{}'.format(b, m) for b in range(1, 4)
                for m in ['cost', 'carbon', 'peak', 'discomfort', 'degradation', 'ramping', 'resilience']]
    data = {}
    for row in rows:
        cid = row['candidate_id']
        if cid in data:
            raise ValueError('Duplicate dataset ID')
        data[cid] = {'y': float(row['ramping']), 'x': [float(row[f]) for f in features]}
    return data


def hash_order(ids, salt):
    return sorted(ids, key=lambda cid: (hashlib.sha256((salt + ':' + cid).encode('utf-8')).hexdigest(), cid))


def create_manifest(data):
    ids = sorted(data)
    elite = sorted(ids, key=lambda cid: (data[cid]['y'], cid))
    specs = []
    for rank in POLICY['tail_hidden_ranks']:
        for seed in POLICY['tail_seeds']:
            hidden = elite[:rank]
            pool = hash_order([cid for cid in ids if cid not in hidden], 'tail:{}'.format(seed))
            test = sorted(hidden + pool[:POLICY['test_n'] - rank])
            specs.append(('elite-{:02d}-s{}'.format(rank, seed), 'hidden_elite', test,
                          {'hidden_best_n': rank, 'seed': seed}))
    for rank in POLICY['region_anchor_ranks']:
        anchor = elite[rank - 1]
        hidden = elite[:POLICY['region_mandatory_hidden_ranks']]
        distances = {cid: sum((a-b)**2 for a, b in zip(data[cid]['x'], data[anchor]['x'])) for cid in ids}
        neighbors = sorted([cid for cid in ids if cid not in hidden], key=lambda cid: (distances[cid], cid))
        test = sorted(hidden + neighbors[:POLICY['test_n'] - len(hidden)])
        specs.append(('region-rank{:02d}'.format(rank), 'local_region', test,
                      {'anchor_id': anchor, 'anchor_rank': rank,
                       'mandatory_best_n': len(hidden), 'distance': 'raw 21-weight Euclidean',
                       'maximum_test_anchor_distance': math.sqrt(max(distances[cid] for cid in test))}))
    tasks = []
    for split, family, test, design in specs:
        train = sorted(set(ids) - set(test))
        order = hash_order(train, 'inner:{}:{}'.format(POLICY['inner_seed'], split))
        inner = []
        for fold in range(POLICY['inner_folds']):
            validation = sorted(order[fold::POLICY['inner_folds']])
            inner.append({'fold': fold + 1, 'train_ids': sorted(set(train) - set(validation)),
                          'validation_ids': validation})
        tasks.append({'split_id': split, 'family': family, 'train_ids': train, 'test_ids': test,
                      'inner_splits': inner, 'design': design,
                      'train_best': min(data[cid]['y'] for cid in train),
                      'test_best': min(data[cid]['y'] for cid in test)})
    return {'protocol_id': PROTOCOL, 'dataset_sha256': DATA_SHA256,
            'policy': POLICY, 'canonical_candidate_ids': ids, 'tasks': tasks}


def validate_manifest(manifest, data):
    # Full deterministic reconstruction checks IDs, labels, geometry, inner folds and metadata.
    if manifest != create_manifest(data):
        raise ValueError('Manifest differs from deterministic experimental protocol')
    for task in manifest['tasks']:
        if task['train_best'] <= task['test_best']:
            raise ValueError('Every task must contain a genuine breakthrough opportunity')


def load_release():
    checksums = read_json(os.path.join(BASE, 'CHECKSUMS.json'))
    for name, expected in checksums.items():
        if sha256(os.path.join(BASE, name)) != expected:
            raise ValueError('Frozen v3.3 checksum mismatch: ' + name)
    if read_json(os.path.join(BASE, 'POLICY.json')) != POLICY:
        raise ValueError('Unexpected scoring policy')
    data = load_data()
    manifest = read_json(os.path.join(BASE, 'split_manifest.json'))
    validate_manifest(manifest, data)
    return data, manifest


def score_task(train_values, actual, predictions):
    """Input maps are ID-aligned. No optimizer, top-K cutoff, or uncertainty required."""
    if not train_values or not actual or set(actual) != set(predictions):
        raise ValueError('Nonempty train/test and exact prediction membership required')
    values = list(train_values) + list(actual.values()) + list(predictions.values())
    if not all(isinstance(v, (float, int)) and math.isfinite(v) for v in values):
        raise ValueError('Finite numeric labels and predictions required')
    baseline, oracle = min(train_values), min(actual.values())
    if baseline <= oracle:
        raise ValueError('No positive breakthrough opportunity')
    chosen = min(actual, key=lambda cid: (predictions[cid], cid))
    truth_best = min(actual, key=lambda cid: (actual[cid], cid))
    achieved = baseline - actual[chosen]
    error = predictions[chosen] - actual[chosen]
    low = [cid for cid in actual if actual[cid] <= POLICY['diagnostic_cutoff']]
    optimum_rank = 1 + sum((predictions[cid], cid) < (predictions[truth_best], truth_best) for cid in actual)
    return {'score': 100.0 * achieved / (baseline - oracle),
            'train_best': baseline, 'test_best': oracle, 'opportunity': baseline - oracle,
            'selected_id': chosen, 'selected_actual': actual[chosen], 'selected_prediction': predictions[chosen],
            'achieved_improvement': achieved, 'regret': actual[chosen] - oracle,
            'actual_breakthrough': actual[chosen] < baseline,
            'predicted_breakthrough': predictions[chosen] < baseline,
            'false_breakthrough': predictions[chosen] < baseline and actual[chosen] >= baseline,
            'selected_bias': error, 'true_best_id': truth_best, 'true_best_predicted_rank': optimum_rank,
            'true_best_bias': predictions[truth_best] - actual[truth_best],
            'global_mae': sum(abs(predictions[cid] - actual[cid]) for cid in actual) / len(actual),
            'low_n': len(low),
            'low_bias': sum(predictions[cid] - actual[cid] for cid in low) / len(low) if low else None}


def evaluate_rows(rows, manifest, data):
    grouped = {}
    for row in rows:
        key = (row['split_id'], row['candidate_id'])
        if key in grouped:
            raise ValueError('Duplicate prediction row: ' + str(key))
        grouped[key] = float(row['predicted_mean'])
    expected = {(task['split_id'], cid) for task in manifest['tasks'] for cid in task['test_ids']}
    if set(grouped) != expected:
        raise ValueError('Predictions must cover every exact frozen test ID once; missing={}, extra={}'.format(
            len(expected - set(grouped)), len(set(grouped) - expected)))
    scores = []
    for task in manifest['tasks']:
        result = score_task([data[cid]['y'] for cid in task['train_ids']],
                            {cid: data[cid]['y'] for cid in task['test_ids']},
                            {cid: grouped[(task['split_id'], cid)] for cid in task['test_ids']})
        result.update(split_id=task['split_id'], family=task['family'])
        scores.append(result)
    families = {}
    for family in POLICY['family_weights']:
        part = [r for r in scores if r['family'] == family]
        families[family] = {
            key: sum(float(r[key]) for r in part) / len(part)
            for key in ['score', 'regret', 'achieved_improvement', 'actual_breakthrough', 'false_breakthrough']}
        families[family]['worst_score'] = min(r['score'] for r in part)
        families[family]['tasks'] = len(part)
    total = sum(POLICY['family_weights'][f] * families[f]['score'] for f in families)
    return {'protocol_id': PROTOCOL, 'score': total, 'families': families,
            'status': 'EXPERIMENTAL', 'independent_sample_count': None}, scores


def provenance(paths, extra=None):
    def git(*args):
        try:
            return subprocess.check_output(['git'] + list(args), cwd=ROOT).decode('utf-8', 'replace').strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    result = {'protocol_id': PROTOCOL, 'python': sys.version, 'git_commit': git('rev-parse', 'HEAD'),
              'git_status': git('status', '--short'), 'dataset_sha256': DATA_SHA256,
              'input_hashes': {os.path.relpath(p, ROOT): sha256(p) for p in paths},
              'seeds': {'tail': POLICY['tail_seeds'], 'inner': POLICY['inner_seed']},
              'simulator_calls': 0}
    result.update(extra or {})
    return result


def freeze():
    targets = ['POLICY.json', 'split_manifest.json', 'CHECKSUMS.json', 'split_summary.csv', 'freeze_provenance.json']
    if any(os.path.exists(os.path.join(BASE, name)) for name in targets):
        raise ValueError('Release already exists; validate it, do not overwrite frozen files')
    data = load_data()
    manifest = create_manifest(data)
    validate_manifest(manifest, data)
    write_json(os.path.join(BASE, 'POLICY.json'), POLICY)
    write_json(os.path.join(BASE, 'split_manifest.json'), manifest)
    summaries = []
    for task in manifest['tasks']:
        row = {'split_id': task['split_id'], 'family': task['family'],
               'train_n': len(task['train_ids']), 'test_n': len(task['test_ids']),
               'train_best': task['train_best'], 'test_best': task['test_best'],
               'opportunity': task['train_best'] - task['test_best'],
               'breakthrough_test_n': sum(data[cid]['y'] < task['train_best'] for cid in task['test_ids'])}
        summaries.append(row)
    write_csv(os.path.join(BASE, 'split_summary.csv'), summaries)
    write_json(os.path.join(BASE, 'CHECKSUMS.json'),
               {name: sha256(os.path.join(BASE, name)) for name in targets[:2]})
    write_json(os.path.join(BASE, 'freeze_provenance.json'), provenance(
        [__file__, os.path.join(ROOT, 'dataset_same8/train.csv')],
        {'training_performed': False, 'dependencies': 'standard library only',
         'design_frozen_before_new_model_evidence': True}))
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    if sys.argv[1:] == ['freeze']:
        freeze()
    elif sys.argv[1:] == ['validate']:
        _, manifest = load_release()
        print('Validated {} experimental tasks'.format(len(manifest['tasks'])))
    else:
        raise SystemExit('Usage: python analysis/v33_protocol/protocol.py freeze|validate')
