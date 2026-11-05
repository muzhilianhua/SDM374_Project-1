"""V3.3.2: combine breakthrough stress tests with rotating, neighbor-supported low-value tests."""
import importlib.util
import math
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
spec = importlib.util.spec_from_file_location('_v331_frozen', BASE.parent / 'release_v1/protocol.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
read_json, write_json, read_csv, write_csv, sha256 = old.read_json, old.write_json, old.read_csv, old.write_csv, old.sha256
P1_ACTUAL = old.P1_ACTUAL
PROTOCOL = 'sdm374.offline-breakthrough.v3.3.2'
POLICY = dict(old.POLICY, protocol_id=PROTOCOL,
    family_weights={'hidden_elite': .25, 'local_region': .25, 'supported_low': .5},
    points={'best_identification': 35, 'true_low_accuracy': 30, 'predicted_low_accuracy': 30, 'p1_accuracy': 5},
    supported_elite_n=20, supported_folds=5, supported_seed=3320,
    supported_partition='elite true-rank round robin; four test and sixteen forced train per task',
    supported_neighbors='reserve nearest non-held point for each mandatory hidden elite in training',
    supported_recognition='35 / (1 + test_top1_regret / 5); no train-best breakthrough requirement',
    stress_recognition='35 * max(0, (train_best - selected_actual) / (train_best - test_best))',
    score='35 best identification + 30 true-low MAE + 30 predicted-low MAE + 5 P1; two task regimes',
    task_overlap='14 dependent tasks; each top20 elite tests once and trains four times in supported tasks',
    geometry_claim='nearest-neighbor support is enforced; strict convex-hull interpolation is diagnostic only')


def squared_distance(a, b):
    return sum((x-y)**2 for x, y in zip(a, b))


def create_manifest(data):
    manifest = old.create_manifest(data)
    ids = sorted(data)
    elite = sorted(data, key=lambda cid: (data[cid]['y'], cid))[:POLICY['supported_elite_n']]
    for task in manifest['tasks']:
        task['evaluation_mode'] = 'breakthrough'
        task['mandatory_train_ids'] = []
    for fold in range(POLICY['supported_folds']):
        hidden = elite[fold::POLICY['supported_folds']]
        forced_train = set(elite) - set(hidden)
        neighbors = {}
        for cid in hidden:
            nearest = min((other for other in ids if other not in hidden),
                          key=lambda other: (squared_distance(data[cid]['x'], data[other]['x']), other))
            neighbors[cid] = nearest
            forced_train.add(nearest)
        split = 'supported-f{:02d}'.format(fold + 1)
        pool = old.legacy.hash_order([cid for cid in ids if cid not in hidden and cid not in forced_train],
                                     'supported:{}:{}'.format(POLICY['supported_seed'], fold + 1))
        test = sorted(hidden + pool[:POLICY['test_n'] - len(hidden)])
        train = sorted(set(ids) - set(test))
        order = old.legacy.hash_order(train, 'inner:{}:{}'.format(POLICY['inner_seed'], split))
        inner = []
        for k in range(POLICY['inner_folds']):
            validation = sorted(order[k::POLICY['inner_folds']])
            inner.append({'fold': k + 1, 'train_ids': sorted(set(train) - set(validation)),
                          'validation_ids': validation})
        manifest['tasks'].append({'split_id': split, 'family': 'supported_low',
            'evaluation_mode': 'within_test', 'train_ids': train, 'test_ids': test,
            'inner_splits': inner, 'mandatory_hidden_ids': hidden, 'mandatory_train_ids': sorted(forced_train),
            'train_best': min(data[cid]['y'] for cid in train),
            'test_best': min(data[cid]['y'] for cid in test),
            'design': {'elite_n': len(elite), 'elite_test_n': len(hidden), 'elite_train_n': len(elite)-len(hidden),
                       'fold': fold + 1, 'reserved_neighbors': neighbors,
                       'reserved_neighbor_distances': {cid: math.sqrt(squared_distance(data[cid]['x'], data[n]['x']))
                                                       for cid, n in neighbors.items()}}})
    manifest.update(protocol_id=PROTOCOL, policy=POLICY)
    return manifest


def source_provenance(extra=None):
    files = sorted(BASE.glob('*.py')) + [BASE.parent / 'release_v1/protocol.py', BASE.parent / 'protocol.py',
        ROOT / 'dataset_same8/train.csv', ROOT / 'SUBMISSION_READY/01_FIRST_REAL_TEST/weights.json',
        ROOT / 'SUBMISSION_READY/KNOWN_OBSERVATIONS.json']
    record = old.legacy.provenance([str(path) for path in files], extra)
    record.update(protocol_id=PROTOCOL, supported_seed=POLICY['supported_seed'])
    return record


LOCKED = ['POLICY.json', 'split_manifest.json', 'P1.json', 'recipes.json', 'protocol.py',
          'recipes.py', 'study.py', '../release_v1/protocol.py', '../protocol.py']


def validate_manifest(manifest, data):
    if manifest != create_manifest(data):
        raise ValueError('Frozen tasks differ from deterministic v3.3.2 design')
    elite = sorted(data, key=lambda cid: (data[cid]['y'], cid))[:POLICY['supported_elite_n']]
    for task in manifest['tasks']:
        train, test = set(task['train_ids']), set(task['test_ids'])
        if train & test or train | test != set(data) or len(test) != 410:
            raise ValueError('Outer partition corrupted')
        if not set(task['mandatory_train_ids']) <= train:
            raise ValueError('Mandatory neighbors or elite training points lost')
        if task['evaluation_mode'] == 'breakthrough' and task['train_best'] <= task['test_best']:
            raise ValueError('Stress task lacks breakthrough opportunity')
    supported = [task for task in manifest['tasks'] if task['family'] == 'supported_low']
    for cid in elite:
        if sum(cid in task['train_ids'] for task in supported) != 4 or sum(cid in task['test_ids'] for task in supported) != 1:
            raise ValueError('Elite rotation must be four training appearances and one test appearance')


def load_release():
    checksums = read_json(BASE / 'CHECKSUMS.json')
    if set(checksums) != set(LOCKED):
        raise ValueError('Incomplete release checksums')
    for name, digest in checksums.items():
        if sha256(BASE / name) != digest:
            raise ValueError('Frozen release changed: ' + name)
    if read_json(BASE / 'POLICY.json') != POLICY:
        raise ValueError('Frozen policy changed')
    data = old.legacy.load_data()
    manifest = read_json(BASE / 'split_manifest.json')
    validate_manifest(manifest, data)
    p1 = read_json(BASE / 'P1.json')
    if p1 != old.load_p1_source():
        raise ValueError('P1 changed')
    return data, manifest, p1


def score_task(train_values, actual, predictions, p1_prediction, feedback=False, mode='breakthrough'):
    train = list(train_values) + ([P1_ACTUAL] if feedback else [])
    if not train or not actual or set(actual) != set(predictions):
        raise ValueError('Nonempty exact train/test membership required')
    if not all(math.isfinite(v) for v in train + list(actual.values()) + list(predictions.values()) + [p1_prediction]):
        raise ValueError('Finite labels and predictions required')
    if mode not in ('breakthrough', 'within_test'):
        raise ValueError('Unknown task mode')
    baseline, oracle = min(train), min(actual.values())
    chosen = min(actual, key=lambda cid: (predictions[cid], cid))
    best = min(actual, key=lambda cid: (actual[cid], cid))
    regret = actual[chosen] - oracle
    low = [cid for cid in actual if actual[cid] <= POLICY['cutoff']]
    pred_low = [cid for cid in actual if predictions[cid] <= POLICY['cutoff']]
    if not low:
        raise ValueError('Empty true-low set')
    scale = POLICY['accuracy_scale']
    if mode == 'breakthrough':
        if baseline <= oracle:
            raise ValueError('Stress task lacks breakthrough opportunity')
        recognition = 35 * max(0., (baseline - actual[chosen]) / (baseline - oracle))
    else:
        recognition = 35 / (1 + regret / scale)
    mae = lambda ids: sum(abs(predictions[cid]-actual[cid]) for cid in ids) / len(ids)
    true_mae, pred_mae = mae(low), mae(pred_low) if pred_low else None
    parts = {'best_identification': recognition,
        'true_low_accuracy': 30 / (1 + true_mae / scale),
        'predicted_low_accuracy': 30 / (1 + pred_mae / scale) if pred_low else 0.,
        'p1_accuracy': 5 / (1 + abs(p1_prediction - P1_ACTUAL) / scale) if not feedback else 0.}
    return dict(score=sum(parts.values()), **{'points_' + k: v for k, v in parts.items()},
        evaluation_mode=mode, train_best=baseline, test_best=oracle, opportunity=baseline-oracle,
        selected_id=chosen, selected_actual=actual[chosen], selected_prediction=predictions[chosen], regret=regret,
        achieved_improvement=baseline-actual[chosen], actual_breakthrough=actual[chosen] < baseline,
        false_breakthrough=predictions[chosen] < baseline and actual[chosen] >= baseline,
        selected_bias=predictions[chosen] - actual[chosen],
        true_best_id=best, true_best_bias=predictions[best]-actual[best],
        true_best_predicted_rank=1 + sum((predictions[cid], cid) < (predictions[best], best) for cid in actual),
        true_low_mae=true_mae, low_n=len(low), low_bias=sum(predictions[cid]-actual[cid] for cid in low)/len(low),
        global_mae=mae(list(actual)), predicted_low_mae=pred_mae, predicted_low_n=len(pred_low),
        p1_prediction=p1_prediction, p1_bias=p1_prediction-P1_ACTUAL,
        p1_absolute_error=abs(p1_prediction-P1_ACTUAL), p1_seen_in_fit=feedback, maximum_score=95 if feedback else 100)


def evaluate_rows(rows, manifest, data, feedback=False):
    predictions = {}
    for row in rows:
        key = (row['split_id'], row['candidate_id'])
        value = float(row['predicted_mean'])
        if key in predictions or not math.isfinite(value):
            raise ValueError('Duplicate or nonfinite prediction')
        predictions[key] = value
    expected = {(task['split_id'], cid) for task in manifest['tasks'] for cid in task['test_ids'] + ['P1']}
    if set(predictions) != expected:
        raise ValueError('Exact task membership including P1 required')
    scores = []
    for task in manifest['tasks']:
        actual = {cid: data[cid]['y'] for cid in task['test_ids']}
        pred = {cid: predictions[(task['split_id'], cid)] for cid in actual}
        result = score_task([data[cid]['y'] for cid in task['train_ids']], actual, pred,
            predictions[(task['split_id'], 'P1')], feedback, task['evaluation_mode'])
        mandatory = task['mandatory_hidden_ids']
        result.update(split_id=task['split_id'], family=task['family'], mandatory_hidden_n=len(mandatory),
            mandatory_hidden_mae=sum(abs(pred[cid]-actual[cid]) for cid in mandatory)/len(mandatory),
            mandatory_hidden_bias=sum(pred[cid]-actual[cid] for cid in mandatory)/len(mandatory))
        scores.append(result)
    keys = ['score', 'regret', 'selected_actual', 'actual_breakthrough', 'false_breakthrough',
            'true_low_mae', 'predicted_low_n', 'p1_absolute_error', 'p1_prediction', 'mandatory_hidden_mae']
    keys += ['points_' + k for k in POLICY['points']]
    families = {}
    for family in POLICY['family_weights']:
        part = [r for r in scores if r['family'] == family]
        families[family] = {key: sum(float(r[key]) for r in part)/len(part) for key in keys}
        families[family].update(tasks=len(part), worst_score=min(r['score'] for r in part))
    metrics = {key: sum(POLICY['family_weights'][f]*families[f][key] for f in families) for key in keys}
    metrics.update(protocol_id=PROTOCOL, families=families, stage='feedback' if feedback else 'primary',
        maximum_score=95 if feedback else 100, status='COMPLETE',
        p1_prediction_min=min(r['p1_prediction'] for r in scores), p1_prediction_max=max(r['p1_prediction'] for r in scores),
        worst_score=min(r['score'] for r in scores),
        caveat='Neighbor-supported low tests plus elite/region stress; no continuous-optimization guarantee')
    return metrics, scores


def freeze():
    names = ['POLICY.json', 'split_manifest.json', 'P1.json', 'CHECKSUMS.json', 'split_summary.csv', 'freeze_provenance.json']
    if any((BASE / name).exists() for name in names):
        raise ValueError('Refuse to overwrite a frozen release')
    old.load_release()
    data = old.legacy.load_data()
    manifest = create_manifest(data)
    validate_manifest(manifest, data)
    p1 = old.load_p1_source()
    write_json(BASE / 'POLICY.json', POLICY)
    write_json(BASE / 'split_manifest.json', manifest)
    write_json(BASE / 'P1.json', p1)
    elite5 = set(sorted(data, key=lambda cid: (data[cid]['y'], cid))[:5])
    summaries = []
    for task in manifest['tasks']:
        summaries.append({'split_id': task['split_id'], 'family': task['family'],
            'evaluation_mode': task['evaluation_mode'], 'train_n': len(task['train_ids']), 'test_n': len(task['test_ids']),
            'train_best': task['train_best'], 'test_best': task['test_best'],
            'top5_in_train': len(elite5 & set(task['train_ids'])),
            'top5_in_test': len(elite5 & set(task['test_ids'])),
            'mandatory_hidden_n': len(task['mandatory_hidden_ids']),
            'true_low_n': sum(data[cid]['y'] <= POLICY['cutoff'] for cid in task['test_ids'])})
    write_csv(BASE / 'split_summary.csv', summaries)
    write_json(BASE / 'CHECKSUMS.json', {name: sha256(BASE / name) for name in LOCKED})
    write_json(BASE / 'freeze_provenance.json', source_provenance({
        'training_performed': False, 'frozen_before_new_model_evidence': True,
        'authorization': 'User requested rotating elite participation and interpolation-oriented tests',
        'new_design_choices': '5-fold top20 rotation, reserved neighbors, supported tasks 50%; old stress families 25% each'}))
    load_release()
    print('Frozen v3.3.2: 9 stress tasks + 5 rotating neighbor-supported tasks')


if __name__ == '__main__':
    if sys.argv[1:] == ['freeze']:
        freeze()
    elif sys.argv[1:] == ['validate']:
        load_release()
        print('Validated v3.3.2')
    else:
        raise SystemExit('Usage: protocol.py freeze|validate')
