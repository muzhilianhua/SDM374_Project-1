"""Frozen v3.3.1: breakthrough recognition, numeric accuracy and a small P1 component."""
import importlib.util
import hashlib
import math
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
spec = importlib.util.spec_from_file_location('_v33_initial_design', BASE.parent / 'protocol.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
read_json, write_json = legacy.read_json, legacy.write_json
read_csv, write_csv, sha256 = legacy.read_csv, legacy.write_csv, legacy.sha256
PROTOCOL = 'sdm374.offline-breakthrough.v3.3.1'
P1_ACTUAL = 119.71206678869203
P1_SHA256 = '98ce57bb5224551712a56c6269a86e4aa801d5d784dcdbb8758688817591b1f2'
POLICY = dict(legacy.POLICY, protocol_id=PROTOCOL, status='FROZEN',
    points={'breakthrough': 35, 'true_low_accuracy': 30, 'predicted_low_accuracy': 30, 'p1_accuracy': 5},
    accuracy_scale=5.0, cutoff=132.8,
    score='sum of 35-point clipped breakthrough utility, two 30-point MAE skills and 5-point P1 skill',
    score_clipping='breakthrough utility floored at zero; each accuracy component positive and bounded',
    empty_predicted_low='zero points; always report count',
    empty_true_low='fail the task; do not skip',
    p1_usage='excluded from primary fits; 5 points; feedback refits separately reported out of 95',
    feedback_baseline='minimum of original outer-train labels and P1; never rank against primary total',
    inner_tuning='not performed; published modal parameters fixed before evidence',
    task_overlap='all nine tasks contain p01969; not independent samples')


def create_manifest(data):
    manifest = legacy.create_manifest(data)
    manifest.update(protocol_id=PROTOCOL, policy=POLICY)
    elite = sorted(data, key=lambda cid: (data[cid]['y'], cid))
    for task in manifest['tasks']:
        n = task['design'].get('hidden_best_n', task['design'].get('mandatory_best_n'))
        task['mandatory_hidden_ids'] = elite[:n]
    return manifest


def load_p1_source():
    path = ROOT / 'SUBMISSION_READY/01_FIRST_REAL_TEST/weights.json'
    # Git's Windows checkout may convert LF to CRLF; verify the original LF payload bytes.
    if hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() != P1_SHA256:
        raise ValueError('P1 source payload checksum mismatch')
    weights = read_json(path)
    if len(weights) != 3 or any(len(row) != 7 for row in weights):
        raise ValueError('P1 must have three rows of seven weights')
    if any(not math.isfinite(v) or v < 0 for row in weights for v in row):
        raise ValueError('Invalid P1 weights')
    if any(abs(sum(row) - 1) > 1e-8 for row in weights):
        raise ValueError('P1 rows must sum to one')
    observations = read_json(ROOT / 'SUBMISSION_READY/KNOWN_OBSERVATIONS.json')['observations']
    observation = next(row for row in observations if row['id'] == 'P1')
    if observation['measured_R'] != P1_ACTUAL or observation['weights_sha256'] != P1_SHA256:
        raise ValueError('P1 observation contract mismatch')
    return {'candidate_id': 'P1', 'actual': P1_ACTUAL, 'weights': weights,
            'source_weights_sha256': P1_SHA256,
            'role': 'known inspected failure case, not independent validation'}


def source_provenance(extra=None):
    files = sorted(BASE.glob('*.py')) + [BASE.parent / 'protocol.py', ROOT / 'dataset_same8/train.csv',
        ROOT / 'SUBMISSION_READY/01_FIRST_REAL_TEST/weights.json',
        ROOT / 'SUBMISSION_READY/KNOWN_OBSERVATIONS.json']
    record = legacy.provenance([str(path) for path in files], extra)
    record['protocol_id'] = PROTOCOL
    return record


def load_release():
    checksums = read_json(BASE / 'CHECKSUMS.json')
    expected_names = {'POLICY.json', 'split_manifest.json', 'P1.json', 'recipes.json',
                      'protocol.py', 'recipes.py', 'study.py', '../protocol.py'}
    if set(checksums) != expected_names:
        raise ValueError('Incomplete release checksums')
    for name, digest in checksums.items():
        if sha256(BASE / name) != digest:
            raise ValueError('Frozen release changed: ' + name)
    if read_json(BASE / 'POLICY.json') != POLICY:
        raise ValueError('Frozen policy mismatch')
    data = legacy.load_data()
    manifest = read_json(BASE / 'split_manifest.json')
    if manifest != create_manifest(data):
        raise ValueError('Frozen tasks differ from deterministic design')
    p1 = read_json(BASE / 'P1.json')
    if p1 != load_p1_source():
        raise ValueError('Frozen P1 mismatch')
    for task in manifest['tasks']:
        if task['train_best'] <= task['test_best']:
            raise ValueError('Task lacks breakthrough opportunity')
    return data, manifest, p1


def score_task(train_values, actual, predictions, p1_prediction, feedback=False):
    if not math.isfinite(p1_prediction):
        raise ValueError('Finite P1 prediction required')
    train_values = list(train_values) + ([P1_ACTUAL] if feedback else [])
    result = legacy.score_task(train_values, actual, predictions)
    true_low = [cid for cid in actual if actual[cid] <= POLICY['cutoff']]
    pred_low = [cid for cid in actual if predictions[cid] <= POLICY['cutoff']]
    if not true_low:
        raise ValueError('No true low samples: invalid task')
    mae = lambda ids: sum(abs(predictions[cid] - actual[cid]) for cid in ids) / len(ids)
    true_mae, pred_mae = mae(true_low), mae(pred_low) if pred_low else None
    scale = POLICY['accuracy_scale']
    p1_error = p1_prediction - P1_ACTUAL
    parts = {
        'breakthrough': POLICY['points']['breakthrough'] * max(0., result['score'] / 100.),
        'true_low_accuracy': POLICY['points']['true_low_accuracy'] / (1 + true_mae / scale),
        'predicted_low_accuracy': POLICY['points']['predicted_low_accuracy'] / (1 + pred_mae / scale)
                                  if pred_low else 0.,
        'p1_accuracy': POLICY['points']['p1_accuracy'] / (1 + abs(p1_error) / scale)
                       if not feedback else 0.}
    result.update(raw_breakthrough_score=result['score'], score=sum(parts.values()),
        **{'points_' + key: value for key, value in parts.items()},
        true_low_mae=true_mae, predicted_low_mae=pred_mae, predicted_low_n=len(pred_low),
        predicted_low_bias=sum(predictions[cid] - actual[cid] for cid in pred_low) / len(pred_low)
                           if pred_low else None,
        predicted_low_underestimate_gt5_fraction=sum(actual[cid] - predictions[cid] > 5 for cid in pred_low) / len(pred_low)
                                                if pred_low else None,
        p1_prediction=p1_prediction, p1_bias=p1_error, p1_absolute_error=abs(p1_error),
        p1_seen_in_fit=feedback, maximum_score=95 if feedback else 100)
    return result


def evaluate_rows(rows, manifest, data, feedback=False):
    predictions = {}
    for row in rows:
        key = (row['split_id'], row['candidate_id'])
        if key in predictions:
            raise ValueError('Duplicate task/sample prediction')
        value = float(row['predicted_mean'])
        if not math.isfinite(value):
            raise ValueError('Nonfinite prediction')
        predictions[key] = value
    expected = {(task['split_id'], cid) for task in manifest['tasks'] for cid in task['test_ids'] + ['P1']}
    if set(predictions) != expected:
        raise ValueError('Require complete exact test membership plus one P1 prediction per task')
    scores = []
    for task in manifest['tasks']:
        actual = {cid: data[cid]['y'] for cid in task['test_ids']}
        pred = {cid: predictions[(task['split_id'], cid)] for cid in task['test_ids']}
        result = score_task([data[cid]['y'] for cid in task['train_ids']], actual, pred,
                            predictions[(task['split_id'], 'P1')], feedback)
        mandatory = task['mandatory_hidden_ids']
        result.update(split_id=task['split_id'], family=task['family'],
            mandatory_hidden_n=len(mandatory),
            mandatory_hidden_mae=sum(abs(pred[cid] - actual[cid]) for cid in mandatory) / len(mandatory),
            mandatory_hidden_bias=sum(pred[cid] - actual[cid] for cid in mandatory) / len(mandatory))
        scores.append(result)
    aggregate_keys = ['score', 'regret', 'selected_actual', 'actual_breakthrough', 'false_breakthrough',
        'true_low_mae', 'predicted_low_n', 'p1_absolute_error', 'p1_prediction', 'mandatory_hidden_mae']
    aggregate_keys += ['points_' + key for key in POLICY['points']]
    families = {}
    for family in POLICY['family_weights']:
        part = [row for row in scores if row['family'] == family]
        families[family] = {key: sum(float(row[key]) for row in part) / len(part) for key in aggregate_keys}
        families[family].update(tasks=len(part), worst_score=min(row['score'] for row in part))
    metrics = {key: sum(POLICY['family_weights'][family] * families[family][key] for family in families)
               for key in aggregate_keys}
    metrics.update(protocol_id=PROTOCOL, families=families, status='COMPLETE',
        stage='feedback' if feedback else 'primary', maximum_score=95 if feedback else 100,
        p1_prediction_min=min(row['p1_prediction'] for row in scores),
        p1_prediction_max=max(row['p1_prediction'] for row in scores),
        worst_score=min(row['score'] for row in scores),
        caveat='Known-label designed tasks reuse p01969 and P1; no independent continuous-optimization guarantee')
    return metrics, scores


def freeze():
    names = ['POLICY.json', 'split_manifest.json', 'P1.json', 'split_summary.csv', 'CHECKSUMS.json', 'freeze_provenance.json']
    if any((BASE / name).exists() for name in names):
        raise ValueError('Frozen release exists; do not overwrite')
    data = legacy.load_data()
    p1 = load_p1_source()
    manifest = create_manifest(data)
    write_json(BASE / 'POLICY.json', POLICY)
    write_json(BASE / 'split_manifest.json', manifest)
    write_json(BASE / 'P1.json', p1)
    summaries = []
    for task in manifest['tasks']:
        summaries.append({'split_id': task['split_id'], 'family': task['family'],
            'train_n': len(task['train_ids']), 'test_n': len(task['test_ids']),
            'train_best': task['train_best'], 'test_best': task['test_best'],
            'opportunity': task['train_best'] - task['test_best'],
            'mandatory_hidden_n': len(task['mandatory_hidden_ids']),
            'true_low_n': sum(data[cid]['y'] <= POLICY['cutoff'] for cid in task['test_ids']),
            'breakthrough_test_n': sum(data[cid]['y'] < task['train_best'] for cid in task['test_ids'])})
    write_csv(BASE / 'split_summary.csv', summaries)
    locked = ['POLICY.json', 'split_manifest.json', 'P1.json', 'recipes.json',
              'protocol.py', 'recipes.py', 'study.py', '../protocol.py']
    write_json(BASE / 'CHECKSUMS.json', {name: sha256(BASE / name) for name in locked})
    write_json(BASE / 'freeze_provenance.json', source_provenance({
        'training_performed': False, 'frozen_before_new_model_evidence': True,
        'recipe_sha256': sha256(BASE / 'recipes.json'),
        'score_and_split_authorization': 'User requested fixed 35/30/30/5 protocol and top-three retraining'}))
    load_release()
    print('Frozen v3.3.1: nine tasks, 35/30/30/5 points, P1 excluded from primary fits')


if __name__ == '__main__':
    if sys.argv[1:] == ['freeze']:
        freeze()
    elif sys.argv[1:] == ['validate']:
        load_release()
        print('Validated frozen v3.3.1')
    else:
        raise SystemExit('Usage: protocol.py freeze|validate')
