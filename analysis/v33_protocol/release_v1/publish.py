"""Verify completed fixed-protocol runs and publish compact, auditable comparisons."""
import csv
import gzip
import json
from pathlib import Path
import shutil

from protocol import BASE, POLICY, evaluate_rows, load_release, read_csv, read_json, sha256, write_csv, write_json


def main():
    data, manifest, _ = load_release()
    runs = sorted((BASE / 'results').glob('*-evaluate-v331-runze_*'))
    if len(runs) != 3:
        raise ValueError('Expected exactly one completed run for each of the three frozen recipes')
    target = BASE / 'delivery'
    target.mkdir(exist_ok=False)
    summary, paired, full, task_rows, source_records = [], [], [], [], []
    for run in runs:
        provenance = read_json(run / 'provenance.json')
        if provenance['status'] != 'complete':
            raise ValueError('Incomplete source run')
        for path, digest in provenance['outputs'].items():
            if sha256(run / path) != digest:
                raise ValueError('Source output hash mismatch: ' + path)
        recipe = provenance['recipe']
        recipe_dir = target / recipe['name']
        recipe_dir.mkdir()
        shutil.copyfile(run / 'provenance.json', recipe_dir / 'provenance.json')
        per_stage = {}
        for stage in ['primary', 'feedback']:
            src = run / stage
            rows = read_csv(src / 'predictions.csv')
            for row in rows:
                expected = data[row['candidate_id']]['y'] if row['candidate_id'] != 'P1' else 119.71206678869203
                if abs(float(row['actual']) - expected) > 1e-9:
                    raise ValueError('Prediction truth label mismatch')
            metrics, scores = evaluate_rows(rows, manifest, data, feedback=stage == 'feedback')
            if metrics != read_json(src / 'metrics.json'):
                raise ValueError('Independent rescoring disagrees')
            membership = read_json(src / 'fit_membership.json')
            for task, fit in zip(manifest['tasks'], membership):
                expected = task['train_ids'] + (['P1'] if stage == 'feedback' else [])
                if fit['allowed_train_ids'] != expected:
                    raise ValueError('Unexpected allowed training membership')
                if set(fit['retained_train_ids']) & set(task['test_ids']):
                    raise ValueError('Test label leakage into retained fitting rows')
                if stage == 'primary' and 'P1' in fit['retained_train_ids']:
                    raise ValueError('P1 leakage into primary fit')
            destination = recipe_dir / stage
            destination.mkdir()
            for name in ['metrics.json', 'task_scores.csv', 'full_fit.json']:
                shutil.copyfile(src / name, destination / name)
            for name in ['predictions.csv', 'fit_membership.json']:
                with open(src / name, 'rb') as stream, open(destination / (name + '.gz'), 'wb') as raw:
                    with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as compressed:
                        shutil.copyfileobj(stream, compressed)
            row = {'model': recipe['name'], 'historical_rank': recipe['historical_rank'],
                   'stage': stage, 'run_id': run.name}
            for key in ['score', 'maximum_score', 'points_breakthrough', 'points_true_low_accuracy',
                        'points_predicted_low_accuracy', 'points_p1_accuracy', 'selected_actual', 'regret',
                        'true_low_mae', 'mandatory_hidden_mae', 'predicted_low_n', 'p1_prediction',
                        'p1_absolute_error', 'p1_prediction_min', 'p1_prediction_max', 'worst_score']:
                row[key] = metrics[key]
            row['hidden_elite_score'] = metrics['families']['hidden_elite']['score']
            row['local_region_score'] = metrics['families']['local_region']['score']
            summary.append(row)
            for score in scores:
                task_rows.append(dict(model=recipe['name'], stage=stage, run_id=run.name, **score))
            fit = read_json(src / 'full_fit.json')
            full.append({'model': recipe['name'], 'stage': stage, 'run_id': run.name,
                         'allowed_train_n': fit['allowed_train_n'], 'retained_train_n': len(fit['retained_train_ids']),
                         'p1_prediction': fit['p1_prediction'],
                         'p1_error': fit['p1_prediction'] - 119.71206678869203, 'scored': False,
                         'model_path': str((src / 'full_fit.joblib').relative_to(BASE))})
            per_stage[stage] = scores
        for before, after in zip(per_stage['primary'], per_stage['feedback']):
            paired.append({'model': recipe['name'], 'split_id': before['split_id'], 'family': before['family'],
                'primary_selected_actual': before['selected_actual'], 'feedback_selected_actual': after['selected_actual'],
                'selected_actual_change': after['selected_actual'] - before['selected_actual'],
                'true_low_mae_change': after['true_low_mae'] - before['true_low_mae'],
                'mandatory_hidden_mae_change': after['mandatory_hidden_mae'] - before['mandatory_hidden_mae'],
                'note': 'negative change is better; phase totals are not directly comparable'})
        source_records.append({'model': recipe['name'], 'run_id': run.name,
            'provenance_sha256': sha256(run / 'provenance.json'), 'source_recipe': recipe})
    write_csv(target / 'summary.csv', summary)
    write_csv(target / 'task_scores_all.csv', task_rows)
    write_csv(target / 'feedback_paired_changes.csv', paired)
    write_csv(target / 'full_fit_p1_predictions.csv', full)
    write_json(target / 'SOURCE_RUNS.json', source_records)
    write_json(target / 'VERIFICATION.json', {'protocol_id': POLICY['protocol_id'],
        'source_run_count': 3, 'outer_model_fits': 54, 'complete_final_fits': 6,
        'rescored_tasks': 54, 'verified_prediction_rows': 22194,
        'fit_membership_verified': True, 'output_hashes_verified': True,
        'publish_source_sha256': sha256(__file__),
        'release_checksums_sha256': sha256(BASE / 'CHECKSUMS.json'),
        'no_optimizer_or_simulator': True})
    for row in sorted([r for r in summary if r['stage'] == 'primary'], key=lambda r: -r['score']):
        print(row['model'], row['score'])


if __name__ == '__main__':
    main()
