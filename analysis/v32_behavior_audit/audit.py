"""Audit saved frozen predictions without fitting or changing the scoring protocol."""
import csv
import gzip
import hashlib
import json
import os
import platform
import subprocess
import zipfile
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
OUT = os.path.dirname(__file__)


def mean(values):
    return sum(values) / len(values)


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def read_csv(path):
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(name, rows):
    with open(os.path.join(OUT, name), 'w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    sources = [
        ('selection_11_groups/heldout_predictions.csv.gz', 'group', 'actual', 'predicted'),
        ('full_data_configs/outer_predictions.csv.gz', 'config', 'y_true', 'y_pred'),
    ]
    data_path = os.path.join(ROOT, 'dataset_same8/train.csv')
    truth = {r['candidate_id']: float(r['ramping']) for r in read_csv(data_path)}
    zip_path = os.path.join(ROOT, 'frozen_data/SDM374_data_splits_frozen_v1.zip')
    assert digest(zip_path) == '91fbd29f05d64ff2470e8b8a15ade78b7f234b234bf32f03a67fb4ca5d15a09d'
    with zipfile.ZipFile(zip_path) as archive:
        manifest = json.loads(archive.read('SDM374_data_splits_frozen_v1/split_manifest.json').decode('utf-8'))
    expected = {s['split_id']: set(s['outer_test_ids']) for s in manifest['outer_splits']}
    assert digest(data_path) == manifest['dataset']['sha256']
    folds = []
    inputs = [data_path, zip_path]
    counts = {}
    for relative, model_key, actual_key, prediction_key in sources:
        path = os.path.join(ROOT, 'analysis/surrogate_release/evidence', relative)
        inputs.append(path)
        rows = read_csv(path)
        counts[relative] = len(rows)
        grouped = defaultdict(list)
        for r in rows:
            actual = float(r[actual_key])
            assert abs(actual - truth[r['candidate_id']]) < 1e-9
            grouped[(r[model_key], r['split_id'])].append((r['candidate_id'], actual, float(r[prediction_key])))
        for model in set(key[0] for key in grouped):
            assert set(key[1] for key in grouped if key[0] == model) == set(expected)
        for (model, split), part in sorted(grouped.items()):
            assert len(part) == len(expected[split]) and set(r[0] for r in part) == expected[split]
            low = [r for r in part if r[1] <= 132.8]
            for k in [1, 5, 10]:
                selected = sorted(part, key=lambda r: (r[2], r[0]))[:k]
                oracle = sorted(part, key=lambda r: (r[1], r[0]))[:k]
                folds.append(dict(model=model, split=split, k=k,
                    low_bias=mean([r[2]-r[1] for r in low]),
                    low_overprediction_fraction=mean([float(r[2]>r[1]) for r in low]),
                    selected_actual=mean([r[1] for r in selected]),
                    selected_bias=mean([r[2]-r[1] for r in selected]),
                    selected_under_by_gt5_fraction=mean([float(r[1]-r[2]>5) for r in selected]),
                    selection_gap=mean([r[1] for r in selected])-mean([r[1] for r in oracle]),
                    selected_ids=';'.join(r[0] for r in selected)))
    write_csv('fold_diagnostics.csv', folds)
    summary = []
    for model, k in sorted(set((r['model'], r['k']) for r in folds)):
        part = [r for r in folds if r['model'] == model and r['k'] == k]
        row = dict(model=model, k=k)
        for key in ['low_bias', 'low_overprediction_fraction', 'selected_actual', 'selected_bias',
                    'selected_under_by_gt5_fraction', 'selection_gap']:
            row[key] = mean([r[key] for r in part])
        row['worst_fold_selection_gap'] = max(r['selection_gap'] for r in part)
        summary.append(row)
    write_csv('summary.csv', summary)
    provenance = dict(python=platform.python_version(), packages='Python standard library only',
        training_performed=False, seeds='none: deterministic saved-prediction audit',
        bias_definition='predicted minus actual; positive = overprediction',
        averaging='equal weight to each of the ten reused frozen folds',
        counts=counts, inputs={os.path.relpath(p, ROOT): digest(p) for p in inputs},
        source_sha256=digest(__file__),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
        git_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT).decode('utf-8', 'replace'),
        limitations='Historical provider-declared fitting; reused folds; no continuous-candidate truth or new independent validation.')
    with open(os.path.join(OUT, 'provenance.json'), 'w', encoding='utf-8') as stream:
        json.dump(provenance, stream, indent=2)
    for row in summary:
        if row['k'] == 1:
            print('{model}: low bias {low_bias:.2f}, selected bias {selected_bias:.2f}, regret {selection_gap:.2f}, worst {worst_fold_selection_gap:.2f}'.format(**row))


if __name__ == '__main__':
    main()
