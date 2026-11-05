"""Research entry point for v3.3; no change to the approved sdm evaluate protocol."""
import argparse
from datetime import datetime, timezone
import os
import sys

try:
    from importlib import metadata
except ImportError:
    metadata = None

from protocol import (BASE, ROOT, evaluate_rows, load_release, provenance, read_csv,
                      sha256, write_csv, write_json)


def fit_predictions(config_path, data, manifest):
    # Imports are lazy: the scorer itself has no ML dependency.
    from sdm.config import load_config
    from sdm.data import load_dataset
    from sdm.registry import build_model
    config = load_config(config_path)
    ds = load_dataset(config.dataset)
    if ds.sha256 != sha256(os.path.join(ROOT, 'dataset_same8/train.csv')):
        raise ValueError('Config must use the official dataset')
    positions = {cid: i for i, cid in enumerate(ds.ids)}
    predictions = []
    for task in manifest['tasks']:
        print('Fitting ' + task['split_id'], flush=True)
        train = [positions[cid] for cid in task['train_ids']]
        test = [positions[cid] for cid in task['test_ids']]
        # A new complete model for every task. No pretrained full-data model is accepted.
        model = build_model(config.model.name, config.model.params)
        model.fit(ds.X[train], ds.y[train])
        mean, _ = model.predict(ds.X[test])
        if len(mean) != len(test):
            raise ValueError('Wrong model output shape')
        for cid, value in zip(task['test_ids'], mean):
            predictions.append({'split_id': task['split_id'], 'candidate_id': cid,
                                'predicted_mean': float(value)})
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--config', help='Existing experiment TOML; use model settings only; refit all tasks')
    source.add_argument('--predictions', help='External exact-task predictions; fitting exclusion is provider-declared')
    parser.add_argument('--output', required=True, help='New output directory; never overwritten')
    args = parser.parse_args()
    if args.config and sys.version_info[:2] != (3, 11):
        parser.error('Fresh model fits require the project Python 3.11 environment')
    if os.path.exists(args.output):
        parser.error('Output already exists; choose a new directory')
    data, manifest = load_release()
    input_path = os.path.abspath(args.config or args.predictions)
    source_files = [os.path.join(BASE, name) for name in
                    ['protocol.py', 'evaluate.py', 'POLICY.json', 'split_manifest.json', 'CHECKSUMS.json']]
    if args.config:
        source_files.extend(os.path.join(parent, name)
                            for parent, _, names in os.walk(os.path.join(ROOT, 'src/sdm'))
                            for name in sorted(names) if name.endswith('.py'))
        source_files.extend(os.path.join(ROOT, name) for name in ['pyproject.toml', 'uv.lock'])
    packages = {}
    for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'xgboost', 'catboost']:
        if metadata is None:
            packages[name] = None
            continue
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    record = provenance(source_files + [input_path], {
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'mode': 'fresh_fixed_config_refits' if args.config else 'external_predictions',
        'packages': packages, 'argv': sys.argv, 'optimizer_used': False,
        'old_config_evaluate_section': 'ignored: tasks are exclusively v3.3 manifest',
        'hyperparameter_tuning': 'none; existing settings carry historical selection bias',
        'fitting_exclusion': 'enforced by runner' if args.config else 'provider-declared, not independently proven'})
    os.makedirs(args.output)
    write_json(os.path.join(args.output, 'provenance.json'), record)
    try:
        rows = fit_predictions(input_path, data, manifest) if args.config else read_csv(input_path)
        metrics, scores = evaluate_rows(rows, manifest, data)
        write_csv(os.path.join(args.output, 'predictions.csv'), rows)
        write_csv(os.path.join(args.output, 'task_scores.csv'), scores)
        write_json(os.path.join(args.output, 'metrics.json'), metrics)
        record.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
                      outputs={name: sha256(os.path.join(args.output, name)) for name in
                               ['predictions.csv', 'task_scores.csv', 'metrics.json']})
        print('Experimental v3.3 score: {:.6f}'.format(metrics['score']))
    except Exception as error:
        record.update(status='failed', error=repr(error))
        raise
    finally:
        write_json(os.path.join(args.output, 'provenance.json'), record)


if __name__ == '__main__':
    main()
