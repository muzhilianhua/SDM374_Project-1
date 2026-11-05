"""Fresh paired refits for frozen v3.3.1; no optimizer and no simulator."""
import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import time

import joblib
import numpy as np
from protocol import (BASE, ROOT, PROTOCOL, P1_ACTUAL, evaluate_rows, load_release,
                      read_json, sha256, source_provenance, write_csv, write_json)
import recipes  # Registers the isolated study plugin.
from sdm.registry import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    data, manifest, p1 = load_release()
    recipe = next(r for r in read_json(BASE / 'recipes.json')['recipes'] if r['name'] == args.recipe)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    version_names = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'xgboost', 'catboost', 'joblib']
    record = source_provenance({'run_id': output.name, 'started_utc': datetime.now(timezone.utc).isoformat(),
        'recipe': recipe, 'packages': {name: metadata.version(name) for name in version_names},
        'release_sha256': sha256(BASE / 'CHECKSUMS.json'), 'optimizer_used': False,
        'tuning_performed': False, 'source_state': 'uncommitted content hashes recorded; Git identity unavailable',
        'thread_policy': 'OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1',
        'phase_comparison': 'feedback uses new known best and max95; totals not ranked across stages'})
    write_json(output / 'provenance.json', record)
    start = time.monotonic()
    try:
        for feedback in (False, True):
            stage = 'feedback' if feedback else 'primary'
            stage_dir = output / stage
            stage_dir.mkdir()
            predictions, membership = [], []
            for task in manifest['tasks']:
                split_start = time.monotonic()
                ids = task['train_ids'] + (['P1'] if feedback else [])
                X = np.asarray([data[cid]['x'] if cid != 'P1' else np.ravel(p1['weights']) for cid in ids])
                y = np.asarray([data[cid]['y'] if cid != 'P1' else P1_ACTUAL for cid in ids])
                model = build_model(recipe['registered_model'], recipe['params']).fit(X, y)
                test_ids = task['test_ids'] + ['P1']
                test_X = np.asarray([data[cid]['x'] if cid != 'P1' else np.ravel(p1['weights']) for cid in test_ids])
                mean, _ = model.predict(test_X)
                for cid, value in zip(test_ids, mean):
                    predictions.append({'split_id': task['split_id'], 'candidate_id': cid,
                        'actual': data[cid]['y'] if cid != 'P1' else P1_ACTUAL, 'predicted_mean': float(value)})
                membership.append({'split_id': task['split_id'], 'stage': stage,
                    'allowed_train_ids': ids, 'retained_train_ids': [ids[i] for i in model.retained_indices_],
                    'fit_metrics': model.metrics})
                joblib.dump(model, stage_dir / (task['split_id'] + '.joblib'), compress=3)
                print('{} {} {} {:.1f}s'.format(args.recipe, stage, task['split_id'], time.monotonic()-split_start), flush=True)
            metrics, scores = evaluate_rows(predictions, manifest, data, feedback)
            write_csv(stage_dir / 'predictions.csv', predictions)
            write_csv(stage_dir / 'task_scores.csv', scores)
            write_json(stage_dir / 'metrics.json', metrics)
            write_json(stage_dir / 'fit_membership.json', membership)
            # Save a complete final fit separately; its in-sample errors never enter benchmark scores.
            ids = sorted(data) + (['P1'] if feedback else [])
            X = np.asarray([data[cid]['x'] if cid != 'P1' else np.ravel(p1['weights']) for cid in ids])
            y = np.asarray([data[cid]['y'] if cid != 'P1' else P1_ACTUAL for cid in ids])
            final = build_model(recipe['registered_model'], recipe['params']).fit(X, y)
            final_prediction = float(final.predict(np.asarray(p1['weights']))[0][0])
            joblib.dump(final, stage_dir / 'full_fit.joblib', compress=3)
            write_json(stage_dir / 'full_fit.json', {'allowed_train_n': len(ids),
                'retained_train_ids': [ids[i] for i in final.retained_indices_],
                'fit_metrics': final.metrics, 'p1_prediction': final_prediction,
                'p1_seen_in_fit': feedback, 'scored': False,
                'note': 'Preserves lineage trimming; full fit means all allowed rows provided before trimming'})
            print('{} {} score={:.6f}; full-fit P1={:.6f}'.format(args.recipe, stage, metrics['score'], final_prediction), flush=True)
        record.update(status='complete', elapsed_seconds=time.monotonic()-start,
            completed_utc=datetime.now(timezone.utc).isoformat(),
            outputs={str(p.relative_to(output)): sha256(p) for p in output.rglob('*') if p.is_file() and p.name != 'provenance.json'})
    except Exception as error:
        record.update(status='failed', error=repr(error), elapsed_seconds=time.monotonic()-start)
        raise
    finally:
        write_json(output / 'provenance.json', record)


if __name__ == '__main__':
    main()
