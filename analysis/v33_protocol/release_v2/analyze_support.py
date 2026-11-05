"""Inspect support after each recipe's actual trimming; does not change scores or splits."""
import numpy as np
from scipy.optimize import linprog
from protocol import BASE, load_release, read_json, squared_distance, write_csv, write_json


def main():
    data, manifest, _ = load_release()
    tasks = {t['split_id']: t for t in manifest['tasks'] if t['family'] == 'supported_low'}
    rows, summary = [], {}
    for run in sorted((BASE / 'results').glob('*-evaluate-v332-runze_*')):
        recipe = read_json(run / 'provenance.json')['recipe']['name']
        membership = read_json(run / 'primary/fit_membership.json')
        for fit in membership:
            if fit['split_id'] not in tasks:
                continue
            task = tasks[fit['split_id']]
            retained = fit['retained_train_ids']
            X = np.asarray([data[cid]['x'] for cid in retained])
            constraints = np.vstack([X.T, np.ones(len(X))])
            for cid in task['mandatory_hidden_ids']:
                nearest_distance = min(squared_distance(data[cid]['x'], data[n]['x']) for n in retained)**.5
                result = linprog(np.zeros(len(X)), A_eq=constraints,
                    b_eq=np.r_[data[cid]['x'], 1.], bounds=(0., None), method='highs')
                residual = float(np.max(np.abs(constraints @ result.x-np.r_[data[cid]['x'], 1.]))) if result.success else None
                neighbor = task['design']['reserved_neighbors'][cid]
                rows.append({'model': recipe, 'split_id': task['split_id'], 'candidate_id': cid,
                    'reserved_neighbor_retained': neighbor in retained,
                    'actual_fit_nearest_distance': nearest_distance,
                    'inside_actual_fit_convex_hull': bool(result.success and residual <= 1e-7),
                    'solver_status': int(result.status), 'max_reconstruction_error': residual})
        part = [r for r in rows if r['model'] == recipe]
        summary[recipe] = {'points': len(part),
            'reserved_neighbors_retained': sum(r['reserved_neighbor_retained'] for r in part),
            'convex_hull_feasible': sum(r['inside_actual_fit_convex_hull'] for r in part)}
    write_csv(BASE / 'retained_geometry.csv', rows)
    write_json(BASE / 'retained_geometry_summary.json', summary)
    print(summary)


if __name__ == '__main__':
    main()
