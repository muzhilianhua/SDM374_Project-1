"""Diagnose nearest-neighbor support and convex-hull membership; no model labels are fitted."""
import numpy as np
from scipy.optimize import linprog
from protocol import BASE, load_release, squared_distance, write_csv


def main():
    data, manifest, _ = load_release()
    rows = []
    for task in manifest['tasks']:
        if task['family'] != 'supported_low':
            continue
        train = task['train_ids']
        X = np.asarray([data[cid]['x'] for cid in train])
        constraints = np.vstack([X.T, np.ones(len(X))])
        for cid in task['mandatory_hidden_ids']:
            x = np.asarray(data[cid]['x'])
            neighbor = task['design']['reserved_neighbors'][cid]
            global_distance = min(squared_distance(data[cid]['x'], data[other]['x']) for other in data if other != cid)**.5
            training_distance = min(squared_distance(data[cid]['x'], data[other]['x']) for other in train)**.5
            result = linprog(np.zeros(len(X)), A_eq=constraints, b_eq=np.r_[x, 1.],
                             bounds=(0., None), method='highs')
            residual = float(np.max(np.abs(constraints @ result.x - np.r_[x, 1.]))) if result.success else None
            rows.append({'split_id': task['split_id'], 'candidate_id': cid,
                'reserved_neighbor_id': neighbor, 'reserved_neighbor_is_train': neighbor in train,
                'nearest_train_distance': training_distance, 'global_nearest_distance': global_distance,
                'distance_ratio_to_global_nearest': training_distance/global_distance if global_distance else None,
                'inside_allowed_train_convex_hull': bool(result.success and residual <= 1e-7),
                'solver_status': int(result.status), 'max_reconstruction_error': residual,
                'scope': 'allowed outer training rows before model-specific trimming; not a scoring criterion'})
    write_csv(BASE / 'geometry_diagnostics.csv', rows)
    print('Convex-hull feasible: {}/{}; all reserved neighbors in allowed train: {}'.format(
        sum(r['inside_allowed_train_convex_hull'] for r in rows), len(rows),
        all(r['reserved_neighbor_is_train'] for r in rows)))


if __name__ == '__main__':
    main()
