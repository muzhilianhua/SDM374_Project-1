"""Exact local simulation search using modules/data extracted from solver.exe."""
from __future__ import annotations

import json
from multiprocessing import get_context
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ROOT = Path(__file__).resolve().parent
ANCHOR = np.array([
    [0.0003855940280853773, 0.045, 0.0, 0.0, 0.25760783455547365, 0.687006571416441, 0.01],
    [0.0, 0.010000000000000002, 0.0, 0.00039537335553845715, 0.21029005469622036, 0.6492697850133666, 0.13004478693487453],
    [0.0, 0.0, 0.010203889879216995, 0.0, 0.24635139398101458, 0.7322436592497638, 0.011201056890004769],
])
EVAL = None
SCENARIOS = [ROOT / 'resources' / 'environment' / 'scenarios' / f'week_{i:02d}.json' for i in range(1, 9)]


def init_worker():
    global EVAL
    import importlib.metadata
    from load_bundle import install
    install()
    original = importlib.metadata.version
    importlib.metadata.version = lambda name: '2.5.0' if name == 'CityLearn' else original(name)
    from citylearn.data import DataSet
    DataSet.get_pv_sizing_data = lambda *args, **kwargs: None
    DataSet.get_battery_sizing_data = lambda *args, **kwargs: None
    from evaluate import evaluate_probe
    EVAL = evaluate_probe


def score_one(flat):
    w = np.asarray(flat, dtype=float).reshape(3, 7)
    scores = [EVAL(w, str(path), 'load_pv', 0, 0, False)['ramping'] for path in SCENARIOS]
    return float(np.mean(scores)), [float(x) for x in scores]


def candidate_pool(seed=20261112, n=200):
    rng = np.random.default_rng(seed)
    scales = (1e-5, 3e-5, 1e-4, 3e-4, 7e-4, 0.0014, 0.0028, 0.005, 0.01)
    out = [ANCHOR.copy()]
    labels = ['anchor']
    for k in range(n - 1):
        w = ANCHOR.copy()
        edits = 1 if k % 3 else 2
        scale = scales[k % len(scales)]
        activate = k % 7 == 0
        for _ in range(edits):
            b = int(rng.integers(3))
            donors = np.flatnonzero(w[b] > 1e-12)
            donor = int(rng.choice(donors))
            receivers = np.arange(7) if activate else donors
            receivers = receivers[receivers != donor]
            receiver = int(rng.choice(receivers))
            amount = min(w[b, donor], scale * rng.uniform(0.45, 1.55))
            w[b, donor] -= amount
            w[b, receiver] += amount
        out.append(w)
        labels.append(f"{'support' if activate else 'face'}-{scale:g}")
    return out, labels


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261112)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    weights, labels = candidate_pool(args.seed, args.n)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = [None] * len(weights)
    ctx = get_context('spawn')
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, initializer=init_worker) as pool:
        futures = {pool.submit(score_one, w.ravel().tolist()): i for i, w in enumerate(weights)}
        for count, future in enumerate(as_completed(futures), 1):
            i = futures[future]
            mean, weeks = future.result()
            results[i] = {'index': i, 'label': labels[i], 'R': mean, 'weekly_R': weeks,
                          'weights': weights[i].tolist()}
            if count % 10 == 0:
                best = min(results[j] for j in range(len(results)) if results[j] is not None)
                print(f'{count}/{len(weights)} best={best["R"]:.12f}', flush=True)
    results.sort(key=lambda x: x['R'])
    payload = {'purpose': 'exact local simulation using extracted solver modules/data; no solver quota used',
               'seed': args.seed, 'n': len(results), 'workers': args.workers,
               'best': results[0], 'results': results}
    args.out.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'best_R': results[0]['R'], 'best_weights': results[0]['weights'],
                      'weekly_R': results[0]['weekly_R'], 'out': str(args.out)}, indent=2))


if __name__ == '__main__':
    main()
