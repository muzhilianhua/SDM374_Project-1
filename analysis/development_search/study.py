"""Bounded local development search using measured historical anchors, never a graded policy.

All weights/labels are read from external development logs. No anchor is compiled
into a policy or configuration. Outputs include pending attempts and returned
scores; uncertain or failed attempts are not automatically retried.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from sdm.online.evaluator import SolverEvaluator, pinned_sha256
from sdm.runs import git_state
from sdm.simplex import check_weights


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def key(w):
    w = np.asarray(w).reshape(3, 7)
    return tuple(np.round((w / w.sum(axis=1, keepdims=True)).ravel(), 12))


def load_history(path):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    unique = {}
    for row in records:
        raw = np.asarray(row['weights'], dtype=float).reshape(3, 7)
        # The solver rescales each building internally; older development logs
        # contain a few equivalent, non-unit representations.
        if not np.isfinite(raw).all() or (raw < 0).any() or (raw.sum(axis=1) <= 0).any():
            raise ValueError('invalid historical weights')
        w = check_weights((raw / raw.sum(axis=1, keepdims=True)).ravel())[0]
        score = float(row['R'])
        if not np.isfinite(score):
            raise ValueError('nonfinite historical score')
        unique.setdefault(key(w), (w, score))
    return np.array([r[0] for r in unique.values()]), np.array([r[1] for r in unique.values()])


def candidate_pool(anchor, rng, size=6000):
    """Zero-sum transfers preserve simplex faces; some proposals activate new coordinates."""
    anchor = np.asarray(anchor).reshape(3, 7)
    output, families = [], []
    scales = (0.0001, 0.0003, 0.0008, 0.002, 0.006, 0.015, 0.04)
    for index in range(size):
        w = anchor.copy()
        activate = index % 5 == 0
        scale = scales[index % len(scales)]
        edits = 1 if index % 4 == 0 else int(rng.integers(2, 5))
        for _ in range(edits):
            building = int(rng.integers(3))
            donors = np.flatnonzero(w[building] > 0)
            donor = int(rng.choice(donors))
            recipients = np.arange(7) if activate else donors
            recipients = recipients[recipients != donor]
            if not len(recipients):
                continue
            recipient = int(rng.choice(recipients))
            delta = min(w[building, donor], scale * rng.uniform(0.5, 1.5))
            w[building, donor] -= delta
            w[building, recipient] += delta
        output.append(w.ravel())
        families.append(('support' if activate else 'face') + f'-{scale:g}')
    return check_weights(np.array(output)), families


def select_batch(pool, families, X, y, rng, n=50):
    """Mix predicted low points, model disagreement and unscreened multi-scale probes."""
    seen = {key(w) for w in X}
    valid, labels = [], []
    for w, family in zip(pool, families):
        k = key(w)
        if k not in seen:
            valid.append(w); labels.append(family); seen.add(k)
    pool = np.array(valid)
    if len(pool) < n:
        raise ValueError('insufficient distinct candidates')
    anchor = X[np.argmin(y)]
    distances = np.linalg.norm(X - anchor, axis=1)
    training = np.argsort(distances, kind='stable')[:min(1800, len(X))]
    model = ExtraTreesRegressor(n_estimators=96, min_samples_leaf=2, max_features=1.0,
                                n_jobs=1, random_state=int(rng.integers(2**31)))
    model.fit(X[training], y[training])
    member_predictions = np.array([tree.predict(pool) for tree in model.estimators_])
    mean = member_predictions.mean(axis=0)
    # Tree spread is a search heuristic, not calibrated predictive uncertainty.
    spread = member_predictions.std(axis=0)
    indices = []
    for order, count in [(np.argsort(mean, kind='stable'), n//3),
                         (np.argsort(mean-spread, kind='stable'), n//3),
                         (rng.permutation(len(pool)), n)]:
        if count == 0:
            continue
        used = 0
        for index in order:
            if index not in indices:
                indices.append(int(index)); used += 1
            if used == count or len(indices) == n:
                break
        if len(indices) == n:
            break
    return pool[indices], [{'family': labels[i], 'predicted_R': float(mean[i]),
                            'tree_spread': float(spread[i])} for i in indices]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', type=Path, default=Path('analysis/replay/evidence/eval_log.jsonl'))
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--calls', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--seed', type=int, default=20261112)
    args = parser.parse_args()
    if not 1 <= args.calls <= 19 or not 1 <= args.batch_size <= 50:
        parser.error('development budget must be 1-19 calls of 1-50 points')
    if args.out.exists():
        parser.error('output must be new; uncertain attempts require manual reconciliation')
    args.out.mkdir(parents=True)
    X, y = load_history(args.history)
    expected = pinned_sha256(Path('solver/CHECKSUMS.sha256'), 'solver.exe')
    actual = hashlib.sha256(args.exe.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError('course executable hash mismatch')
    provenance = {'purpose': 'development-only search with historical anchors; not a graded policy',
                  'git': git_state(), 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'history_sha256': hashlib.sha256(args.history.read_bytes()).hexdigest(),
                  'dataset_sha256': hashlib.sha256(Path('dataset_same8/train.csv').read_bytes()).hexdigest(),
                  'solver_sha256': actual, 'seed': args.seed, 'calls': args.calls,
                  'batch_size': args.batch_size, 'initial_points': len(X), 'initial_best_R': float(y.min()),
                  'python': platform.python_version(),
                  'packages': {name: version(name) for name in ('numpy', 'scipy', 'scikit-learn')},
                  'timestamp': datetime.now(timezone.utc).isoformat()}
    dump(args.out/'provenance.json', provenance)
    # Uses an independent development copy and never resets its existing counter.
    evaluator = SolverEvaluator(exe=args.exe, checksums=Path('solver/CHECKSUMS.sha256'),
                                quota=args.calls, timeout=3600)
    rng = np.random.default_rng(args.seed)
    initial_best = float(y.min())
    for call in range(1, args.calls+1):
        anchor = X[np.argmin(y)]
        pool, families = candidate_pool(anchor, rng)
        weights, predictions = select_batch(pool, families, X, y, rng, args.batch_size)
        pending = {'call': call, 'weights': weights.tolist(), 'predictions': predictions,
                   'status': 'pending', 'timestamp': datetime.now(timezone.utc).isoformat()}
        path = args.out/f'call_{call:02d}.json'
        # Pending evidence is durable before evaluator launch; no automatic retries.
        with path.open('w', encoding='utf-8') as stream:
            json.dump(pending, stream); stream.flush(); os.fsync(stream.fileno())
        start = time.perf_counter()
        print(f'call {call}/{args.calls}: {len(weights)} new points, incumbent {y.min():.10f}', flush=True)
        try:
            scores = evaluator.evaluate(weights)
        except BaseException as error:
            pending.update(status='uncertain_or_failed', error=repr(error), evaluator=evaluator.describe())
            dump(path, pending)
            raise
        pending.update(status='completed', R=scores.tolist(), seconds=time.perf_counter()-start,
                       evaluator=evaluator.describe())
        dump(path, pending)
        X = np.concatenate([X, weights]); y = np.concatenate([y, scores])
        best = int(np.argmin(y))
        dump(args.out/'best.json', {'R': float(y[best]), 'weights': X[best].reshape(3, 7).tolist(),
                                   'initial_best_R': initial_best, 'improved': bool(y[best] < initial_best),
                                   'completed_calls': call, 'new_points': call*args.batch_size,
                                   'best_new_R': float(y[provenance['initial_points']:].min())})
        print(f'completed: batch min {scores.min():.10f}; best {y.min():.10f}', flush=True)


if __name__ == '__main__':
    main()
