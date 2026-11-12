"""Private, extracted-runtime development experiment; never a submission policy."""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path
import numpy as np
from local_exact_search import ANCHOR, ROOT, SCENARIOS, init_worker

warnings.filterwarnings('ignore', category=FutureWarning)
ENVS = []


def setup():
    init_worker()
    import torch
    torch.set_num_threads(1)
    from citylearn.citylearn import CityLearnEnv
    from citylearn_task.evaluate import load_scenario, _scenario_schema
    from citylearn_task.temporal import padded_schema
    from disturbances import perturb_schema
    for path in SCENARIOS:
        spec = load_scenario(str(path))
        with padded_schema(_scenario_schema(spec), spec) as schema:
            perturb_schema(schema, 'load_pv', 0, 0)
            env = CityLearnEnv(schema, buildings=list(range(3)),
                simulation_start_time_step=0,
                simulation_end_time_step=schema['simulation_end_time_step'],
                random_seed=spec.seed, central_agent=False,
                active_actions=['electrical_storage', 'cooling_device'])
        for building in env.buildings:
            model = getattr(building, 'stochastic_power_outage_model', None)
            if model is not None:
                model.random_seed = spec.seed
        ENVS.append(env)


def simulate(weights, capture=False):
    from controller import ProbeAgent
    import controller
    from initialization import single_accounting
    from citylearn_task.temporal import decision_observations
    from citylearn_task.evaluate import normalize_weights
    records = []
    original_choose = controller.choose_candidate
    context = {'week': 0, 'step': 0, 'building': 0}
    def traced(values, w, ties, normalize):
        chosen = original_choose(values, w, ties, normalize)
        features = np.asarray(values, dtype=float)
        if normalize and len(features) > 1:
            span = np.ptp(features, axis=0)
            active = span > 1e-10 * np.maximum(1.0, np.max(np.abs(features), axis=0))
            f = np.zeros_like(features)
            f[:, active] = (features[:, active] - features[:, active].min(axis=0))/span[active]
            features = f
        records.append({**context, 'features': features.copy(), 'chosen': int(chosen)})
        context['building'] += 1
        return chosen
    if capture:
        controller.choose_candidate = traced
    nets = []
    try:
        for week, env in enumerate(ENVS):
            agent = ProbeAgent(env, normalize_weights(weights), mode='bounded')
            obs, _ = env.reset()
            # The extracted evaluator installs this accounting shim once per
            # environment.  Re-installing it after reset trips its guard.
            if not getattr(env.buildings[0], '_teacher_single_accounting', False):
                single_accounting(env)
            for step in range(168):
                context.update(week=week, step=step, building=0)
                actions = agent.predict(decision_observations(env, obs), deterministic=True)
                obs, _, terminated, truncated, _ = env.step(actions)
                if (terminated or truncated) and step < 167:
                    raise RuntimeError('early termination')
            net = np.asarray([np.asarray(b.net_electricity_consumption, dtype=float)[:168]
                              for b in env.buildings])
            nets.append(net)
    finally:
        controller.choose_candidate = original_choose
    nets = np.asarray(nets)
    weekly = np.abs(np.diff(nets.sum(axis=1), axis=1)).sum(axis=1)
    return {'R': float(weekly.mean()), 'weekly_R': weekly.tolist(), 'nets': nets,
            'records': records}


def generate(weights, records, limit):
    candidates = []
    # Each candidate crosses an actual local action indifference hyperplane.
    for record in records:
        b = record['building']
        f = record['features']
        chosen = record['chosen']
        differences = f - f[chosen]
        gaps = differences @ weights[b]
        for donor in np.flatnonzero(weights[b] > 1e-10):
            for receiver in range(7):
                if receiver == donor:
                    continue
                slopes = differences[:, receiver] - differences[:, donor]
                valid = (slopes < -1e-12) & (gaps >= -1e-10)
                indexes = np.flatnonzero(valid)
                for alternative in indexes:
                    delta = max(0., gaps[alternative]) / (-slopes[alternative])
                    # Cross enough to clear the controller's 1e-12 tie tolerance.
                    amount = delta + max(1e-8, delta * .002)
                    if amount > min(weights[b, donor], .02):
                        continue
                    candidate = weights.copy()
                    candidate[b, donor] -= amount
                    candidate[b, receiver] += amount
                    candidates.append({'weights': candidate.tolist(), 'amount': float(amount),
                        'building': int(b), 'donor': int(donor), 'receiver': receiver,
                        'week': record['week'], 'step': record['step'],
                        'chosen': chosen, 'alternative': int(alternative)})
    # Select across different coordinates and across logarithmic distance scales.
    candidates.sort(key=lambda c: c['amount'])
    selected, keys, seen = [], set(), set()
    for item in candidates:
        bucket = (item['building'], item['donor'], item['receiver'],
                  int(np.floor(np.log10(max(item['amount'], 1e-12)))))
        vector_key = np.asarray(item['weights']).round(12).tobytes()
        if bucket in keys or vector_key in seen:
            continue
        keys.add(bucket)
        seen.add(vector_key)
        selected.append(item)
        if len(selected) == limit:
            break
    return selected, len(candidates)


def worker_init():
    setup()


def score_candidate(item):
    start = time.perf_counter()
    result = simulate(np.asarray(item['weights']))
    return {**item, 'R': result['R'], 'weekly_R': result['weekly_R'],
            'seconds': time.perf_counter() - start}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=36)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--out', type=Path, default=ROOT/'boundary_results.json')
    args = parser.parse_args()
    started = time.perf_counter()
    setup()
    w = ANCHOR.copy()
    anchor = simulate(w, capture=True)
    print(f'anchor R={anchor["R"]:.12f}; decisions={len(anchor["records"])}; elapsed={time.perf_counter()-started:.1f}s', flush=True)
    # A reset must reproduce an identical net-load trajectory, not just mean R.
    repeat = simulate(w)
    assert np.array_equal(anchor['nets'], repeat['nets']), 'reset changed trajectory'
    assert abs(anchor['R'] - 104.82363781426102) < 1e-7, 'anchor mismatch'
    payload = {'kind': 'extracted-runtime boundary development search', 'formal_solver_calls': 0,
        'anchor_R': anchor['R'], 'anchor_weights': w.tolist(),
        'anchor_weekly_R': anchor['weekly_R'], 'reset_exact_match': True,
        'results': [], 'best': {'R': anchor['R'], 'weights': w.tolist(), 'weekly_R': anchor['weekly_R']}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.out.write_text(json.dumps(payload, indent=2)+'\n', encoding='utf-8')
    save()
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn'), initializer=worker_init) as pool:
        current = anchor
        for round_index in range(args.rounds):
            candidates, raw_count = generate(w, current['records'], args.n)
            print(f'round {round_index+1}: {raw_count} feasible boundaries; testing {len(candidates)}', flush=True)
            save()
            futures = [pool.submit(score_candidate, c) for c in candidates]
            for index, future in enumerate(as_completed(futures), 1):
                result = future.result()
                result['round'] = round_index + 1
                payload['results'].append(result)
                if result['R'] < payload['best']['R']:
                    payload['best'] = result
                save()
                print(f'{index}/{len(candidates)} R={result["R"]:.12f} best={payload["best"]["R"]:.12f} seconds={result["seconds"]:.1f}', flush=True)
            if payload['best']['R'] >= current['R'] - 1e-10:
                break
            w = np.asarray(payload['best']['weights'])
            current = simulate(w, capture=True)
    payload['elapsed_seconds'] = time.perf_counter() - started
    payload['results'].sort(key=lambda r:r['R'])
    save()
    print(json.dumps({'best': payload['best'], 'output': str(args.out)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
