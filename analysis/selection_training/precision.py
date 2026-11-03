"""Explicit, reversible canonicalization for new exports; never edits old weights."""
import json

import numpy as np

from sdm.simplex import check_weights


def canonical_weights(weights, decimals=8, threshold=0.):
    if decimals not in (4, 6, 8) or threshold < 0 or not np.isfinite(threshold):
        raise ValueError('unsupported precision or threshold')
    original = check_weights(weights)
    rows = original.copy().reshape(-1, 7)
    rows[rows < threshold] = 0.
    totals = rows.sum(axis=1)
    if (totals <= 0).any():
        raise ValueError('cleanup removed an entire building')
    rows /= totals[:, None]
    factor = 10**decimals
    units = np.floor(rows*factor+.5).astype(np.int64)
    largest = np.argmax(rows, axis=1)
    units[np.arange(len(rows)), largest] += factor-units.sum(axis=1)
    return check_weights((units/factor).reshape(original.shape))


def export_weights(path, weights, decimals=8):
    weights = check_weights(weights)
    canonical = canonical_weights(weights, decimals)
    if not np.array_equal(weights, canonical):
        raise ValueError('export input must already be canonical')
    path.parent.mkdir(parents=True, exist_ok=True)
    matrices = weights.reshape(-1, 3, 7)
    rows = ['[\n'+',\n'.join('  ['+', '.join(f'{x:.{decimals}f}' for x in row)+']'
                             for row in matrix)+'\n]' for matrix in matrices]
    path.write_text((rows[0] if len(rows) == 1 else '[\n'+',\n'.join(rows)+'\n]')+'\n')
    np.testing.assert_array_equal(check_weights(json.loads(path.read_text())), weights)


def prioritized_unique(weights, eligible, first):
    """Keep the designated feasible anchor's identity when another run exports the same W."""
    if first not in eligible:
        raise ValueError('fixed exploratory anchor fails canonical eligibility; do not substitute')
    order = np.r_[first, np.asarray(eligible)[np.asarray(eligible)!=first]]
    _,positions = np.unique(np.round(weights[order],9),axis=0,return_index=True)
    return order[np.sort(positions)]
