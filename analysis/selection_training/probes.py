"""Separate, fixed-policy exploration probes after the candidate-only pack exposed concentration."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from sdm.data import load_dataset
from sdm.evaluate import median_spacing, nearest_distances
from sdm.optimizers._neighborhood import exact_neighbors, shared_directions
from sdm.paths import REPO_ROOT
from sdm.simplex import check_weights

from audit import fitted_models, provenance
from precision import canonical_weights, export_weights
from run import RESULTS, write


def main():
    ds = load_dataset(REPO_ROOT/'dataset_same8')
    out = RESULTS/'direction_probes'
    out.mkdir(parents=True,exist_ok=True)
    runs = json.loads((RESULTS/'runs.json').read_text())
    provenance(out/'provenance.json',ds)
    eligible = np.argsort(ds.y,kind='stable')[:len(ds.y)//5]
    anchors = [int(np.argmin(ds.y))]
    while len(anchors)<20:
        remaining = np.setdiff1d(eligible,anchors)
        distance = np.linalg.norm(ds.X[remaining,None]-ds.X[np.array(anchors)][None],axis=2).min(axis=1)
        anchors.append(int(remaining[np.argmax(distance)]))
    spacing = median_spacing(ds.X)
    directions = shared_directions(2051,19)
    neighbors = exact_neighbors(ds.X[anchors[1:]],directions,.2*spacing)
    raw = np.stack([neighbors[i,i] for i in range(19)])
    fixed = check_weights(json.loads((RESULTS/'precision-new/canonical/corrected-search-2040-r1.json').read_text()))
    raw = check_weights(np.vstack([fixed,raw]))
    points = canonical_weights(raw)
    export_weights(out/'weights.json',points)
    write(out/'raw_weights.json',raw.reshape(-1,3,7).tolist())
    parsed = check_weights(json.loads((out/'weights.json').read_text()))
    np.testing.assert_array_equal(points,parsed)
    distances = nearest_distances(points,ds.X)
    assert (distances>=1e-6).all() and (distances/spacing<=.75).all()
    assert len(np.unique(np.round(points,9),axis=0))==20
    models,refs = fitted_models(runs)
    predictions = {name:model.predict(parsed) for name,model in models.items()}
    rows = []
    for i,anchor in enumerate(anchors):
        nearest = int(np.argmin(np.linalg.norm(ds.X-points[i],axis=1)))
        row = {'probe':f'p{i+1:02}','kind':'fixed_search_candidate' if i==0 else 'direction_probe',
            'source_anchor':ds.ids[anchor],'anchor_measured_R':ds.y[anchor],
            'candidate_measured_R':None,'distance_ratio':distances[i]/spacing,
            'actual_distance_to_source_anchor':np.linalg.norm(points[i]-ds.X[anchor]),
            'nearest_observed_id':ds.ids[nearest],'new_candidate_actual_R_known':False}
        for judge,(mean,scale) in predictions.items():
            row[judge+'_mean'],row[judge+'_error'] = mean[i],scale[i]
        rows.append(row)
        export_weights(out/f'weights/p{i+1:02}.json',points[i:i+1])
    pd.DataFrame(rows).to_csv(out/'manifest.csv',index=False)
    vectors = points-ds.X[anchors[0]]
    vectors /= np.linalg.norm(vectors,axis=1,keepdims=True)
    cosine = vectors@vectors.T
    pair = np.linalg.norm(points[:,None]-points[None],axis=2)
    triangle = np.triu_indices(20,1)
    write(out/'summary.json',{'policy_declared_in':'analysis/selection_training/README.md#adaptive-direction-only-extension',
        'seed':2051,'raw_probe_radius':.2*spacing,'exported':20,'novel':True,'feasible':True,
        'distinct_source_anchors':len(set(anchors)),
        'distinct_nearest_anchors':len(set(r['nearest_observed_id'] for r in rows)),
        'minimum_pair_distance':float(pair[triangle].min()),
        'median_pair_distance':float(np.median(pair[triangle])),
        'maximum_direction_cosine':float(cosine[triangle].max()),
        'median_direction_cosine':float(np.median(cosine[triangle])),
        'model_runs':refs,'prediction_screening':False,'simulator_calls':0,
        'is_replacement_online_packet':False})
    print((out/'summary.json').read_text(),flush=True)


if __name__ == '__main__':
    main()
