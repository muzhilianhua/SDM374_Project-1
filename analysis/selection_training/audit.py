"""Precision, measured-pool, shift, dual-stability and diversity audits with provenance."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from sdm.config import load_config
from sdm.data import load_dataset, sha256_of
from sdm.evaluate import median_spacing, nearest_distances
from sdm.models.selection_stack import measured_pools
from sdm.optimizers._neighborhood import exact_neighbors, shared_directions
from sdm.paths import REPO_ROOT
from sdm.registry import build_model
from sdm.runs import git_state, resolve_run
from sdm.simplex import check_weights

from precision import canonical_weights, export_weights, prioritized_unique
from run import BASE, RESULTS, write


def provenance(path, ds, extra=()):
    # The global manifest grows after later CLI phases; preserve exactly what this audit used.
    snapshot = path.parent/'run_manifest_snapshot.json'
    snapshot.write_bytes((RESULTS/'runs.json').read_bytes())
    sources = [*BASE.glob('*.py'), BASE/'README.md', snapshot,
        REPO_ROOT/'src/sdm/models/selection_stack.py', REPO_ROOT/'src/sdm/models/honest_stack.py',
        REPO_ROOT/'src/sdm/models/method_study.py', REPO_ROOT/'src/sdm/optimizers/_neighborhood.py',
        REPO_ROOT/'src/sdm/optimizers/neighborhood_trust.py', REPO_ROOT/'uv.lock', *extra]
    write(path, {'git': git_state(), 'dataset_sha256': ds.sha256,
        'packages': {p: version(p) for p in ('numpy','pandas','scipy','scikit-learn','catboost','xgboost')},
        'sources': {str(p.relative_to(REPO_ROOT)): sha256_of(p) for p in sources},
        'seeds': {'pools':2042,'directions':[2043,2044,2045],
                  'subsets':list(range(2046,2051)),'regions':2031}, 'simulator_calls':0})


class Chunked:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        pairs = [self.model.predict(X[i:i+256]) for i in range(0,len(X),256)]
        return tuple(np.concatenate([p[k] for p in pairs]) for k in (0,1))


def fitted_models(runs, legacy=True):
    refs = {name:runs[name+'-fit']['run'] for name in ('reference','corrected')}
    if legacy:
        refs.update(histgb='20261029-125052-optimize-histgb-v32',
                    xgboost='20261029-125052-optimize-xgboost-v32',
                    extratrees='20261029-125052-optimize-extratrees-default')
    return {name:Chunked(resolve_run(ref).load_model()) for name,ref in refs.items()}, refs


def historical_inputs(ds):
    path = REPO_ROOT/'analysis/common_judge/delivery/primary_weights.json'
    names, weights, sources = [], [], []
    for name,value in json.loads(path.read_text()).items():
        names.append('method-'+name)
        weights.append(check_weights(value)[0])
        sources.append(str(path.relative_to(REPO_ROOT))+'#'+name)
    files = {'operational': 'analysis/offline_submission/submission/weights.json',
        'challenger':'analysis/robust_search/delivery/challenger_weights.json',
        'previous-honest':'analysis/stable_training/delivery/calibrated_exploratory_weights.json',
        'previous-mild':'analysis/stable_training/delivery/exploratory_weights.json',
        'baseline':'weights.json'}
    for name,rel in files.items():
        names.append(name)
        weights.append(check_weights(json.loads((REPO_ROOT/rel).read_text()))[0])
        sources.append(rel)
    names.append('p01969')
    weights.append(ds.X[np.argmin(ds.y)])
    sources.append('dataset_same8/train.csv#p01969')
    return names, check_weights(np.array(weights)), sources


def search_inputs(runs):
    names, weights, sources = [], [], []
    for name in ('reference','corrected'):
        for seed in (2040,2041):
            label = f'{name}-search-{seed}'
            run = resolve_run(runs[label]['run'])
            points = check_weights(run.read_json('weights.json'))
            for rank,point in enumerate(points,1):
                names.append(label+f'-r{rank}')
                weights.append(point)
                sources.append(run.id+f'#{rank}')
    return names,check_weights(np.array(weights)),sources


def precision_audit(which):
    out = RESULTS/('precision-'+which)
    out.mkdir(parents=True,exist_ok=True)
    ds = load_dataset(REPO_ROOT/'dataset_same8')
    runs = json.loads((RESULTS/'runs.json').read_text())
    names,raw,sources = historical_inputs(ds) if which == 'old' else search_inputs(runs)
    models,refs = fitted_models(runs)
    provenance(out/'provenance.json', ds)
    write(out/'raw_inputs.json', {'names':names,'weights':raw.reshape(-1,3,7).tolist(),
        'sources':sources,'model_runs':refs})
    variants = {'raw':raw, **{f'd{d}':canonical_weights(raw,d) for d in (8,6,4)},
                'clean1e-6-d8':canonical_weights(raw,8,1e-6)}
    spacing = median_spacing(ds.X)
    rows = []
    for variant,points in variants.items():
        distance = nearest_distances(points,ds.X)
        prediction = {name:model.predict(points) for name,model in models.items()}
        for i,name in enumerate(names):
            for judge,(mean,scale) in prediction.items():
                rows.append({'candidate':name,'source':sources[i],'variant':variant,'judge':judge,
                    'mean':mean[i],'error_scale':scale[i],'risk':mean[i]+scale[i],
                    'distance_ratio':distance[i]/spacing,'novel':bool(distance[i]>=1e-6),
                    'feasible':True,'max_coordinate_change':np.max(abs(points[i]-raw[i])),
                    'vector_change':np.linalg.norm(points[i]-raw[i]),
                    'zero_count':int(np.sum(points[i]==0)),
                    'tiny_positive_count':int(np.sum((points[i]>0)&(points[i]<1e-6)))})
        for i,name in enumerate(names):
            if variant == 'd8':
                export_weights(out/'canonical'/f'{name}.json',points[i:i+1])
    frame = pd.DataFrame(rows)
    frame['mean_rank'] = frame.groupby(['variant','judge'])['mean'].rank(method='min')
    frame['risk_rank'] = frame.groupby(['variant','judge'])['risk'].rank(method='min')
    baseline = frame[frame.variant=='raw'][['candidate','judge','mean','risk','mean_rank','risk_rank']]
    frame = frame.merge(baseline,on=['candidate','judge'],suffixes=('','_raw'),validate='many_to_one')
    for col in ('mean','risk','mean_rank','risk_rank'):
        frame[col+'_change'] = frame[col]-frame[col+'_raw']
    frame.to_csv(out/'predictions.csv.gz',index=False)
    summary = frame.groupby(['variant','judge']).agg(
        maximum_mean_change=('mean_change',lambda x:abs(x).max()),
        maximum_risk_change=('risk_change',lambda x:abs(x).max()),
        maximum_mean_rank_change=('mean_rank_change',lambda x:abs(x).max()),
        maximum_coordinate_change=('max_coordinate_change','max'),
        feasible=('feasible','all'),novel_count=('novel','sum')).reset_index()
    summary.to_csv(out/'summary.csv',index=False)
    write(out/'counts.json', {variant:{'points':len(points),
        'distinct_9dp':len(np.unique(np.round(points,9),axis=0)),
        'within_trust':int(np.sum(nearest_distances(points,ds.X)/spacing<=.75)),
        'tiny_positive_values':int(np.sum((points>0)&(points<1e-6))),
        'min_positive':float(points[points>0].min())} for variant,points in variants.items()})
    print('Precision '+which+': '+str(len(names))+' candidates',flush=True)


def measured_checks():
    out = RESULTS/'measured'
    out.mkdir(parents=True,exist_ok=True)
    ds = load_dataset(REPO_ROOT/'dataset_same8')
    runs = json.loads((RESULTS/'runs.json').read_text())
    provenance(out/'provenance.json',ds)
    rows,folds = [],[]
    for seed in (2026,2027,2028):
        for name in ('reference','corrected'):
            run = resolve_run(runs[f'{name}-{seed}']['run'])
            frame = pd.read_csv(run.path/'predictions.csv')
            for index,(split,part) in enumerate(frame.groupby('split_id',sort=False)):
                part = part.sort_values('candidate_id',kind='stable').reset_index(drop=True)
                pools = [np.arange(len(part))]+measured_pools(len(part),2042+index)
                for pool_index,pool in enumerate(pools):
                    for rule in ('mean','risk'):
                        score = part.predicted_mean.to_numpy()
                        if rule=='risk':
                            score = score+part.predicted_std.to_numpy()
                        choice = pool[np.argmin(score[pool])]
                        picked = part.iloc[choice]
                        record = {'seed':seed,'model':name,'run':run.id,'split_id':split,
                            'pool':'whole' if pool_index==0 else str(pool_index),
                            'pool_size':len(pool),'rule':rule,'selected_id':picked.candidate_id,
                            'selected_actual_R':picked.actual,'pool_best_R':part.iloc[pool].actual.min(),
                            'regret':picked.actual-part.iloc[pool].actual.min(),
                            'selected_prediction':picked.predicted_mean,
                            'signed_optimism':picked.actual-picked.predicted_mean}
                        (folds if pool_index==0 else rows).append(record)
    pd.DataFrame(rows).to_csv(out/'pool_selection.csv',index=False)
    pd.DataFrame(folds).to_csv(out/'fold_selection.csv',index=False)
    pd.DataFrame(rows).groupby(['model','rule']).agg(
        selected_actual_R=('selected_actual_R','mean'),regret=('regret','mean'),
        signed_optimism=('signed_optimism','mean'),pools=('pool','count')).reset_index().to_csv(
        out/'pool_summary.csv',index=False)


def region_checks():
    out = RESULTS/'regions'
    out.mkdir(parents=True,exist_ok=True)
    ds = load_dataset(REPO_ROOT/'dataset_same8')
    provenance(out/'provenance.json',ds)
    regions = KMeans(n_clusters=3,n_init=10,random_state=2031).fit_predict(ds.X)
    pd.DataFrame({'candidate_id':ds.ids,'region':regions}).to_csv(out/'regions.csv',index=False)
    predictions,selection = [],[]
    for region in range(3):
        train,test = np.flatnonzero(regions!=region),np.flatnonzero(regions==region)
        ratio = nearest_distances(ds.X[test],ds.X[train])/median_spacing(ds.X[train])
        for name in ('reference','corrected'):
            print(f'Region {region}, {name}',flush=True)
            cfg = load_config(REPO_ROOT/f'experiments/selection-{name}-2026.toml')
            model = build_model(cfg.model.name,cfg.model.params).fit(ds.X[train],ds.y[train])
            mean,scale = Chunked(model).predict(ds.X[test])
            for i,index in enumerate(test):
                predictions.append({'region':region,'model':name,'candidate_id':ds.ids[index],
                    'actual':ds.y[index],'mean':mean[i],'error_scale':scale[i],'distance_ratio':ratio[i]})
            for support,pool in [('all',np.arange(len(test))),('ratio<=.75',np.flatnonzero(ratio<=.75))]:
                for rule,score in [('mean',mean),('risk',mean+scale)]:
                    row = {'region':region,'model':name,'support':support,'rule':rule,'pool_size':len(pool),
                        'mae':float(np.mean(abs(ds.y[test]-mean))),
                        'strength':float(getattr(model,'strength_',0.))}
                    if len(pool):
                        chosen = pool[np.argmin(score[pool])]
                        row.update(selected_id=ds.ids[test[chosen]],selected_actual_R=ds.y[test[chosen]],
                            regret=ds.y[test[chosen]]-ds.y[test[pool]].min(),
                            signed_optimism=ds.y[test[chosen]]-mean[chosen])
                    selection.append(row)
            pd.DataFrame(predictions).to_csv(out/'predictions.csv.gz',index=False)
            pd.DataFrame(selection).to_csv(out/'selection.csv',index=False)


def candidate_checks():
    out = RESULTS/'candidates'
    out.mkdir(parents=True,exist_ok=True)
    ds = load_dataset(REPO_ROOT/'dataset_same8')
    runs = json.loads((RESULTS/'runs.json').read_text())
    names,raw,sources = historical_inputs(ds)
    keep = [names.index(n) for n in ('challenger','operational','previous-honest','baseline','p01969')]
    names,points,sources = [names[i] for i in keep],raw[keep],[sources[i] for i in keep]
    new_names,new_raw,new_sources = search_inputs(runs)
    for label in ('reference-search-2040-r1','corrected-search-2040-r1'):
        i = new_names.index(label)
        names.append(label)
        points = np.vstack([points,canonical_weights(new_raw[i:i+1])])
        sources.append(new_sources[i]+'#canonical8')
    check_weights(points)
    models,refs = fitted_models(runs)
    provenance(out/'provenance.json',ds)
    write(out/'inputs.json',{'names':names,'sources':sources,'weights':points.reshape(-1,3,7).tolist(),
                            'model_runs':refs})
    center = {name:model.predict(points) for name,model in models.items()}
    anchor = names.index('p01969')
    central = []
    for i,name in enumerate(names):
        row = {'candidate':name,'source':sources[i],
            'distance_ratio':nearest_distances(points[i:i+1],ds.X)[0]/median_spacing(ds.X)}
        for judge,(mean,scale) in center.items():
            row[judge+'_mean'],row[judge+'_error'] = mean[i],scale[i]
            row[judge+'_delta'] = mean[i]-mean[anchor]
        corrected = models['corrected'].model
        raw_mean,features = corrected.context_.features(corrected.predictor_,points[i:i+1])
        normalized = (features-corrected.correction_.location)/corrected.correction_.scale
        row['correction_feature_max_abs_z'] = np.max(abs(normalized))
        row['raw_stack_mean'] = raw_mean[0]
        row['applied_correction'] = row['corrected_mean']-raw_mean[0]
        row['correction_strength'] = corrected.strength_
        central.append(row)
    pd.DataFrame(central).to_csv(out/'central.csv',index=False)
    neighbor = []
    for radius in (.02,.05):
        for seed in (2043,2044,2045):
            weights = exact_neighbors(points,shared_directions(seed,128),radius)
            pred = {name:model.predict(weights.reshape(-1,21)) for name,model in models.items()}
            np.savez_compressed(out/f'neighbors-{radius}-{seed}.npz',weights=weights,
                **{name+'_'+k:value[j].reshape(len(points),-1)
                   for name,value in pred.items() for j,k in enumerate(('mean','error'))})
            for i,name in enumerate(names):
                row = {'candidate':name,'radius':radius,'seed':seed}
                deltas = []
                for judge,(mean,scale) in pred.items():
                    m,s = mean.reshape(len(points),-1)[i],scale.reshape(len(points),-1)[i]
                    delta = m-center[judge][0][anchor]
                    row[judge+'_delta_median'] = np.median(delta)
                    row[judge+'_q80_risk'] = np.quantile(m+s,.8)
                    if judge in ('reference','histgb','xgboost'):
                        deltas.append(delta)
                row['legacy_consensus_success'] = np.mean(np.max(deltas,axis=0)<-3.)
                neighbor.append(row)
            print(f'Neighborhood {radius}, seed{seed}',flush=True)
            pd.DataFrame(neighbor).to_csv(out/'neighbors.csv',index=False)
    subset_rows = []
    for seed in range(2046,2051):
        chosen = np.sort(np.random.default_rng(seed).choice(len(ds.y),int(.9*len(ds.y)),replace=False))
        def fit(name):
            cfg = load_config(REPO_ROOT/f'experiments/selection-{name}-2026.toml')
            return name,build_model(cfg.model.name,cfg.model.params).fit(ds.X[chosen],ds.y[chosen])
        with ThreadPoolExecutor(max_workers=2) as pool:
            fits = list(pool.map(fit,('reference','corrected')))
        write(out/f'subset-{seed}.json',{'seed':seed,'indices':chosen.tolist(),
            'strength':{name:getattr(model,'strength_',0.) for name,model in fits},
            'metrics':{name:model.metrics for name,model in fits}})
        for name,model in fits:
            mean,scale = model.predict(points)
            risk = mean+scale
            incumbent = names.index('operational')
            for i,candidate in enumerate(names):
                subset_rows.append({'seed':seed,'model':name,'candidate':candidate,'mean':mean[i],
                    'error_scale':scale[i],'risk':risk[i],
                    'delta_vs_p01969':mean[i]-mean[anchor],
                    'risk_delta_vs_p01969':risk[i]-risk[anchor],
                    'risk_delta_vs_operational':risk[i]-risk[incumbent]})
        pd.DataFrame(subset_rows).to_csv(out/'training_subsets.csv',index=False)
        print(f'Matched90% training subset {seed} finished',flush=True)
    pd.DataFrame(subset_rows).groupby(['model','candidate']).agg(
        mean_R=('mean','mean'),mean_std=('mean','std'),
        risk_delta_operational=('risk_delta_vs_operational','mean'),
        risk_improved_subsets=('risk_delta_vs_operational',lambda x:int(np.sum(x < -1e-9)))).reset_index().to_csv(
        out/'training_subset_summary.csv',index=False)


def diversity():
    out = RESULTS/'diversity'
    out.mkdir(parents=True,exist_ok=True)
    ds = load_dataset(REPO_ROOT/'dataset_same8')
    runs = json.loads((RESULTS/'runs.json').read_text())
    provenance(out/'provenance.json',ds)
    names,raw,sources = search_inputs(runs)
    points = canonical_weights(raw)
    distance = nearest_distances(points,ds.X)
    spacing = median_spacing(ds.X)
    eligible = np.flatnonzero((distance>=1e-6)&(distance/spacing<=.75))
    first = names.index('corrected-search-2040-r1')
    eligible = prioritized_unique(points,eligible,first)
    anchor = ds.X[np.argmin(ds.y)]
    directions = points-anchor
    directions /= np.linalg.norm(directions,axis=1,keepdims=True)
    selected = [first]
    while len(selected)<min(20,len(eligible)):
        remaining = np.setdiff1d(eligible,selected)
        minimum = np.linalg.norm(directions[remaining,None,:]-directions[np.array(selected)][None,:,:],axis=2).min(axis=1)
        selected.append(int(remaining[np.argmax(minimum)]))
    pack = points[selected]
    export_weights(out/'weights.json',pack)
    models,refs = fitted_models(runs)
    predictions = {name:model.predict(pack) for name,model in models.items()}
    rows = []
    for rank,i in enumerate(selected):
        nearest = np.argmin(np.linalg.norm(ds.X-points[i],axis=1))
        row = {'rank':rank+1,'candidate':names[i],'source':sources[i],
               'distance_ratio':distance[i]/spacing,'nearest_observed_id':ds.ids[nearest]}
        for judge,(mean,scale) in predictions.items():
            row[judge+'_mean'],row[judge+'_error'] = mean[rank],scale[rank]
        rows.append(row)
        export_weights(out/f'weights/d{rank+1:02}.json',points[i:i+1])
    pd.DataFrame(rows).to_csv(out/'manifest.csv',index=False)
    pair = np.linalg.norm(pack[:,None]-pack[None,:],axis=2)
    angular = directions[selected]@directions[selected].T
    triangle = np.triu_indices(len(selected),1)
    write(out/'summary.json',{'input_rows':len(points),'eligible_distinct':len(eligible),'exported':len(pack),
        'minimum_pair_distance':float(pair[triangle].min()),
        'median_pair_distance':float(np.median(pair[triangle])),
        'maximum_direction_cosine':float(angular[triangle].max()),
        'median_direction_cosine':float(np.median(angular[triangle])),
        'distinct_nearest_anchors':len(set(r['nearest_observed_id'] for r in rows)),
        'model_runs':refs,'is_replacement_online_packet':False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('precision-old','precision-new','measured','regions','candidates','diversity'))
    phase = parser.parse_args().phase
    if phase.startswith('precision-'):
        precision_audit(phase.split('-')[1])
    else:
        {'measured':measured_checks,'regions':region_checks,'candidates':candidate_checks,'diversity':diversity}[phase]()
