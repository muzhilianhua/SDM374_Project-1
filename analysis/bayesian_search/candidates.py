"""Fit trusted local models, search Bayesian acquisitions and export an explicitly unmeasured shortlist."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sdm.cli import execute
from sdm.data import load_dataset
from sdm.evaluate import median_spacing, nearest_distances
from sdm.runs import resolve_run
from sdm.simplex import check_weights

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'analysis/bayesian_search/delivery'


def write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def train(name):
    run=execute('train',ROOT/f'experiments/bayes-study-{name}.toml')
    print('TRAIN',name,run.id,flush=True)
    return name,run.id


def main():
    os.chdir(ROOT)
    # Fixed shortlist policy before inspecting any newly generated candidate predictions.
    write(OUT/'candidate_plan.json',dict(models=['gp-tree-blend','gp-ard'],
        acquisitions=['ei','mean','pessimistic'],points_per_search=10,seed=2026,
        rule='minimize worst panel mean delta relative to best observed anchor; spread five choices',
        caveat='Panel members are dependent; agreement and GP posterior spread are not real guarantees',
        real_measurements_added=0,simulator_calls=0))
    with ProcessPoolExecutor(max_workers=2) as pool:
        trained=dict(pool.map(train,['gp-tree-blend','gp-ard']))
    write(OUT/'training_runs.json',trained)
    search_records=[]
    points=[]; labels=[]
    for name,run_id in trained.items():
        original=(ROOT/f'experiments/bayes-study-{name}.toml').read_text(encoding='utf-8')
        for acquisition in ['ei','mean','pessimistic']:
            config=ROOT/f'experiments/bayes-search-{name}-{acquisition}.toml'
            config.write_text(original.replace('acquisition="ei"',f'acquisition="{acquisition}"'),encoding='utf-8')
            run=execute('optimize',config,n=10,model_run=run_id)
            batch=check_weights(json.loads((run.path/'weights.json').read_text()))
            for rank,w in enumerate(batch,1):
                points.append(w);labels.append(f'{name}-{acquisition}-r{rank:02d}')
            search_records.append(dict(model=name,acquisition=acquisition,run_id=run.id))
            print('SEARCH',name,acquisition,run.id,flush=True)
    write(OUT/'search_runs.json',search_records)
    data=load_dataset(ROOT/'dataset_same8')
    best=int(np.argmin(data.y)); anchor=data.X[best]
    controls={'best-observed':anchor,
              'course-baseline':check_weights(json.loads((ROOT/'weights.json').read_text()))[0],
              'submitted-119-approx':check_weights(json.loads((OUT/'submitted_weights.json').read_text()))[0],
              'old-operational':check_weights(json.loads((ROOT/'analysis/offline_submission/submission/weights.json').read_text()))[0]}
    for name,point in controls.items():
        points.append(point);labels.append(name)
    points=check_weights(np.array(points))
    # These are only our own freshly created models; no external pickle is deserialized.
    blend=resolve_run(trained['gp-tree-blend']).load_model()
    ard=resolve_run(trained['gp-ard']).load_model()
    panel={'gp-additive':blend.gp,'tree-stack':blend.tree,'gp-tree-blend':blend,'gp-ard':ard}
    frame=pd.DataFrame({'candidate':labels})
    all_deltas=[]
    for name,model in panel.items():
        means=[];stds=[]
        for start in range(0,len(points),64):
            m,s=model.predict(points[start:start+64]);means.extend(m);stds.extend(s)
        frame[name+'_mean']=means;frame[name+'_spread']=stds
        am,astd=model.predict(anchor[None])
        delta=np.array(means)-am[0]
        frame[name+'_delta_best_observed']=delta
        all_deltas.append(delta)
    frame['worst_mean_delta_best_observed']=np.max(all_deltas,axis=0)
    spacing=median_spacing(data.X)
    frame['distance_ratio']=nearest_distances(points,data.X)/spacing
    frame['distance_from_best_observed']=np.linalg.norm(points-anchor,axis=1)
    frame['measured_R']=np.nan
    frame.loc[frame.candidate=='best-observed','measured_R']=data.y[best]
    frame['reported_approximate_R']=np.nan
    frame.loc[frame.candidate=='submitted-119-approx','reported_approximate_R']=119
    frame.to_csv(OUT/'candidate_comparison.csv',index=False)
    selected=[]
    for index in np.argsort(frame.worst_mean_delta_best_observed.to_numpy()[:60],kind='stable'):
        if all(np.linalg.norm(points[index]-points[other])>=.15*spacing for other in selected):
            selected.append(int(index))
        if len(selected)==5:
            break
    write(OUT/'all_candidate_weights.json',dict(zip(labels,points.reshape(-1,3,7).tolist())))
    folder=OUT/'shortlist';folder.mkdir(exist_ok=True)
    for rank,index in enumerate(selected,1):
        write(folder/f'{rank:02d}_{labels[index]}.json',points[index].reshape(3,7).tolist())
    write(folder/'weights.json',points[selected].reshape(-1,3,7).tolist())
    frame.iloc[selected].to_csv(folder/'comparison.csv',index=False)
    write(OUT/'best_observed_115.7648.json',anchor.reshape(3,7).tolist())
    write(OUT/'candidate_decision.json',dict(selected=[labels[i] for i in selected],
        primary=labels[selected[0]],primary_actual_R=None,
        measured_fallback=dict(candidate_id=data.ids[best],historical_R=float(data.y[best]),
            caveat='Eligibility and consistency with current evaluator must be verified; not a new measured result'),
        measured_improvement_established=False,simulator_calls=0))
    print(frame.iloc[selected][['candidate','worst_mean_delta_best_observed','distance_ratio']].to_string(index=False),flush=True)


if __name__=='__main__':
    main()
