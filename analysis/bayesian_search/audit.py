"""Historical selection diagnostics and fixed-shortlist robustness checks; no invented simulator labels."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sdm.data import load_dataset
from sdm.models.gp_tree_blend import GPTreeBlend
from sdm.runs import resolve_run
from sdm.simplex import check_weights, project

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'analysis/bayesian_search/delivery'


def subset_audit(seed):
    data=load_dataset(ROOT/'dataset_same8')
    rng=np.random.default_rng(seed)
    train=np.sort(rng.choice(len(data.y),size=int(.9*len(data.y)),replace=False))
    model=GPTreeBlend().fit(data.X[train],data.y[train])
    weights=json.loads((OUT/'all_candidate_weights.json').read_text())
    names=json.loads((OUT/'candidate_decision.json').read_text())['selected']+['best-observed','submitted-119-approx']
    points=check_weights([weights[n] for n in names])
    rows=[]
    for judge,member in [('gp-additive',model.gp),('tree-stack',model.tree),('gp-tree-blend',model)]:
        mean,std=member.predict(points)
        reference=mean[names.index('best-observed')]
        for name,m,s in zip(names,mean,std):
            rows.append(dict(seed=seed,judge=judge,candidate=name,mean=m,spread=s,delta_best_observed=m-reference))
    return rows


def main():
    os.chdir(ROOT)
    (OUT/'audit_plan.json').write_text(json.dumps(dict(
        subset_seeds=[2061,2062,2063],subset_fraction=.9,
        shortlist_fixed_before_audit=True,replace_primary_from_audit=False,
        perturbation_radii=[.01,.03],directions_per_radius=128,seed=2064,
        caveat='Overlapping subsets and local surrogate stability are not independent simulator validation',
        simulator_calls=0),indent=2),encoding='utf-8')
    head=(ROOT/'.git/HEAD').read_text().strip()
    commit=(ROOT/'.git'/head.removeprefix('ref: ')).read_text().strip()
    (OUT/'audit_provenance.json').write_text(json.dumps(dict(commit=commit,branch=head,
        dataset_sha256=load_dataset(ROOT/'dataset_same8').sha256,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_model_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT/'src/sdm/models').glob('*.py')},seeds=[2061,2062,2063,2064]),indent=2),encoding='utf-8')
    records=json.loads((OUT/'evaluation.json').read_text())
    extra=OUT/'extension/evaluation.json'
    if extra.exists():records+=json.loads(extra.read_text())
    pd.DataFrame(records).sort_values('v32_score',ascending=False).to_csv(OUT/'model_comparison.csv',index=False)
    historical=[];pools=[]
    for record in records:
        predictions=pd.read_csv(ROOT/'runs'/record['run_id']/'predictions.csv',dtype={'candidate_id':str})
        for split,part in predictions.groupby('split_id',sort=True):
            part=part.sort_values('candidate_id',kind='stable').reset_index(drop=True)
            chosen=part.sort_values(['predicted_mean','candidate_id'],kind='stable').iloc[0]
            tail=part.nsmallest(max(1,len(part)//10),'predicted_mean')
            historical.append(dict(model=record['model'],split=split,selected_actual_R=chosen.actual,
                regret=chosen.actual-part.actual.min(),selected_prediction=chosen.predicted_mean,
                selected_optimism=chosen.actual-chosen.predicted_mean,
                predicted_tail_optimism=(tail.actual-tail.predicted_mean).mean(),
                predicted_tail_mae=(tail.actual-tail.predicted_mean).abs().mean()))
            permutation=np.random.default_rng(2065).permutation(len(part))
            for pool,indices in enumerate(np.array_split(permutation,8)):
                subset=part.iloc[indices]
                choice=subset.sort_values(['predicted_mean','candidate_id'],kind='stable').iloc[0]
                pools.append(dict(model=record['model'],split=split,pool=pool,selected_actual_R=choice.actual,
                    regret=choice.actual-subset.actual.min(),optimism=choice.actual-choice.predicted_mean))
    hist=pd.DataFrame(historical);hist.to_csv(OUT/'historical_fold_selection.csv',index=False)
    hist.groupby('model').mean(numeric_only=True).to_csv(OUT/'historical_selection_summary.csv')
    pools=pd.DataFrame(pools);pools.to_csv(OUT/'historical_pools.csv',index=False)
    pools.groupby('model').mean(numeric_only=True).to_csv(OUT/'historical_pool_summary.csv')
    trained=json.loads((OUT/'training_runs.json').read_text())
    blend=resolve_run(trained['gp-tree-blend']).load_model()
    ard=resolve_run(trained['gp-ard']).load_model()
    panel={'gp-additive':blend.gp,'tree-stack':blend.tree,'gp-ard':ard}
    weights=json.loads((OUT/'all_candidate_weights.json').read_text())
    names=json.loads((OUT/'candidate_decision.json').read_text())['selected']+['best-observed','submitted-119-approx']
    stability=[];rng=np.random.default_rng(2064)
    for name in names:
        central=check_weights(weights[name])[0]
        for radius in [.01,.03]:
            direction=rng.normal(size=(128,3,7));direction-=direction.mean(axis=2,keepdims=True)
            direction=direction.reshape(128,21);direction/=np.linalg.norm(direction,axis=1,keepdims=True)
            points=project(central+radius*direction)
            actual_radius=np.linalg.norm(points-central,axis=1)
            for judge,model in panel.items():
                mean,_=model.predict(points)
                m,_=model.predict(central[None])
                stability.append(dict(candidate=name,judge=judge,nominal_radius=radius,
                    actual_mean_radius=actual_radius.mean(),central_mean=m[0],mean=mean.mean(),
                    q90=float(np.quantile(mean,.9)),max_shift=float(np.max(np.abs(mean-m[0])))))
    pd.DataFrame(stability).to_csv(OUT/'local_stability.csv',index=False)
    print('local stability complete',flush=True)
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for part in pool.map(subset_audit,[2061,2062,2063]):rows+=part
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'subset_audit.csv',index=False)
    frame.groupby(['candidate','judge']).agg(mean_delta=('delta_best_observed','mean'),
        worst_delta=('delta_best_observed','max'),improved=('delta_best_observed',lambda a:int((a<0).sum()))
        ).to_csv(OUT/'subset_summary.csv')
    print('subset audit complete',flush=True)


if __name__=='__main__':main()
