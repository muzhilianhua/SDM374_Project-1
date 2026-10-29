"""Prepare and verify a reproducible 50-point online portfolio; never invoke a real evaluator."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
from sdm.data import load_dataset, sha256_of
from sdm.evaluate import median_spacing, nearest_distances
from sdm.paths import REPO_ROOT
from sdm.runs import PACKAGES, git_state, resolve_run
from sdm.simplex import FEATURES, check_weights

SEED=20261029
MODELS={'stack':'artifacts/best_model',
        'catboost':'20261029-171329-train-method-cat-weighted',
        'histgb':'20261029-125052-optimize-histgb-v32',
        'xgboost':'20261029-125052-optimize-xgboost-v32',
        'extratrees':'20261029-125052-optimize-extratrees-default'}
ANCHORS=[('fixed-blend-primary','analysis/common_judge/delivery/exported_weights/primary/blend-fixed.json'),
         ('fixed-blend-posthoc','analysis/common_judge/delivery/exported_weights/posthoc/blend-fixed.json'),
         ('operational','analysis/offline_submission/submission/weights.json'),
         ('consensus-challenger','analysis/robust_search/delivery/challenger_weights.json')]


def read(path):
    text=gzip.open(path,'rt').read() if str(path).endswith('.gz') else path.read_text()
    return json.loads(text,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))


def write(path,value):
    text=json.dumps(value,indent=2,allow_nan=False)+'\n'
    if str(path).endswith('.gz'):
        with gzip.open(path,'wt') as f:f.write(text)
    else:path.write_text(text)


def round_simplex(points):
    rows=check_weights(points).reshape(-1,7)
    units=np.rint(rows*1e8).astype(np.int64)
    units[np.arange(len(rows)),rows.argmax(axis=1)] += 100_000_000-units.sum(axis=1)
    return check_weights((units/1e8).reshape(-1,21))


def spread_pick(points,order,selected,count,separation):
    chosen=[]
    for i in order:
        if i in selected or i in chosen:continue
        previous=selected+chosen
        if not previous or np.linalg.norm(points[previous]-points[i],axis=1).min()>=separation:
            chosen.append(int(i))
            if len(chosen)==count:return chosen
    raise ValueError(f'Only {len(chosen)} separated points available; need {count}')


def exploration_pick(points,indices,selected,count,utility,spacing,separation):
    chosen=[]
    for _ in range(count):
        previous=selected+chosen
        distance=np.min(np.linalg.norm(points[indices,None]-points[previous],axis=2),axis=1)
        score=np.where(distance>=separation,utility[indices]+2*distance/spacing,-np.inf)
        if not np.isfinite(score).any():raise ValueError('Insufficient separated exploration points')
        chosen.append(int(indices[int(np.argmax(score))]))
    return chosen


def score_pool(label,reference,out):
    command=[str(Path(sys.executable).with_name('sdm')),'predict','--model-run',reference,
             '--weights',str((out/'pool_weights.json').resolve())]
    result=subprocess.run(command,cwd=REPO_ROOT,text=True,capture_output=True,check=True)
    pred=json.loads(result.stdout)
    write(out/f'predictions-{label}.json.gz',pred)
    return label,pred,command


def build(out):
    out.mkdir(parents=True,exist_ok=False)
    dataset=load_dataset(REPO_ROOT/'dataset_same8')
    original=read(REPO_ROOT/'analysis/common_judge/delivery/inputs.json')
    points=[];meta=[]
    def add(label,weights,kind,source):
        points.append(check_weights(weights)[0]);meta.append({'origin':label,'pool_kind':kind,'source':source})
    for label,path in ANCHORS:add(label,read(REPO_ROOT/path),'anchor',path)
    best=int(dataset.y.argmin());assert dataset.ids[best]=='p01969'
    add('course-baseline',read(REPO_ROOT/'weights.json'),'control','weights.json')
    add('p01969-calibration-repeat',dataset.X[best],'control','dataset_same8:p01969')
    add('uniform',np.full((3,7),1/7),'control','equal objectives in every building')
    add('pure-ramping',np.tile([0,0,0,0,0,1,0],(3,1)),'control','ramping objective only')
    add('ramping-resilience-half',np.tile([0,0,0,0,0,.5,.5],(3,1)),'control','equal ramping and resilience')
    for row,w in zip(original['rows'],original['weights']):
        if row['kind']=='proposal':add(row['label'],w,'prior-proposal',f"{row['source_run']}:rank{row['rank']}")
    assert len(points)==1509
    rng=np.random.default_rng(SEED)
    centers=np.vstack([dataset.X[np.argsort(dataset.y,kind='stable')[:64]],points[:4]])
    chosen=centers[rng.integers(0,len(centers),size=3600)].reshape(-1,3,7)
    concentration=rng.choice([30.,100.,300.],size=(3600,1,1))
    draws=rng.gamma(concentration*chosen+.05);draws/=draws.sum(axis=2,keepdims=True)
    local=round_simplex(draws)
    global_points=round_simplex(rng.dirichlet(np.full(7,.6),size=(600,3)))
    for i,w in enumerate(local):add('local-dirichlet',w,'new-local',f'seed{SEED}:draw{i}')
    for i,w in enumerate(global_points):add('coverage-dirichlet',w,'new-coverage',f'seed{SEED}:draw{i}')
    X=check_weights(np.asarray(points));spacing=median_spacing(dataset.X)
    distance=nearest_distances(X,dataset.X)
    frame=pd.DataFrame(meta);frame['point_index']=np.arange(len(X))
    frame['nearest_train_distance']=distance;frame['distance_ratio']=distance/spacing
    (out/'pool_weights.json').write_text(json.dumps(X.reshape(-1,3,7).tolist(),separators=(',',':'))+'\n')
    predictions={};commands=[];evidence={};hashes={}
    for label,ref in MODELS.items():
        run=resolve_run(ref);metadata=run.read_json('meta.json')
        assert metadata['dataset']['sha256']==dataset.sha256 and not metadata['git']['dirty']
        evidence[label]={'reference':ref,'meta':metadata,'config':run.config_path.read_text(),
                         'metrics':run.read_json('metrics.json')}
        hashes[str(run.path/'model.pkl')]=sha256_of(run.path/'model.pkl')
    with ThreadPoolExecutor(2) as executor:
        for label,pred,command in executor.map(lambda item:score_pool(*item,out),MODELS.items()):
            for key in ['predicted_mean','predicted_std']:
                assert len(pred[key])==len(X) and np.isfinite(pred[key]).all()
            frame[label+'_mean']=pred['predicted_mean'];frame[label+'_proxy']=pred['predicted_std']
            predictions[label]=pred;commands.append(command)
    strong=['stack','catboost','histgb','xgboost']
    frame['committee_risk']=np.maximum.reduce([frame[j+'_mean']+frame[j+'_proxy'] for j in strong])
    frame['committee_disagreement']=frame[[j+'_mean' for j in strong]].std(axis=1,ddof=0)
    novel=distance>1e-6;separation=.25*spacing
    controls=list(range(4,9));anchors=list(range(4))
    for i in anchors:
        assert novel[i] and frame.loc[i,'distance_ratio']<=.75
        assert np.linalg.norm(X[controls+[j for j in anchors if j!=i]]-X[i],axis=1).min()>=separation
    exploit_order=frame[(frame.distance_ratio<=.75)&novel].sort_values(
        ['committee_risk','stack_mean','point_index'],kind='stable').index.to_numpy()
    exploit=anchors+spread_pick(X,exploit_order,controls+anchors,26,separation)
    eligible=frame[(frame.distance_ratio<=1)&novel&(frame.stack_mean<=frame.loc[4,'stack_mean']+15)].index.to_numpy()
    utility=(frame.committee_disagreement+.5*frame.stack_proxy).to_numpy()
    explore=exploration_pick(X,eligible,controls+exploit,15,utility,spacing,separation)
    indices=exploit+explore+controls
    assert len(indices)==len(set(indices))==50
    batch=check_weights(X[indices]);nearest_pair=np.linalg.norm(batch[:,None]-batch[None,:],axis=2)
    np.fill_diagonal(nearest_pair,np.inf);assert nearest_pair.min()>=separation
    selected=frame.loc[indices].copy().reset_index(drop=True)
    selected.insert(0,'candidate_id',[f'r01-{i:03}' for i in range(1,51)])
    selected.insert(1,'strategy',['exploit']*30+['explore']*15+['control']*5)
    selected['intentional_offline_repeat']=selected.origin=='p01969-calibration-repeat'
    assert (selected.loc[~selected.intentional_offline_repeat,'nearest_train_distance']>1e-6).all()
    selected['known_offline_R']=[float(dataset.y[best]) if v else None for v in selected.intentional_offline_repeat]
    write(out/'weights.json',batch.reshape(50,3,7).tolist())
    folder=out/'weights';folder.mkdir()
    for candidate,w in zip(selected.candidate_id,batch):write(folder/f'{candidate}.json',w.reshape(3,7).tolist())
    candidates=pd.DataFrame(batch,columns=FEATURES);candidates.insert(0,'candidate_id',selected.candidate_id)
    candidates.to_csv(out/'candidates.csv',index=False,float_format='%.17g')
    candidates.rename(columns={name:f'w{i}' for i,name in enumerate(FEATURES)}).to_csv(
        out/'weights_w0_w20.csv',index=False,float_format='%.17g')
    selected.to_csv(out/'plan.csv',index=False,float_format='%.17g')
    pending=candidates.copy();pending.insert(0,'round',1)
    pending['measured_R']='';pending['evaluator_version']='';pending['status']='pending';pending['notes']=''
    pending.to_csv(out/'pending_results.csv',index=False,float_format='%.17g')
    frame.to_csv(out/'source_pool.csv.gz',index=False,compression='gzip')
    with open(out/'pool_weights.json','rb') as source,gzip.open(out/'pool_weights.json.gz','wb') as target:
        target.write(source.read())
    (out/'pool_weights.json').unlink()
    write(out/'model_evidence.json',evidence)
    pdf=REPO_ROOT.parent/'SDM374_Project1_v3.pdf'
    source_paths=['analysis/common_judge/delivery/inputs.json','weights.json',*[p for _,p in ANCHORS]]
    write(out/'provenance.json',{'git':git_state(),'dataset_sha256':dataset.sha256,'seed':SEED,
        'python':platform.python_version(),'packages':{p:version(p) for p in (*PACKAGES,'catboost')},
        'source_pdf':{'name':pdf.name,'sha256':sha256_of(pdf),'constraint_page':3,'budget_page':7},
        'source_hashes':{p:sha256_of(REPO_ROOT/p) for p in source_paths},'model_hashes':hashes,
        'prediction_commands':commands,'training_spacing':spacing,'min_batch_separation':separation,
        'planned_round':1,'planned_evaluations':50,'maximum_rounds':20,'maximum_per_round':50,
        'simulator_calls':0,'known_row_repeats':1,'ratio_basis':'first-round heuristic, not replay validated'})
    print(selected[['candidate_id','strategy','origin','stack_mean','distance_ratio']].to_string(index=False))


def verify(out,replay=False):
    W=check_weights(read(out/'weights.json'))
    assert W.shape==(50,21)
    plan=pd.read_csv(out/'plan.csv');csv=pd.read_csv(out/'candidates.csv',float_precision='round_trip')
    assert plan.candidate_id.tolist()==[f'r01-{i:03}' for i in range(1,51)]==csv.candidate_id.tolist()
    assert plan.strategy.value_counts().to_dict()=={'exploit':30,'explore':15,'control':5}
    np.testing.assert_array_equal(csv[list(FEATURES)].to_numpy(),W)
    ppt_csv=pd.read_csv(out/'weights_w0_w20.csv',float_precision='round_trip')
    np.testing.assert_array_equal(ppt_csv[[f'w{i}' for i in range(21)]].to_numpy(),W)
    assert ppt_csv.candidate_id.tolist()==plan.candidate_id.tolist()
    assert len(list((out/'weights').glob('*.json')))==50
    for row,w in zip(plan.itertuples(),W):np.testing.assert_array_equal(check_weights(read(out/'weights'/f'{row.candidate_id}.json'))[0],w)
    matrix=np.linalg.norm(W[:,None]-W[None,:],axis=2);np.fill_diagonal(matrix,np.inf)
    provenance=read(out/'provenance.json');assert matrix.min()>=provenance['min_batch_separation']
    dataset=load_dataset(REPO_ROOT/'dataset_same8');assert dataset.sha256==provenance['dataset_sha256']
    distances=nearest_distances(W,dataset.X)
    assert (distances[~plan.intentional_offline_repeat]>1e-6).all()
    assert int(plan.intentional_offline_repeat.sum())==1
    for label in ['exploit','explore']:
        assert (plan.loc[plan.strategy==label,'distance_ratio']<=(.75 if label=='exploit' else 1)+1e-9).all()
    pool=check_weights(read(out/'pool_weights.json.gz'))
    for label in MODELS:
        pred=read(out/f'predictions-{label}.json.gz')
        for field,key in [('mean','predicted_mean'),('proxy','predicted_std')]:
            np.testing.assert_allclose(plan[label+'_'+field],np.asarray(pred[key])[plan.point_index],rtol=1e-12)
    np.testing.assert_array_equal(W,pool[plan.point_index])
    np.testing.assert_array_equal(W[45],check_weights(read(REPO_ROOT/'weights.json'))[0])
    for i,(_,path) in enumerate(ANCHORS):np.testing.assert_array_equal(W[i],check_weights(read(REPO_ROOT/path))[0])
    best=int(dataset.y.argmin());np.testing.assert_array_equal(W[46],dataset.X[best])
    np.testing.assert_array_equal(W[47],np.full(21,1/7))
    np.testing.assert_array_equal(W[48],np.tile([0,0,0,0,0,1,0],3))
    np.testing.assert_array_equal(W[49],np.tile([0,0,0,0,0,.5,.5],3))
    pending=pd.read_csv(out/'pending_results.csv',float_precision='round_trip')
    assert len(pending)==50 and (pending['round']==1).all() and (pending.status=='pending').all()
    assert pending.measured_R.isna().all()
    assert pending.candidate_id.tolist()==plan.candidate_id.tolist()
    np.testing.assert_array_equal(pending[list(FEATURES)].to_numpy(),W)
    if replay:
        command=[str(Path(sys.executable).with_name('sdm')),'predict','--model-run',MODELS['stack'],
                 '--weights',str((out/'weights.json').resolve())]
        result=subprocess.run(command,cwd=REPO_ROOT,text=True,capture_output=True,check=True)
        actual=json.loads(result.stdout)
        np.testing.assert_allclose(actual['predicted_mean'],plan.stack_mean,rtol=1e-9,atol=1e-9)
        np.testing.assert_allclose(actual['predicted_std'],plan.stack_proxy,rtol=1e-9,atol=1e-9)
    if (out/'checksums.json').exists():
        for name,digest in read(out/'checksums.json').items():assert sha256_of(out/name)==digest,name
    record={'valid':True,'batch_count':50,'shape':[50,3,7],'nonnegative':True,'each_building_has_positive_weight':True,
        'max_row_sum_error':float(np.abs(W.reshape(50,3,7).sum(axis=2)-1).max()),
        'unique_within_batch':50,'minimum_pair_distance':float(matrix.min()),
        'strategy_counts':{'exploit':30,'explore':15,'control':5},'intentional_offline_repeats':1,
        'offline_novel_points':49,'csv_json_individual_files_match':True,'all_results_pending':True,
        'stack_cli_replay':replay,'simulator_calls':0}
    print(json.dumps(record,indent=2));return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['build','verify']);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--replay-cli',action='store_true')
    args=parser.parse_args()
    if args.phase=='build':build(args.out)
    else:verify(args.out,args.replay_cli)


if __name__=='__main__':main()
