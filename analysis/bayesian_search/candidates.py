"""Reviewed offline research pool; no solver screening and no official submission2 selection."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sdm.cli import execute
from sdm.config import load_config
from sdm.data import load_dataset
from sdm.evaluate import median_spacing,nearest_distances
from sdm.runs import resolve_run,git_state
from sdm.simplex import check_weights,project
ROOT=Path(__file__).resolve().parents[2]
HISTORY=ROOT/'analysis/bayesian_search/delivery'
OUT=HISTORY/'current'
SEARCHES=[('gp-ard','ei'),('gp-ard','mean'),('gp-ard','pessimistic'),('gp-tree-blend','mean'),('gp-tree-blend','pessimistic')]
_records=json.loads((ROOT/'analysis/bayesian_search/team_records.json').read_text())
REAL_RESULTS={key:float(_records[key]['R_decimal']) for key in ['submission_1','baseline_weights']}
def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def validate_search_config(path,name,acquisition):
 config=load_config(path)
 training=load_config(ROOT/f'experiments/bayes-study-{name}.toml')
 if config.model!=training.model:raise ValueError('Search model differs from committed training recipe')
 if config.optimizer is None or config.optimizer.name!='bayes-pool':raise ValueError('Expected bayes-pool')
 actual=config.optimizer.params.get('acquisition','pessimistic')
 if actual!=acquisition:raise ValueError(f'Acquisition mismatch: expected {acquisition}, got {actual}')
 if acquisition=='ei' and name!='gp-ard':raise ValueError('EI is reserved for the raw ARD GP posterior')
 return config
def line_pool(anchor,targets,radii=np.arange(.02,.151,.01)):
 rows=[]
 for label,target in targets.items():
  direction=target-anchor;norm=np.linalg.norm(direction)
  if norm<=1e-12:raise ValueError('Line target equals anchor')
  for radius in radii:
   raw=anchor+radius*direction/norm;point=project(raw)
   rows.append((f'line-{label}-d{radius:.2f}',point,float(radius),float(np.linalg.norm(point-anchor)),bool(np.min(raw)<0)))
 return rows
def eligible_by_veto(gain,threshold):return np.asarray(gain)<=threshold
def main():
 os.chdir(ROOT);OUT.mkdir(exist_ok=True)
 for name,mode in SEARCHES:validate_search_config(ROOT/f'experiments/bayes-search-{name}-{mode}.toml',name,mode)
 state=git_state()
 if state['dirty'] is not False:raise RuntimeError('Evidence requires available Git and committed source/configs')
 write(OUT/'plan.json',dict(git=state,searches=SEARCHES,seed=2026,decision=23,actual_results=REAL_RESULTS,offline_bar=115.76,simulator_calls=0,eligibility='ExtraTrees gain <= min(4.9, its submission1 gain)',ranking='ARD mean gain + 0.5 posterior SD',historical_r06_recommendation_withdrawn=True,official_submission2_selected=False))
 trained={}
 for name in ['gp-ard','gp-tree-blend']:
  run=execute('train',ROOT/f'experiments/bayes-study-{name}.toml');trained[name]=run.id;print('TRAIN',name,run.id,flush=True)
 veto_run=execute('train',ROOT/'experiments/extratrees-v32.toml');write(OUT/'training_runs.json',dict(trained,extratrees=veto_run.id))
 points=[];labels=[];sources=[]
 for name,mode in SEARCHES:
  path=ROOT/f'experiments/bayes-search-{name}-{mode}.toml';validate_search_config(path,name,mode)
  run=execute('optimize',path,n=10,model_run=trained[name]);batch=check_weights(run.read_json('weights.json'))
  for rank,point in enumerate(batch,1):points.append(point);labels.append(f'{name}-{mode}-r{rank:02d}')
  sources.append(dict(model=name,acquisition=mode,run_id=run.id))
 data=load_dataset(ROOT/'dataset_same8');best=int(np.argmin(data.y));anchor=data.X[best]
 old=json.loads((HISTORY/'all_candidate_weights.json').read_text())
 targets={name:np.array(old[name]).ravel() for name in ['gp-tree-blend-ei-r10','gp-tree-blend-ei-r05','gp-tree-blend-ei-r06']}
 rays=line_pool(anchor,targets)
 for label,point,_,_,_ in rays:labels.append(label);points.append(point)
 candidate_count=len(points)
 controls={'best-observed':anchor,'course-baseline':check_weights(json.loads((ROOT/'weights.json').read_text()))[0],'submission-1':check_weights(json.loads((HISTORY/'submitted_weights.json').read_text()))[0]}
 for label,point in controls.items():labels.append(label);points.append(point)
 points=check_weights(points);gp=resolve_run(trained['gp-ard']).load_model();veto=resolve_run(veto_run.id).load_model()
 frame=pd.DataFrame({'candidate':labels})
 for name,model in [('gp-ard',gp),('extratrees',veto)]:
  mean,std=model.predict(points);frame[name+'_mean']=mean;frame[name+'_spread']=std;frame[name+'_gain']=mean-mean[labels.index('best-observed')]
 threshold=min(4.9,float(frame.loc[frame.candidate=='submission-1','extratrees_gain'].iloc[0]))
 frame['veto_threshold']=threshold;frame['veto_eligible']=eligible_by_veto(frame.extratrees_gain,threshold)
 frame['ranking_objective']=frame['gp-ard_gain']+.5*frame['gp-ard_spread']
 frame['distance_ratio']=nearest_distances(points,data.X)/median_spacing(data.X);frame['distance_from_best_observed']=np.linalg.norm(points-anchor,axis=1);frame['measured_R']=np.nan
 for label,value in [('best-observed',float(data.y[best])),('submission-1',REAL_RESULTS['submission_1']),('course-baseline',REAL_RESULTS['baseline_weights'])]:frame.loc[frame.candidate==label,'measured_R']=value
 frame.to_csv(OUT/'candidate_comparison.csv',index=False);write(OUT/'all_candidate_weights.json',dict(zip(labels,points.reshape(-1,3,7).tolist())))
 pd.DataFrame([dict(candidate=a,nominal_radius=c,actual_radius=d,projected=e) for a,b,c,d,e in rays]).to_csv(OUT/'line_geometry.csv',index=False)
 eligible=frame.iloc[:candidate_count].query('veto_eligible and distance_ratio > 0.000001')
 lead=None if eligible.empty else str(eligible.sort_values(['ranking_objective','candidate']).iloc[0].candidate)
 write(OUT/'decision.json',dict(research_lead=lead,official_submission2_selected=False,simulator_calls=0,independent_veto='ExtraTrees, not used in generation or ranking',veto_threshold=threshold,actual_R=None,known_results=REAL_RESULTS,limitation='Veto informed by only two real points; no real improvement guarantee'))
 if lead is not None:write(OUT/'RESEARCH_LEAD_NOT_OFFICIAL_SUBMISSION.json',points[labels.index(lead)].reshape(3,7).tolist())
 write(OUT/'search_runs.json',sources);print('RESEARCH LEAD',lead,'official submission2 remains unselected',flush=True)
if __name__=='__main__':main()
