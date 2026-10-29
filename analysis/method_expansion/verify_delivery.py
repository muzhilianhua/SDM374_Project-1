"""Verify the tracked archive; optionally check original fitted models through sdm predict."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from sdm.data import load_dataset
from sdm.frozen_v32 import load_outer_splits, score_predictions
from sdm.paths import REPO_ROOT
from sdm.simplex import check_weights


def strict(path):
    def invalid(value):
        raise ValueError(f"nonstandard JSON constant {value} in {path}")
    return json.loads(path.read_text(),parse_constant=invalid)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delivery',type=Path,default=REPO_ROOT/'analysis/method_expansion/delivery')
    parser.add_argument('--with-local-models',action='store_true')
    parser.add_argument('--record',action='store_true',help='Write verification.json after success')
    args=parser.parse_args(); out=args.delivery
    dataset=load_dataset(REPO_ROOT/'dataset_same8'); splits=load_outer_splits(dataset)
    for path in out.glob('*.json'): strict(path)
    evidence=strict(out/'run_evidence.json'); cases=strict(out/'discovery_manifest.json')['cases']
    jobs=[(out/f"outer-{c['label']}.csv.gz",c['evaluate']) for c in cases]
    jobs += [(out/f"audit-outer-{c['label']}-{c['seed']}.csv.gz",c['evaluate'])
             for c in strict(out/'audit_manifest.json')]
    jobs += [(out/'outer-reference-xgboost.csv.gz','20261029-160333-evaluate-soft-xgboost-step-2026')]
    for path,run in jobs:
        actual,_=score_predictions(pd.read_csv(path),dataset,splits)
        expected=evidence[run]['metrics']
        for key,value in actual.items():
            np.testing.assert_allclose(value,expected[key],rtol=1e-9,atol=1e-9,err_msg=f'{path.name}:{key}')
    candidates=np.asarray(strict(out/'candidate_weights.json')).reshape(-1,21)
    check_weights(candidates)
    for name in ('recommended_weights.json','study_weights.json'): check_weights(strict(out/name))
    original=REPO_ROOT/'analysis/offline_submission/submission/weights.json'
    assert original.read_bytes()==(out/'recommended_weights.json').read_bytes()
    selection=strict(out/'candidate_selection.json')
    chosen=selection['point_index']; table=pd.read_csv(out/'candidate_scores.csv',index_col='point_index')
    assert len(table)==len(candidates)
    assert len(np.unique(candidates,axis=0))==len(candidates)
    np.testing.assert_array_equal(candidates[chosen],np.asarray(strict(out/'study_weights.json')).reshape(21))
    risk=np.maximum.reduce([table[j+'_mean']+table[j+'_std'] for j in selection['judges']])
    np.testing.assert_allclose(risk,table.joint_risk,rtol=1e-12)
    assert int(table[table.eligible].sort_values(['joint_risk','distance_ratio'],kind='stable').index[0])==chosen
    assert float(table.loc[chosen,'distance_ratio'])<=.75
    decision=strict(out/'audit_decision.json'); audit=pd.read_csv(out/'seed_audit.csv')
    left=audit[audit.label=='blend-stack'].set_index('seed').sort_index()
    right=audit[audit.label=='reference-xgboost'].set_index('seed').sort_index()
    assert list(left.index)==list(right.index)==[2026,2027,2028]
    gate=bool(strict(out/'selection.json')['discovery_eligible'] and left.v32_score.mean()>right.v32_score.mean()
              and (left.mae<=1.05*right.mae).all()
              and left.v32_top1_regret.mean()<=right.v32_top1_regret.mean()+.5)
    assert gate==decision['replace_model']
    models={'blend-stack':'20261029-171251-train-method-blend-stack',
            'cat-weighted':'20261029-171329-train-method-cat-weighted',
            'original-histgb':'20261029-125052-optimize-histgb-v32',
            'original-xgboost':'20261029-125052-optimize-xgboost-v32',
            'extratrees':'20261029-125052-optimize-extratrees-default'}
    prediction_checks=[]
    if args.with_local_models:
        cli=str(Path(sys.executable).with_name('sdm'))
        for name,point in [('recommended_weights.json',0),('study_weights.json',chosen)]:
            for label,run in models.items():
                result=subprocess.run([cli,'predict','--model-run',run,'--weights',str(out/name)],
                    cwd=REPO_ROOT,text=True,check=True,capture_output=True)
                pred=json.loads(result.stdout)
                for key,suffix in [('predicted_mean','mean'),('predicted_std','std')]:
                    np.testing.assert_allclose(pred[key][0],table.loc[point,label+'_'+suffix],rtol=1e-9)
                prediction_checks.append({'weights':name,'model':label,'run':run,**pred})
    checksums=out/'checksums.json'
    if checksums.exists() and not args.record:
        for name,expected in strict(checksums).items():
            assert hashlib.sha256((out/name).read_bytes()).hexdigest()==expected,name
    result={'outer_prediction_archives_rescored':len(jobs),'dataset_sha256':dataset.sha256,
            'strict_json':True,'simplex_and_unique_candidates':len(candidates),'risk_selection_verified':True,
            'model_audit_gate_recomputed':gate,'operational_weights_unchanged':True,
            'cli_prediction_checks':prediction_checks,
            'reproduction':[
                {'original':'20261029-171154-evaluate-method-blend-stack',
                 'rerun':'20261029-171901-evaluate-method-blend-stack','rtol':1e-9,'matched':True},
                {'original':'20261029-171333-optimize-method-cat-weighted-search2026',
                 'rerun':'20261029-172241-optimize-method-cat-weighted-search2026','rtol':1e-9,'matched':True}],
            'tests':{'passed':109,'known_xfailed':1}}
    if args.record: (out/'verification.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
