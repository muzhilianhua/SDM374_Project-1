"""Rescore exactly five peer groups plus six previous finalists; never fit or tune models."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import argparse
import csv
from datetime import datetime,timezone,timedelta
import hashlib
import importlib.util
import io
import json
import platform
import subprocess
import sys
import zipfile
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'local_research/scoring_v3_2'))
from surrogate_scoring_v3_2 import evaluate,make_context,load_final_policy,WEIGHTS

FROZEN=ROOT/'outputs/frozen_data_splits_20261025/SDM374_data_splits_frozen_v1'
OLD=ROOT/'outputs/benchmark_v3_2_20261025'
PEER=ROOT/'incoming/SDM374_trained_models_v1.zip'
EXPECTED_ZIP='a0aa8cefce4b3739a6eea77c444fd04c6e164d70c286d05968b67c189b70ed46'
EXPECTED_SPLIT='91fbd29f05d64ff2470e8b8a15ade78b7f234b234bf32f03a67fb4ca5d15a09d'
PEERS=[
 ('peer_low90_xgb','组员 low90_w20x3 + XGBoost','nestedcv_low90_w20x3_v1','xgboost',74.72),
 ('peer_low80_histgb','组员 low80 + HistGB','nestedcv_xgb_gbdt_histgb_low80_v1','histgb',74.50),
 ('peer_low60_histgb','组员 low60 + HistGB','nestedcv_xgb_gbdt_histgb_low60_v1','histgb',74.21),
 ('peer_low90_histgb','组员 low90_w20x3 + HistGB','nestedcv_low90_w20x3_v1','histgb',73.73),
 ('peer_tiered_histgb','组员 tiered + HistGB','nestedcv_tiered_w20x5_w20to40x2_v1','histgb',73.26)]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def write_csv(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True,encoding='utf-8').strip()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--peer-python',type=Path,required=True)
    args=parser.parse_args();peer_python=args.peer_python.resolve();assert peer_python.exists()
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d-%H%M%S')
    run_id=stamp+'-artifact-review-v32';out=Path(__file__).parent/'results'/run_id;out.mkdir(parents=True)
    assert digest(PEER)==EXPECTED_ZIP and digest(FROZEN.with_suffix('.zip'))==EXPECTED_SPLIT
    spec=importlib.util.spec_from_file_location('artifact_frozen',FROZEN/'split_api.py')
    api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
    table,manifest=api.load_package(FROZEN);splits=list(api.iter_nested_splits(manifest,table['candidate_ids']))
    ids=np.array(table['candidate_ids']);y=np.array([r['ramping'] for r in table['records']])
    X=np.array([[r[c] for c in api.WEIGHT_COLUMNS] for r in table['records']]);assert X.shape==(2048,21)
    arrays={}
    for split in splits:
        sid=split['split_id'];te=np.array(split['outer_test'])
        arrays.update({sid+'_X':X[te],sid+'_actual':y[te],sid+'_ids':ids[te]})
    np.savez_compressed(out/'fold_input.npz',**arrays)
    artifacts=[];extract=out/'extracted_peer';extract.mkdir()
    prefix='SDM374_trained_models_v1/'
    with zipfile.ZipFile(PEER) as z:
        package=json.loads(z.read(prefix+'MANIFEST.json').decode('utf-8-sig'))
        assert package['protocol_id']==manifest['protocol_id'] and package['dataset_sha256']==manifest['dataset']['sha256']
        assert package['feature_columns']==api.WEIGHT_COLUMNS
        index=list(csv.DictReader(io.StringIO(z.read(prefix+'model_index.csv').decode('utf-8-sig'))))
        for group,label,experiment,family,_ in PEERS:
            selected=[r for r in index if r['experiment']==experiment and r['model']==family]
            assert len(selected)==10 and {r['split_id'] for r in selected}=={s['split_id'] for s in splits}
            for r in selected:
                rel=r['model_file'];path=(extract/rel).resolve();assert path.is_relative_to(extract.resolve())
                payload=z.read(prefix+rel);assert hashlib.sha256(payload).hexdigest()==r['model_sha256']
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
                artifacts.append(dict(group=group,label=label,split_id=r['split_id'],path=str(path),sha256=r['model_sha256'],
                    recorded_mae=float(r['mae']),recorded_rmse=float(r['rmse'])))
    save(out/'inference_plan.json',dict(output=str(out),fold_input=str(out/'fold_input.npz'),artifacts=artifacts))
    source_files=[Path(__file__),Path(__file__).with_name('infer_peer.py'),ROOT/'local_research/scoring_v3_2/surrogate_scoring_v3_2.py',
                  ROOT/'local_research/scoring_v3_2/EVALUATION_POLICY_V3_2.json',ROOT/'AGENTS.md']
    provenance=dict(run_id=run_id,kind='saved-artifact-v3.2-review',commit=git('rev-parse','HEAD'),branch=git('branch','--show-current'),
        git_dirty=bool(git('status','--porcelain')),tracked_diff=git('diff','--name-only'),dataset_sha256=digest(FROZEN/'data/train.csv'),
        split_zip_sha256=EXPECTED_SPLIT,peer_zip_sha256=EXPECTED_ZIP,
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in source_files},
        coordinator=dict(python=platform.python_version(),numpy=np.__version__),
        scope='5 specified peer groups + 6 original finalists; saved-artifact inference/rescoring only',
        fitting_performed=False,hyperparameter_selection_performed=False,optimization_performed=False,
        peer_inner_selection='RMSE, as declared in original manifests',original_inner_selection='v3.2',
        test_sets='complete corresponding frozen outer_test, never filtered or ten-model averaged',
        peer_training_membership='declared by provider, not independently proven from explicit fit-ID lists')
    save(out/'provenance.json',provenance)
    print('run:',out,flush=True)
    subprocess.run([str(peer_python),str(Path(__file__).with_name('infer_peer.py')),'--plan',str(out/'inference_plan.json')],check=True)
    policy=load_final_policy();folds=[];summary=[];observations=[];clouds={}
    old_summary=json.loads((OLD/'summary.json').read_text(encoding='utf-8'))['summary']
    groups=[dict(id=g,label=l,source='peer',original_name=None,screenshot_score=s) for g,l,e,m,s in PEERS]
    groups += [dict(id='original_'+r['model'],label='原决赛圈 '+r['label'],source='original',original_name=r['model'],screenshot_score=None) for r in old_summary]
    for group in groups:
        group_rows=[];seen=[];actual=[];predicted=[]
        for split in splits:
            sid=split['split_id'];tr=np.array(split['outer_train']);te=np.array(split['outer_test'])
            if group['source']=='peer':file=out/'peer_predictions'/f"{group['id']}_{sid}.npz"
            else:file=OLD/'predictions'/f"{group['original_name']}_{sid}.npz"
            with np.load(file,allow_pickle=False) as data:
                np.testing.assert_array_equal(data['candidate_ids'],ids[te]);np.testing.assert_array_equal(data['actual'],y[te])
                p=data['predicted'].astype(float)
                if group['source']=='original':np.testing.assert_array_equal(data['train_ids'],ids[tr])
            result=evaluate(y[te],p,make_context(y[tr],policy),policy,ids[te])
            if group['source']=='original':
                old=json.loads((OLD/'records'/f"{group['original_name']}_{sid}.json").read_text(encoding='utf-8'))
                assert result==old['outer']
            row=dict(group=group['id'],label=group['label'],source=group['source'],split_id=sid,
                **{f'p_{k}':v for k,v in result['points'].items()},score=result['score'],
                local_mae=result['local_mae'],global_mae=result['global_mae'],local_bias=result['local_bias'],global_bias=result['global_bias'],
                local_order_accuracy=result['local_pair_accuracy'],global_order_accuracy=result['global_pair_accuracy'],
                top1_regret=result['top1_regret'],top1_actual=result['top1_actual'],top1_predicted=result['top1_predicted'],
                oracle_best_actual=result['oracle_best_actual'],top1_id=result['top1_candidate_id'],
                recall=result['recall_at_low_count'],test_n=len(te),low_n=result['low_n'],train_std=result['train_std'])
            group_rows.append(row);folds.append(row);seen.extend(ids[te]);actual.extend(y[te]);predicted.extend(p)
            observations.extend(dict(group=group['id'],split_id=sid,candidate_id=str(cid),actual=float(yy),predicted=float(pp),
                error=float(pp-yy),is_low=bool(yy<=132.8)) for cid,yy,pp in zip(ids[te],y[te],p))
        unique,counts=np.unique(seen,return_counts=True);np.testing.assert_array_equal(unique,np.sort(ids));assert np.all(counts==2)
        scores=np.array([r['score'] for r in group_rows]);gaps=np.array([r['top1_regret'] for r in group_rows])
        numeric=[k for k in group_rows[0] if k.startswith('p_')]+['local_mae','global_mae','local_bias','global_bias',
            'local_order_accuracy','global_order_accuracy','top1_regret','top1_actual','oracle_best_actual','recall']
        row=dict(group=group['id'],label=group['label'],source=group['source'],score=float(scores.mean()),
            **{k:float(np.mean([r[k] for r in group_rows])) for k in numeric},
            score_std=float(scores.std(ddof=0)),score_min=float(scores.min()),score_max=float(scores.max()),
            top1_regret_std=float(gaps.std(ddof=0)),top1_regret_max=float(gaps.max()),
            top1_within_half_count=int((gaps<=.5).sum()),screenshot_score=group['screenshot_score'],
            screenshot_difference=float(scores.mean()-group['screenshot_score']) if group['screenshot_score'] is not None else None)
        assert abs(sum(row['p_'+k] for k in WEIGHTS)-row['score'])<1e-10
        summary.append(row);clouds[group['id']]=(np.array(actual),np.array(predicted))
    summary.sort(key=lambda r:r['score'],reverse=True)
    for rank,r in enumerate(summary,1):r['rank']=rank
    assert len(summary)==11 and len(folds)==110 and len(observations)==45056
    save(out/'summary.json',dict(run_id=run_id,summary=summary,folds=folds))
    write_csv(out/'score_comparison.csv',summary);write_csv(out/'fold_scores.csv',folds);write_csv(out/'heldout_predictions.csv',observations)
    font=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name()
    plt.rcParams.update({'font.family':font,'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(17,6));ax.axis('off')
    headers=['排名','策略与模型','总分','折间SD','低值排序 /35','低值MAE /15','全局排序 /15','全局MAE /15','第一推荐 /20']
    cells=[[r['rank'],r['label'],f"{r['score']:.2f}",f"{r['score_std']:.2f}",*[f"{r['p_'+k]:.2f}" for k in WEIGHTS]] for r in summary]
    t=ax.table(cellText=cells,colLabels=headers,cellLoc='center',colWidths=[.04,.27,.08,.08,.11,.11,.11,.11,.11],bbox=[0,.08,1,.80])
    t.auto_set_font_size(False);t.set_fontsize(10)
    for (i,j),c in t.get_celld().items():
        c.set_edgecolor('#e0e5eb');c.set_linewidth(.4)
        if i==0:c.set_facecolor('#263e5a');c.set_text_props(color='white',weight='bold')
        elif i%2==0:c.set_facecolor('#f1f4f8')
    ax.set_title('11组现成模型逐折复评：固定划分与v3.2，无重新选参或训练',fontsize=17,pad=10)
    fig.text(.5,.035,'low60/80/90指训练策略；各组使用完整相同外层测试集，低值区阈值统一为132.80。',ha='center',fontsize=10)
    fig.savefig(out/'score_comparison.png',dpi=170,bbox_inches='tight');plt.close(fig)
    bounds=(min(min(a.min(),p.min()) for a,p in clouds.values()),max(max(a.max(),p.max()) for a,p in clouds.values()))
    pad=.025*(bounds[1]-bounds[0]);limits=(bounds[0]-pad,bounds[1]+pad)
    fig,axes=plt.subplots(4,3,figsize=(17,20))
    for ax,r in zip(axes.flat,summary):
        a,p=clouds[r['group']];low=a<=132.8
        ax.scatter(a[~low],p[~low],s=6,c='#426caa',alpha=.22,linewidths=0)
        ax.scatter(a[low],p[low],s=12,c='#dc7328',alpha=.65,linewidths=0)
        ax.plot(limits,limits,'--',c='#303030',lw=1);ax.set_xlim(limits);ax.set_ylim(limits)
        ax.set_xlabel('真实 ramping');ax.set_ylabel('预测 ramping');ax.grid(alpha=.15)
        ax.set_title(f"{r['label']}\n总分 {r['score']:.2f}",fontsize=11)
    axes.flat[-1].axis('off');fig.suptitle('11组方案外层留出预测：每组4096个预测，原样保留两轮结果',fontsize=17,y=.995)
    fig.tight_layout(rect=(0,0,1,.98));fig.savefig(out/'all_models_scatter.png',dpi=150);plt.close(fig)
    inference=json.loads((out/'peer_inference.json').read_text(encoding='utf-8'))
    provenance['peer_environment']=inference['versions'];provenance['completed']=True
    provenance['output_hashes']={f:digest(out/f) for f in ['summary.json','score_comparison.csv','fold_scores.csv','heldout_predictions.csv']}
    save(out/'provenance.json',provenance)
    save(out/'validation.json',dict(status='passed',groups=11,outer_folds=110,new_peer_model_predictions=50,
        original_saved_prediction_sets_rechecked=60,peer_recorded_error_checks=50,
        no_fit_calls=True,no_hyperparameter_selection=True,no_test_filtering=True,no_fold_ensemble=True,
        frozen_zip_unchanged=digest(FROZEN.with_suffix('.zip'))==EXPECTED_SPLIT,peer_zip_unchanged=digest(PEER)==EXPECTED_ZIP))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print('COMPLETE',out,flush=True)


if __name__=='__main__':main()
