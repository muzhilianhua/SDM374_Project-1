"""Package completed evidence, an owned fitted GP and feasible unmeasured candidate payloads."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
import pandas as pd
from sdm.data import load_dataset
from sdm.simplex import check_weights

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'analysis/bayesian_search/delivery'


def write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def main():
    initial=json.loads((OUT/'evaluation.json').read_text())
    extension=json.loads((OUT/'extension/evaluation.json').read_text())
    records=initial+extension
    pd.DataFrame(records).sort_values('v32_score',ascending=False).to_csv(OUT/'model_comparison.csv',index=False)
    byname={r['model']:r for r in records}
    reference=byname['method-blend-stack'];gp=byname['gp-ard'];blend=byname['gp-tree-blend']
    seeds=json.loads((OUT/'seed_audit/evaluation.json').read_text())
    seeds += [dict(reference,seed=2026),dict(blend,seed=2026)]
    pd.DataFrame(seeds).sort_values(['seed','model']).to_csv(OUT/'seed_audit/comparison.csv',index=False)
    means=pd.DataFrame(seeds).groupby('model').mean(numeric_only=True)
    old=means.loc['method-blend-stack'];new=means.loc['gp-tree-blend']
    ratios=[]
    for seed in [2026,2027,2028]:
        oldrow=next(r for r in seeds if r['seed']==seed and r['model']=='method-blend-stack')
        newrow=next(r for r in seeds if r['seed']==seed and r['model']=='gp-tree-blend')
        ratios.append(newrow['mae']<=1.05*oldrow['mae'] and newrow['v32_local_mae']<=1.05*oldrow['v32_local_mae'])
    gate=bool(new.v32_score>old.v32_score and new.v32_top1_regret<=old.v32_top1_regret and all(ratios))
    write(OUT/'seed_audit/decision.json',dict(gate_passed=gate,
        old_mean_score=old.v32_score,new_mean_score=new.v32_score,
        old_mean_regret=old.v32_top1_regret,new_mean_regret=new.v32_top1_regret,
        decision='Exploratory models and candidates only; no automatic controller replacement',
        independent_confirmation=False,simulator_calls=0))
    # Refresh historical diagnostics for late-arriving ARD experiments, without refitting.
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
                chosen=subset.sort_values(['predicted_mean','candidate_id'],kind='stable').iloc[0]
                pools.append(dict(model=record['model'],split=split,pool=pool,selected_actual_R=chosen.actual,
                    regret=chosen.actual-subset.actual.min(),optimism=chosen.actual-chosen.predicted_mean))
    historical=pd.DataFrame(historical);historical.to_csv(OUT/'historical_fold_selection.csv',index=False)
    historical.groupby('model').mean(numeric_only=True).to_csv(OUT/'historical_selection_summary.csv')
    pools=pd.DataFrame(pools);pools.to_csv(OUT/'historical_pools.csv',index=False)
    pools.groupby('model').mean(numeric_only=True).to_csv(OUT/'historical_pool_summary.csv')
    batch=check_weights(json.loads((OUT/'shortlist/weights.json').read_text()))
    assert batch.shape==(5,21) and len(np.unique(batch,axis=0))==5
    decision=json.loads((OUT/'candidate_decision.json').read_text())
    allweights=json.loads((OUT/'all_candidate_weights.json').read_text())
    primary=decision['primary']
    write(OUT/'NEXT_TEST_WEIGHTS.json',allweights[primary])
    panel=pd.read_csv(OUT/'candidate_comparison.csv').set_index('candidate')
    first=panel.loc[primary]
    write(OUT/'validation.json',dict(shape=[5,3,7],all_feasible=True,distinct=5,
        max_row_sum_error=float(np.max(abs(batch.reshape(5,3,7).sum(axis=2)-1))),
        tests='129 passed, 1 skipped (GUI dependency unavailable), 1 known xfailed',
        reproduce_search='passed at rtol=1e-9',reproduce_gp_training='passed at rtol=1e-9',
        simulator_calls=0,official_submissions=0))
    # Only locally produced trusted run files are copied; do not inspect external pickles.
    trained=json.loads((OUT/'training_runs.json').read_text())
    modeldir=ROOT/'artifacts/bayesian_gp';modeldir.mkdir(exist_ok=True)
    for name in ['config.toml','meta.json','metrics.json','model.pkl']:
        shutil.copy2(ROOT/'runs'/trained['gp-ard']/name,modeldir/name)
    write(modeldir/'CHECKSUMS.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in modeldir.iterdir() if p.is_file() and p.name!='CHECKSUMS.json'})
    head=(ROOT/'.git/HEAD').read_text().strip()
    commit=(ROOT/'.git'/head.removeprefix('ref: ')).read_text().strip()
    write(OUT/'release_provenance.json',dict(base_azq='651c310905348859fba3aa4f6caed052cac222e1',
        branch=head,source_head_at_packaging=commit,
        discovery_source_commit='5b76ab22faffeddae142a8cd1747ddd24b6feb50',
        extension_source_commit='57573fc97f5ff65ad316642073e4f1c8664d9a58',
        candidate_code_commit='990ad03c0caafb6dc98cb0f61378e76e03966c74',
        dataset_sha256=load_dataset(ROOT/'dataset_same8').sha256,
        packages={n:version(n) for n in ['numpy','pandas','scipy','scikit-learn','xgboost','catboost']},
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*list((ROOT/'src/sdm/models').glob('*.py')),ROOT/'src/sdm/optimizers/bayes_pool.py',
                      *list((ROOT/'analysis/bayesian_search').glob('*.py'))]},
        note='Git cloned and locally committed via Dulwich. System git unavailable, so canonical run git fields are null. Supplemental commits and hashes above provide provenance. Generated audit configs were uncommitted when run.',
        simulator_calls=0))
    rows=[]
    for name in ['baseline-v32','method-blend-stack','gp-additive','gp-ard','gp-ard-log','gp-tree-blend']:
        r=byname[name];rows.append(f"| {name} | {r['v32_score']:.2f} | {r['mae']:.2f} | {r['v32_local_mae']:.2f} | {r['v32_top1_regret']:.2f} |")
    table='\n'.join(rows)
    readme=f'''# Bayesian model exploration and unmeasured candidate handoff

Seventeen new model recipes plus three fresh references were evaluated with the unchanged frozen-v3.2 protocol (ten folds each). Four additional matched model-seed evaluations were run. No simulator calls or official submissions were made.

| Model | v3.2 (higher better) | MAE | Low-region MAE | Top1 regret |
|---|---:|---:|---:|---:|
{table}

All lower-is-better columns measure historical held-out labels. Discovery folds are reused; they are not independent confirmation. The strongest GP is an additive Matern kernel with shared seven-objective ARD, a 10% district interaction term and training-only empirical-Bayes marginal likelihood selection. Optimization uses at most18 L-BFGS iterations. Some length/noise parameters reach declared bounds. Posterior standard deviations assume the kernel/noise model and are not validated confidence bounds for optimized points.

The user-reported first submission is the old challenger, rounded to eight decimals. Reported R is approximately119 versus a baseline approximately118; exact measurements and evaluator access are unavailable. It was not added as a training label. New additive GP predicts119.7093; ARD GP121.4889; freshly trained old stack111.6715. This is one post hoc external diagnostic, not proof of calibration.

The fixed equal simple-GP/tree blend improves three-seed average v3.2 from {old.v32_score:.4f} to {new.v32_score:.4f}, but mean top1 regret increases from {old.v32_top1_regret:.4f} to {new.v32_top1_regret:.4f}. The declared replacement gate passes: **{gate}**. No operational controller is automatically replaced.

`NEXT_TEST_WEIGHTS.json` is the first novel point suggested for real measurement, source `{primary}`. It is search rank6, selected from sixty proposals by the predeclared worst panel mean delta versus p01969 plus separation. It is not an official submission and actual R is unknown. Means: additive GP {first['gp-additive_mean']:.4f}, ARD GP {first['gp-ard_mean']:.4f}, tree stack {first['tree-stack_mean']:.4f}, equal blend {first['gp-tree-blend_mean']:.4f}. Nearest-data distance ratio is {first['distance_ratio']:.4f}. Panel members are dependent; agreement does not certify real improvement.

`shortlist/weights.json` contains five3x7 points. `candidate_comparison.csv` includes predictions, error proxies, distances and explicitly blank actual labels. Three overlapping90% subset refits and128 directions at raw perturbation radii.01/.03 are archived. The primary beats p01969 in both additive GP and tree central predictions in all three subsets. Tree predictions still shift by up to3.33 around radius.03; local stability is not guaranteed. Other four points are exploratory backups, with weaker subset agreement. Do not replace the primary based on this audit.

`best_observed_115.7648.json` is unchanged historical p01969. Its dataset label is115.7648192592. Check whether observed rows are eligible and whether the current evaluator uses identical conditions before treating it as a fallback.

Model, optimizer and test plugins are included. These model-path changes cross role B ownership because the user explicitly requested new models. No framework interface, frozen split, scorer, course baseline or training table was changed. Git provenance is supplemental because this computer lacks a system git executable; see release_provenance.json. The working branch is `feat/c-bayesian-study`, based on azq651c310. There was no remote push.

The ZIP is an incremental handoff, not a standalone repository. Apply its relative paths over the azq651c310 checkout, or use the complete local branch already prepared beside the archive. It includes the fitted ARD GP, changed plugins/configs and all canonical evaluation evidence. Keep the unchanged course data and framework from the base repository.

## Replay from this branch

```sh
uv run --no-editable sdm evaluate -c experiments/bayes-study-gp-ard.toml
uv run --no-editable sdm predict --model-run artifacts/bayesian_gp --weights analysis/bayesian_search/delivery/NEXT_TEST_WEIGHTS.json
uv run --no-editable sdm optimize -c experiments/bayes-search-gp-ard-ei.toml -n 10 --model-run artifacts/bayesian_gp
uv run --no-editable pytest -q
```

Python3.11 and the unchanged uv.lock are required. `--no-editable` avoids a Python3.11 Windows locale issue with UTF-8 editable path files in Chinese directories. On this machine uv is also callable with `python -m uv`. Full-fit GP training and the primary source search reproduced within rtol1e-9. The fitted GP is in artifacts/bayesian_gp; only load this owned, trusted pickle. Same-platform reproducibility passed; cross-platform reproducibility has not been tested.

The EI pool is an offline next-batch acquisition step. A genuine online Bayesian optimization loop requires measured feedback, which is not available here. The approximate119 report was never used as an exact observation. Before another official submission, request validation of the primary and, if allowed, p01969 under the same evaluator.
'''
    (OUT/'README.md').write_text(readme,encoding='utf-8')
    chinese=f'''已完成17套新模型探索，并与3个本机重跑的参照统一比较。找到了离线预测明显更好的新模型，也筛出了5组待测权重；尚无新的真实测评结果，不能认定已经超过真实baseline118。

| 模型 | 离线v3.2得分↑ | 整体误差↓ | 低波动区域误差↓ | 首选点损失↓ |
|---|---:|---:|---:|---:|
{table}

以上都使用原冻结协议的完整10折，训练约80%、测试约20%。首选点损失指模型在每个测试折选中的点，比该折实际最佳点高多少。83.15是预测模型的离线评分，越高越好；119、118是实际波动量R，越低越好，不能放在同一尺度比较。

新模型把三栋楼各自的影响用三个高斯过程核相加，再加入10%的楼间耦合。七项指标各自有一个可学习的长度尺度，由允许使用的训练数据最大化边际似然；这属于经验贝叶斯。整体误差较旧集成下降约{100*(1-gp['mae']/reference['mae']):.1f}%，低波动区域误差下降约{100*(1-gp['v32_local_mae']/reference['v32_local_mae']):.1f}%。普通高斯过程、贝叶斯二次回归、ARD二次回归和SVR没有带来同等改善，负结果已保存。

你提供的“第七组离线提交-1.json”对应旧challenger。新加性GP预测119.71，ARD-GP预测121.49，本机旧集成仍预测111.67。你说的实测约119没有加入训练；这一点支持旧模型存在过度乐观的问题，但只凭一个实测点不足以证明新模型全面准确。

另一个固定等权GP/树混合模型的三种子平均得分从{old.v32_score:.2f}提高到{new.v32_score:.2f}，但首选点损失从{old.v32_top1_regret:.2f}变为{new.v32_top1_regret:.2f}，未通过预先写下的替换标准。ARD-GP虽然83.15分，首选点损失2.00也略差于旧集成2026种子的1.82。因此这些方案都保留为探索结果，没有自动替换原控制方案。

建议下一次真实验证先用 NEXT_TEST_WEIGHTS.json，来源{primary}。对这组参数，加性GP预测{first['gp-additive_mean']:.2f}、ARD-GP预测{first['gp-ard_mean']:.2f}、旧树集成预测{first['tree-stack_mean']:.2f}、等权混合预测{first['gp-tree-blend_mean']:.2f}。这几个模型有关联，不是独立测评器。该点与最近训练样本的距离是典型样本间距的{first['distance_ratio']:.2f}倍；三次90%子集重训中，GP和树模型都预测它优于训练表最佳点。小扰动检查仍发现树预测可波动3.33，不能保证真实稳定性。其余4组也已保存，有些在子集检查中失去一致性，作为备选探索点即可。

训练表p01969的历史真实标签为115.7648，文件 best_observed_115.7648.json。若允许原样提交已有样本，而且与当前测评使用相同条件，它值得作为参照；本次没有重新测量它，不能把历史标签当成这次实测。

你接下来只需把首选权重交给能调用测评器的队友验证，保留精确R、所用文件和评估条件。若能验证多个点，再测其余备选和p01969。当前EI只是基于已有数据选择下一批点；收到真实反馈后才能继续“测评—更新模型—再选点”的贝叶斯优化闭环。没有代替你消耗正式提交机会。

代码在 SDM374_Project-1-bayesian-study，真实Git分支为feat/c-bayesian-study，基于最新azq651c310。所有实现、配置、冻结评估记录、5组权重、稳定性检查及已训练GP已打包。ZIP是增量交接包，需合并到azq651c310对应的项目根目录使用；本机旁边已准备好完整工作副本。完整测试129通过、1跳过（GUI依赖未安装）、1项已有预期失败；GP训练与候选搜索复现通过1e-9容差，复训前后对64个候选及参照点的预测完全一致。未测试跨服务器复现。原始数据、baseline和冻结评分器均未改动；没有远端推送或真实测评调用。
'''
    report=ROOT.parent/'贝叶斯模型探索_结果与测评建议.md'
    report.write_text(chinese,encoding='utf-8')
    # A compact visual of the two different questions: prediction quality and selection regret.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['baseline-v32','method-blend-stack','gp-additive','gp-ard','gp-tree-blend']
    labels=['Ridge baseline','Old tree stack','Additive GP','ARD additive GP','GP/tree 50:50']
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    axes[0].barh(labels,[byname[n]['v32_score'] for n in names],color=['#aaa','#999','#80b1d3','#147d92','#66aa77'])
    axes[0].set_xlim(0,100);axes[0].set_title('Frozen v3.2 score (higher is better)')
    axes[1].barh(labels,[byname[n]['v32_top1_regret'] for n in names],color=['#aaa','#999','#80b1d3','#147d92','#66aa77'])
    axes[1].set_title('Held-out top1 regret (lower is better)')
    fig.suptitle('Same 10 frozen folds, model seed 2026; no new simulator measurements')
    fig.tight_layout();fig.savefig(OUT/'model_comparison.png',dpi=160);plt.close(fig)
    # Include all canonical evaluation evidence, but only the best fitted model binary.
    archive=ROOT.parent/'贝叶斯探索_代码结果和5组待测权重.zip'
    files=list(OUT.rglob('*'))+list(modeldir.rglob('*'))
    files += [ROOT/'src/sdm/models/bayes_search.py',ROOT/'src/sdm/models/gp_tree_blend.py',
              ROOT/'src/sdm/optimizers/bayes_pool.py',ROOT/'tests/test_bayes_search.py']
    files += list((ROOT/'analysis/bayesian_search').glob('*.py'))
    files += list((ROOT/'experiments').glob('bayes-*.toml'))
    files += [ROOT/'experiments/method-blend-stack.toml',ROOT/'pyproject.toml',ROOT/'uv.lock',ROOT/'docs/roles/C.md']
    for record in records+seeds:
        files += [p for p in (ROOT/'runs'/record['run_id']).iterdir() if p.suffix in ('.json','.csv','.toml')]
    for name in ['tests.log','reproduce.log','reproduce_gp.log']:
        files.append(ROOT/'analysis/bayesian_search'/name)
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as package:
        for path in sorted(set(files)):
            if path.is_file():package.write(path,str(path.relative_to(ROOT)))
        package.write(report,report.name)
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix('.sha256.txt').write_text(digest+'  '+archive.name+'\n',encoding='utf-8')
    print('report',report);print('archive',archive,'bytes',archive.stat().st_size)
    print('gate',gate,'scores',old.v32_score,new.v32_score,'regrets',old.v32_top1_regret,new.v32_top1_regret)


if __name__=='__main__':main()
