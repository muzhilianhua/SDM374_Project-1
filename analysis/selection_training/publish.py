"""Archive the fresh experiment, exact fitted bundles, exports and negative results."""
from __future__ import annotations

from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from sdm.data import load_dataset, sha256_of
from sdm.paths import REPO_ROOT
from sdm.runs import git_state, resolve_run
from sdm.simplex import check_weights

from audit import provenance
from run import BASE, RESULTS, write


def table(frame):
    frame = frame.round(5)
    line = lambda values: '| '+' | '.join(map(str,values))+' |'
    return '\n'.join([line(frame.columns),line(['---']*len(frame.columns)),
                      *[line(row) for row in frame.itertuples(index=False,name=None)]])


def publish():
    out = BASE/'delivery'
    out.mkdir(parents=True,exist_ok=True)
    ds = load_dataset(REPO_ROOT/'dataset_same8')
    runs = json.loads((RESULTS/'runs.json').read_text())
    decision = json.loads((RESULTS/'decision.json').read_text())
    for name in ('reference','corrected'):
        run = resolve_run(runs[name+'-fit']['run'])
        dest = out/(name+'_model')
        dest.mkdir(exist_ok=True)
        for filename in ('model.pkl','meta.json','metrics.json','config.toml'):
            shutil.copyfile(run.path/filename,dest/filename)
        (dest/'README.md').write_text(f'# Fresh {name} model\n\nSource CLI run: `{run.id}`.\n'
            'See ../README.md and decision.json before use. No simulator score is available.\n')
    for name in ('precision-old','precision-new','measured','regions','candidates','diversity','direction_probes'):
        shutil.copytree(RESULTS/name,out/name,dirs_exist_ok=True)
    raw_sources = out/'raw_source_files'
    raw_sources.mkdir(exist_ok=True)
    paths = [REPO_ROOT/'analysis/common_judge/delivery/primary_weights.json',
        REPO_ROOT/'analysis/offline_submission/submission/weights.json',
        REPO_ROOT/'analysis/robust_search/delivery/challenger_weights.json',
        REPO_ROOT/'analysis/stable_training/delivery/calibrated_exploratory_weights.json',
        REPO_ROOT/'analysis/stable_training/delivery/exploratory_weights.json',REPO_ROOT/'weights.json']
    paths += [resolve_run(r['run']).path/'weights.json' for r in runs.values()
              if r['meta']['kind']=='optimize']
    source_map = {}
    for index,path in enumerate(paths):
        name = f'{index:02}-'+path.name
        shutil.copyfile(path,raw_sources/name)
        source_map[name] = {'source':str(path.relative_to(REPO_ROOT)),'sha256':sha256_of(path)}
    write(raw_sources/'manifest.json',source_map)
    for filename in ('runs.json','decision.json','reproduce.json','reproduce.log','test_result.json'):
        shutil.copyfile(RESULTS/filename,out/filename)
    rows = []
    for name,record in runs.items():
        if record['meta']['kind'] != 'evaluate':
            continue
        metrics = record['metrics']
        rows.append({'experiment':name,'run':record['run'],'v32_score':metrics['v32_score'],
            'mae':metrics['mae'],'local_mae':metrics['v32_local_mae'],
            'regret':metrics['v32_top1_regret'],'selected_actual_R':record['selected_actual_R'],
            'coverage':metrics['std_coverage'],'seconds':record['seconds']})
        run = resolve_run(record['run'])
        for file in ('predictions','fold_scores'):
            pd.read_csv(run.path/(file+'.csv')).to_csv(out/(name+'-'+file+'.csv.gz'),index=False)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out/'evaluation.csv',index=False)
    new_search = resolve_run(runs['corrected-search-2040']['run'])
    new_raw = check_weights(new_search.read_json('weights.json'))[0].reshape(3,7).tolist()
    write(out/'exploratory_raw_weights.json',new_raw)
    shutil.copyfile(RESULTS/'precision-new/canonical/corrected-search-2040-r1.json',
                    out/'exploratory_weights.json')
    challenger = REPO_ROOT/'analysis/robust_search/delivery/challenger_weights.json'
    shutil.copyfile(challenger,out/'first_test_weights.json')
    recommendation = {'working_model':decision['working_recipe']+'_model',
        'new_model_passed_replacement_gate':decision['replacement_gate_passed'],
        'new_weights_source_run':new_search.id,'new_weights_source_rank':1,
        'new_weights_decimal_places':8,'new_weights_exploratory_only':True,
        'first_real_test_candidate':'original_consensus_challenger',
        'first_real_test_weights':'first_test_weights.json',
        'operational_W_changed':False,'first_test_priority_changed':False,
        'simulator_calls':0,'new_measured_rows':0,'independent_SOTA_claim':False}
    write(out/'recommendation.json',recommendation)
    working = recommendation['working_model']
    bundle = out/working
    # Verify the portable bundle against the exact serialized bytes of its canonical export.
    env = os.environ.copy()
    env.pop('PYTHONPATH',None)
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    command = [str(Path(sys.executable).with_name('sdm')),'predict','--model-run',str(bundle),
               '--weights',str(out/'exploratory_weights.json')]
    replay = subprocess.run(command,cwd=REPO_ROOT,env=env,capture_output=True,text=True,check=True)
    prediction = json.loads(replay.stdout)
    points = check_weights(json.loads((out/'exploratory_weights.json').read_text()))
    mean,scale = resolve_run(str(bundle)).load_model().predict(points)
    np.testing.assert_allclose(prediction['predicted_mean'],mean,atol=1e-12,rtol=0)
    np.testing.assert_allclose(prediction['predicted_std'],scale,atol=1e-12,rtol=0)
    write(out/'verification.json',{'cli_command':command,'cli_prediction':prediction,
        'canonical_export_replayed':True,'original_challenger_bytes_preserved':
        sha256_of(challenger)==sha256_of(out/'first_test_weights.json'),
        'fresh_cli_runs_clean':all(not r['meta']['git']['dirty'] for r in runs.values()),
        'read_only_data_sha256':ds.sha256,'simulator_calls':0})
    aggregate = metrics[metrics.experiment!='baseline'].copy()
    aggregate['model'] = aggregate.experiment.str.split('-').str[0]
    aggregate = aggregate.groupby('model')[['v32_score','mae','local_mae','regret','selected_actual_R','coverage']].mean().reset_index()
    pools = pd.read_csv(out/'measured/pool_summary.csv')
    region = pd.read_csv(out/'regions/selection.csv')
    region = region[region.support=='all'].groupby(['model','rule']).agg(
        regret=('regret','mean'),worst_regret=('regret','max'),
        selected_actual_R=('selected_actual_R','mean')).reset_index()
    central = pd.read_csv(out/'candidates/central.csv')
    neighbors = pd.read_csv(out/'candidates/neighbors.csv').groupby(['candidate','radius'])[
        'legacy_consensus_success'].mean().reset_index()
    subsets = pd.read_csv(out/'candidates/training_subset_summary.csv')
    prec = pd.concat([pd.read_csv(out/p/'summary.csv').assign(panel=p) for p in ('precision-old','precision-new')])
    precision = prec[prec.variant=='d8'][['panel','judge','maximum_mean_change','maximum_risk_change','maximum_mean_rank_change']]
    diversity = json.loads((out/'diversity/summary.json').read_text())
    probes = json.loads((out/'direction_probes/summary.json').read_text())
    trained = resolve_run(runs['corrected-fit']['run']).load_model()
    tuning = pd.DataFrame(trained.metrics['internal_selection_tuning'])
    readme = f'''# Selection-conditioned retraining delivery

The new recipe **{'passed' if decision['replacement_gate_passed'] else 'did not pass'}** the prespecified replacement gate.
Working surrogate: **{working}/model.pkl**. The full fresh corrected model is retained for review,
including negative results. The operational W and original consensus challenger priority are preserved.
No real simulator was called; all newly proposed W have unknown actual R. Historical measured R below
belongs only to held-out rows in the unchanged2048-row dataset.

## Fresh frozen-v32 evaluation

Three-seed means (higher v3.2 is better; lower MAE/regret/R is better):

{table(aggregate)}

{table(metrics[['experiment','v32_score','mae','regret','selected_actual_R']])}

Exact clean CLI run IDs, commits, configs, packages, timing and full metrics are in runs.json.
Per-row held-out predictions and frozen per-fold scores are archived. All outer labels were excluded
from the complete fitted procedure, including residual-correction strength tuning. The frozen protocol
and all its data/scorer hashes are unchanged. These reused folds and three model seeds do not constitute
fresh independent evidence or establish SOTA.

## What changed in training

Keep the proven three-family OOF convex blend. Three internal60/20/20 trials separate base fit,
residual fit and measured-pool selection validation. A fixed-alpha ridge correction uses low-prediction
tail, model disagreement, distance and concentration features. Tune only its multiplier against measured
top1 regret, with a global MAE guard. Refit the corrector on honest nested OOF residuals and the base on
all allowed training rows. No continuous optimizer point receives an invented R label. Internal trials
overlap between repetitions and are tuning evidence only. The final mean's residual corrector uses OOF
labels; the honest final evaluation is the outer protocol, not an in-sample calibration score.

Full-data selected multiplier: **{trained.strength_}** (outer models can select different values).

{table(tuning)}

The error proxy is raw label-excluded absolute residual quantiles in distance bins, enlarged by the
absolute applied correction. It is heuristic; no conditional interval or real-margin guarantee follows.
Training on OOF predictions from smaller models and applying to a full-data base introduces a transfer
assumption. The outer evaluation measures consequences rather than assuming this transfer is harmless.
Measured candidate pools still underrepresent continuously optimized candidates with predictions below
the observed minimum. candidates/central.csv reports each point's maximum standardized correction-feature
magnitude and applied correction. These diagnostics do not certify support or correct optimization bias
outside the measured pool distribution.

## Selection on measured candidate pools and region shift

Disjoint-in-each-fold label-free pools of approximately50 historical held-out rows:

{table(pools)}

Signed optimism = measured R minus predicted R at the selected row. Positive means optimistic prediction;
negative means pessimistic. Pools across model seeds/repeats share observations, so their counts are not
independent sample counts. See measured/fold_selection.csv for whole frozen-fold choices.

Auxiliary label-free KMeans three-region holdouts (separate from v3.2):

{table(region)}

See regions/selection.csv for per-region results, restricted-support pool sizes and omitted choices when
a pool is empty. Shift generalization remains unresolved. Auxiliary regions are reused from the prior study.

## Precision audit

All15 previous method representatives, recommendations/controls, and all200 new proposals are compared
raw,8/6/4 decimals and <1e-6 cleanup plus8 decimals. Both new fits and legacy HistGB/XGBoost/ExtraTrees
rescore every representation; rankings, risk, novelty and distance are recomputed. Source raw values
remain in precision-*/raw_inputs.json. Canonical exports normalize by exact integer row totals and use
fixed decimal notation. No arbitrary positive lower bound is imposed.

Eight-decimal sensitivity:

{table(precision)}

See precision-*/summary.csv and predictions.csv.gz for6/4-decimal and cleanup effects. A small input
change can cross a tree split. Surrogate agreement cannot establish that a real simulator treats tiny
positive values and exact zeros identically. Exported candidates are rescored after serialization.

## Candidates and dual stability

exploratory_weights.json is canonical8-decimal rank1 of `{new_search.id}`; raw source W is retained in
exploratory_raw_weights.json. It was fixed before fresh-direction and training-subset audits; no
post-audit best-of-many promotion. Both trained models each produced two50-point CLI batches, with
optimizer seeds2040/2041, trust radius<=.75, actual neighborhood radii .02/.05 and budget12000 inputs.

{table(central[['candidate','distance_ratio','reference_mean','corrected_mean','extratrees_mean']])}

Fresh actual-radius128 directions per radius, seeds2043/2044/2045, legacy correlated three-judge success
fraction for an improvement>3 over p01969:

{table(neighbors)}

Five fresh matched90% training subsets, seeds2046..2050, refit both complete recipes. Central mean and
risk are distinct from input stability and are not a real-R measurement:

{table(subsets)}

Full neighbors, subset indices, selected multipliers, predictions and source provenance are archived.
The first_test_weights.json retains the exact original challenger bytes, and operational W is unchanged.

## Diversified exploratory pack

diversity/weights.json contains{diversity['exported']} novel canonical feasible candidates selected by
farthest-first directions from p01969, initialized with the fixed corrected-search rank1. Minimum pair
distance={diversity['minimum_pair_distance']:.6f}; median={diversity['median_pair_distance']:.6f};
maximum direction cosine={diversity['maximum_direction_cosine']:.6f}; distinct nearest observed anchors=
{diversity['distinct_nearest_anchors']}. Correlated directions can remain; geometry does not create
independent model evidence. This supplementary pack does not replace the previously prepared online50.

The candidate-only pack **failed meaningful direction diversification**: all20 share one nearest
observed anchor and median direction cosine remains high. Preserve this negative result. A separately
declared adaptive, direction-only extension creates direction_probes/weights.json: the fixed exploratory
point plus19 probes around different farthest-first anchors among the measured lowest20% R rows,
seed2051, actual raw perturbation radius .2 times median spacing. No prediction screening or new-model
reselection. All20 serialized points are novel, feasible and inside the trust limit. Anchor measured R
is explicitly separate from the unknown probe R in the manifest.

Direction probes: distinct nearest anchors={probes['distinct_nearest_anchors']}, median direction
cosine={probes['median_direction_cosine']:.6f}, maximum={probes['maximum_direction_cosine']:.6f},
minimum pair distance={probes['minimum_pair_distance']:.6f}. These are exploration points, not20
independent winners or a claimed lower-R batch. Their central predictions are rescored by all five judges.
The original online50 packet and first-real-test priority remain unchanged.

## Use and replay

```sh
env -u PYTHONPATH uv run --locked --group gui sdm predict \\
  --model-run analysis/selection_training/delivery/{working} \\
  --weights analysis/selection_training/delivery/exploratory_weights.json
```

run.py phases: evaluate,fit,search,replay. audit.py phases: precision-old,precision-new,measured,regions,
candidates,diversity. The frozen plan is ../README.md. Full-fit CLI reproduction passed; portable model
CLI replay matches the exported bytes. tests are recorded in test_result.json. All source changes are
on azq within the requested repository. No remote push, official submission or simulator call.
The separate adaptive direction-only pack is generated by probes.py.
'''
    (out/'README.md').write_text(readme)
    provenance(out/'provenance.json',ds,[REPO_ROOT/'tests/test_selection_training.py',RESULTS/'test_result.json'])
    write(out/'checksums.json',{str(p.relative_to(out)):sha256_of(p)
        for p in sorted(out.rglob('*')) if p.is_file() and p.name!='checksums.json'})
    print('Published '+str(out)+': '+working,flush=True)


if __name__ == '__main__':
    publish()
