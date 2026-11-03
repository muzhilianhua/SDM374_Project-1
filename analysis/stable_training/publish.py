"""Package completed experiments, honest calibration evidence and exact exploratory weights."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
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

BASE = REPO_ROOT/'analysis/stable_training'
RESULTS = BASE/'results/study'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def table(frame):
    return '\n'.join(['| '+' | '.join(frame.columns)+' |',
        '|'+ '|'.join(['---']*len(frame.columns))+'|'] +
        ['| '+' | '.join(str(x) for x in row)+' |' for row in frame.itertuples(index=False, name=None)])


def publish(out):
    out.mkdir(parents=True, exist_ok=False)
    ds = load_dataset(REPO_ROOT/'dataset_same8')
    manifest = json.loads((RESULTS/'runs.json').read_text())
    selection = json.loads((RESULTS/'selection.json').read_text())
    decision = json.loads((RESULTS/'audit_decision.json').read_text())
    profile = selection['profile']
    evidence, metric_rows, selected_rows = {}, [], []
    for label, record in manifest.items():
        run = resolve_run(record['run'])
        assert not run.read_json('meta.json')['git']['dirty']
        assert run.read_json('meta.json')['dataset']['sha256'] == ds.sha256
        evidence[label] = {**record, 'config': run.config_path.read_text()}
        if run.read_json('meta.json')['kind'] == 'evaluate':
            metric_rows.append({'recipe': label, **record['metrics']})
            outer = pd.read_csv(run.path/'predictions.csv')
            for split, group in outer.groupby('split_id', sort=False):
                best = group.loc[group.predicted_mean.idxmin()]
                selected_rows.append({'recipe': label, 'split_id': split,
                    'selected_id': best.candidate_id, 'selected_actual_R': float(best.actual),
                    'selected_predicted_R': float(best.predicted_mean),
                    'pool_best_actual_R': float(group.actual.min()),
                    'regret': float(best.actual-group.actual.min())})
            for name in ('predictions.csv', 'fold_scores.csv'):
                pd.read_csv(run.path/name).to_csv(out/(label+'-'+name+'.gz'), index=False)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out/'evaluation_metrics.csv', index=False)
    pd.DataFrame(selected_rows).to_csv(out/'frozen_selected_measured_points.csv', index=False)
    write(out/'run_evidence.json', evidence)
    for name in ('selection.json', 'audit_decision.json'):
        shutil.copyfile(RESULTS/name, out/name)
    for name in ('regions', 'candidates'):
        shutil.copytree(RESULTS/name, out/name)
    training = resolve_run(manifest[profile+'-fit']['run'])
    bundle = out/'trained_model'
    bundle.mkdir()
    for name in ('model.pkl', 'config.toml', 'meta.json', 'metrics.json'):
        shutil.copyfile(training.path/name, bundle/name)
    model = training.load_model()
    honest_run = resolve_run(manifest['honest-fit']['run'])
    honest_bundle = out/'calibrated_reference'
    honest_bundle.mkdir()
    for name in ('model.pkl', 'config.toml', 'meta.json', 'metrics.json'):
        shutil.copyfile(honest_run.path/name, honest_bundle/name)
    honest_checks = []
    for seed in (2026, 2027, 2028):
        before = pd.read_csv(resolve_run(manifest[f'reference-{seed}']['run']).path/'predictions.csv')
        after = pd.read_csv(resolve_run(manifest[f'honest-{seed}']['run']).path/'predictions.csv')
        pd.testing.assert_series_equal(before.candidate_id, after.candidate_id)
        difference = float(np.max(abs(before.predicted_mean-after.predicted_mean)))
        assert difference == 0
        coverage = manifest[f'honest-{seed}']['metrics']['std_coverage']
        honest_checks.append({'seed': seed, 'mean_prediction_max_abs_difference': difference,
                              'coverage': coverage, 'coverage_gate': .55 <= coverage <= .80})
    write(out/'calibration_ablation.json', {'adaptive_follow_up': True,
        'mean_recipe_unchanged': True, 'checks': honest_checks,
        'calibration_gate_passed': all(c['coverage_gate'] for c in honest_checks),
        'predictive_score_improvement_claimed': False})
    calibration_ok = all(c['coverage_gate'] for c in honest_checks)
    calibration = pd.DataFrame({'candidate_id': ds.ids, 'actual': ds.y,
        'nested_prediction': model.calibration_prediction_,
        'raw_scale': model.calibration_raw_scale_,
        'distance_ratio': model.calibration_distance_ratio_})
    calibration.to_csv(out/'nested_calibration.csv', index=False)
    write(out/'nested_calibration_folds.json', [{k: v.tolist() for k, v in fold.items()}
                                             for fold in model.calibration_folds_])
    search = resolve_run(manifest[f'{profile}-neighbors-2026']['run'])
    point = check_weights(search.read_json('weights.json'))[0]
    write(out/'exploratory_weights.json', point.reshape(3, 7).tolist())
    inputs = json.loads((out/'candidates/inputs.json').read_text())
    points = check_weights(inputs['weights'])
    expected = pd.read_csv(out/'candidates/central_predictions.csv')[profile+'_mean'].to_numpy()
    command = [str(Path(sys.executable).with_name('sdm')), 'predict', '--model-run', str(bundle),
               '--weights', '-']
    cli = subprocess.run(command, input=json.dumps(points.tolist()), text=True,
                         capture_output=True, check=True, cwd=REPO_ROOT)
    predictions = json.loads(cli.stdout)
    np.testing.assert_allclose(predictions['predicted_mean'], expected, rtol=1e-12, atol=1e-12)
    check_weights(point)
    # Never accept a candidate bundle containing an invalid search point.
    for label, record in manifest.items():
        if '-central-' in label or '-neighbors-' in label:
            run = resolve_run(record['run'])
            values = check_weights(run.read_json('weights.json'))
            assert len(values) == 50 and len(np.unique(np.round(values, 9), axis=0)) == 50
            assert min(p['nearest_train_distance'] for p in run.read_json('proposals.json')) > 1e-6
            assert max(p['nearest_train_distance'] for p in run.read_json('proposals.json')) <= .75*record['metrics']['train_median_nn_distance']+1e-10
    write(out/'verification.json', {'all_search_weights_feasible_distinct_and_distance_limited': True,
        'model_binary_identical_to_train_run': sha256_of(bundle/'model.pkl') == sha256_of(training.path/'model.pkl'),
        'cli_prediction_max_abs_difference': float(np.max(abs(np.array(predictions['predicted_mean'])-expected))),
        'cli_prediction_points': len(points), 'all_run_sources_committed': True,
        'actual_candidate_R_unknown': True, 'simulator_calls': 0})
    honest_search = resolve_run(manifest['honest-neighbors-2026']['run'])
    honest_point = check_weights(honest_search.read_json('weights.json'))[0]
    write(out/'calibrated_exploratory_weights.json', honest_point.reshape(3, 7).tolist())
    honest_command = [str(Path(sys.executable).with_name('sdm')), 'predict',
                      '--model-run', str(honest_bundle), '--weights', '-']
    replay = subprocess.run(honest_command, input=json.dumps(points.tolist()), text=True,
                            capture_output=True, check=True, cwd=REPO_ROOT)
    honest_predictions = json.loads(replay.stdout)
    np.testing.assert_allclose(honest_predictions['predicted_mean'],
        pd.read_csv(out/'candidates/central_predictions.csv')['honest_mean'], rtol=1e-12, atol=1e-12)
    # Useful audit aggregates: these are predictive stability diagnostics, never new measurements.
    subs = pd.read_csv(out/'candidates/training_subset_predictions.csv')
    stability = subs.groupby(['model', 'candidate']).agg(
        mean=('predicted_mean', 'mean'), spread=('predicted_mean', 'std'),
        min_delta_to_incumbent=('delta_to_incumbent', 'min'),
        max_delta_to_incumbent=('delta_to_incumbent', 'max'),
        improving_pairs=('delta_to_incumbent', lambda x: int((x < 0).sum())))
    stability.to_csv(out/'training_subset_summary.csv')
    region = pd.read_csv(out/'regions/selection.csv')
    region_all = region[region.support == 'all'].groupby(['model', 'rule'])[
        ['regret', 'selected_actual_R', 'mae', 'low20_mae', 'coverage']].mean().reset_index()
    region_all.to_csv(out/'region_summary.csv', index=False)
    frozen_selected = pd.DataFrame(selected_rows)
    actual_summary = frozen_selected.assign(family=frozen_selected.recipe.str.rsplit('-', n=1).str[0])
    actual_summary = actual_summary.groupby('family').agg(
        model_seeds=('recipe', 'nunique'), selected_actual_R=('selected_actual_R', 'mean'),
        regret=('regret', 'mean')).reset_index()
    actual_summary.to_csv(out/'frozen_selected_R_summary.csv', index=False)
    restricted = region[(region.support == 'ratio<=.75') & (region.rule == 'mean')
                        & (region.model == 'reference')]
    support_count = int(restricted.pool_size.sum())
    worst_region_regret = float(region[(region.model == 'reference') & (region.rule == 'mean')
                                       & (region.support == 'all')].regret.max())
    shutil.copyfile(REPO_ROOT/'analysis/robust_search/delivery/challenger_weights.json',
                    out/'first_test_weights.json')
    write(out/'recommendation.json', {
        'working_model': 'calibrated_reference' if calibration_ok else 'artifacts/best_model',
        'predictive_score_improvement': False,
        'regularized_recipe_replacement_gate_passed': decision['replacement_gate_passed'],
        'first_real_test_candidate': 'original_consensus_challenger',
        'first_real_test_weights': 'first_test_weights.json',
        'first_test_priority_changed': False, 'operational_W_changed': False,
        'new_search_weights_exploratory_only': True,
        'region_generalization_solved': False,
        'official_submissions': 0, 'simulator_calls': 0})
    verification = json.loads((out/'verification.json').read_text())
    verification.update({
        'honest_model_binary_identical_to_train_run': sha256_of(honest_bundle/'model.pkl') == sha256_of(honest_run.path/'model.pkl'),
        'honest_cli_replay_points': len(points),
        'calibration_only_outer_means_exact_on_all_seeds': True,
        'first_test_challenger_exact_source_bytes': sha256_of(out/'first_test_weights.json') == sha256_of(REPO_ROOT/'analysis/robust_search/delivery/challenger_weights.json'),
        'test_suite': json.loads((RESULTS/'test_result.json').read_text())})
    write(out/'verification.json', verification)
    columns = ['recipe', 'v32_score', 'mae', 'v32_local_mae', 'v32_top1_regret', 'std_coverage']
    display = metrics[columns].copy()
    for c in columns[1:]:
        display[c] = display[c].map(lambda v: f'{v:.4f}')
    status = ('passes the prespecified surrogate-replacement gate' if decision['replacement_gate_passed']
              else 'does not pass the prespecified surrogate-replacement gate')
    readme = f'''# Stable training: completed comparison

The selected **{profile}** recipe {status}. This decision concerns the surrogate recipe only;
the operational W, old best-model artifact and actual submission proposal remain unchanged.
No simulator was queried. The new exploratory candidate has no measured R.

## Frozen-v32 evidence

{table(display)}

Three-seed mean score: reference **{decision['reference_mean_score']:.4f}**, new
**{decision['new_mean_score']:.4f}**. Exact run IDs/configs/provenance are in run_evidence.json;
all outer predictions and per-fold scores are archived. The profile was frozen before seed audit;
no post-audit reselection occurred. A reused benchmark is selection-aware, not independent SOTA.

Mean historical R of the actually selected held-out points (same measured pools across recipes):

{table(actual_summary.round(4))}

These are measured historical labels of held-out dataset rows, not real evaluations of any new W.

## What changed in training

Five 90% training subsamples, stability-regularized convex family weights, and nested calibration:
each calibration prediction excludes that point's label from every predictive and blending step.
The residual scale is empirical, not a guaranteed interval. Full refitting changes sample size;
report outer coverage rather than inferring validity from internal calibration.

The two regularization profiles were compared under the unchanged protocol. This experiment is
materially different from rerunning the previous model on identical data and parameters.
The nested procedure costs more fits; exact CLI runtimes and thread limits are recorded.

## Adaptive calibration-only ablation

After both initial profiles failed discovery, a separately documented fixed ablation preserved the
old stacking mean exactly and recalibrated absolute residuals using complete label-excluded nested
fits. `calibrated_reference/` contains this fitted model; calibration_ablation.json verifies exact
outer-mean agreement at all three matched seeds. Coverage is reported per seed; no predictive-score
gain is claimed. This follow-up does not change the failed original profile-selection decision.

Calibration-only coverage gate passed: **{calibration_ok}**. The working reference is
`calibrated_reference/` when this gate passes: preserve the validated best mean while correcting
the calibration procedure. The regularized `trained_model/` remains an exploratory comparison.
Empirical coverage is not evidence of a statistically significant calibration improvement.

calibrated_exploratory_weights.json is rank1 of `{honest_search.id}`, using this corrected error
proxy and the same neighborhood-search policy. It remains exploratory. Training-subset mean audits
for this recipe equal the reference by construction; its interval coverage must be checked separately.

## Measured-region stress test (auxiliary)

Three label-free KMeans regions were each held out; models refit using only the other regions.
Each rule selects from the same actually measured held-out pool. The selected central point has
a known historical R; the perturbed neighbors do not. These results are not frozen-v32 scores.

{table(region_all.round(4))}

Also inspect regions/selection.csv for every fold, the support<=.75 pool sizes, and empty pools.
Mean summaries alone can hide one poor region. Regions and observations were reused; no fresh
labels or independent model-validation dataset were created.

**Regional generalization remains unresolved.** The reference's worst region has first-choice
regret{worst_region_regret:.4f}. Only{support_count}/2048 held-out rows satisfy distance ratio<=.75
in this stress test; restricting support changes the eligible pool and may exclude its true best.
The three-region results do not establish a universal winner. Neither neighborhood stability nor
approximately70% global error coverage proves that the selected optimum is reliable under shift.
Future evaluator budget should target insufficiently supported regions and validate new labels.

## Candidates and stability

Two models x two search strategies x two optimizer seeds,50 proposals per run. Ordinary search
uses12000 central trials; neighborhood search uses9600 central trials plus64-point reranking at
actual radii .02/.05,16 directions each, within12000 optimizer input evaluations. CLI export
predictions are bookkeeping outside that optimizer budget. Fresh audits use128 directions per
radius and seeds2037/2038/2039, separate from search directions. Input-neighbor performance remains
a model prediction. Legacy judges are correlated, and ExtraTrees dissent remains visible.

The first real-test priority remains the original consensus challenger: exact source bytes are
copied to first_test_weights.json. New searches do not dominate every stability criterion and are
not promoted automatically. recommendation.json separates this priority from exploratory exports.

candidates/central_predictions.csv contains the full fixed shortlist. candidates/neighbor_summary.csv
contains fresh-direction checks; training_subset_summary.csv contains five new matched90% training
subset audits. New-model mean audits execute its exact final predictive-core fit without residual
calibration, because calibration cannot affect mean predictions. This saves computation without
changing the mean estimator. No error-scale or interval claim is made for these reduced audits.

**exploratory_weights.json** is prespecified as rank1 of `{search.id}` (seed2026), not a winner
chosen after inspecting the fresh-direction audit. Original training-subsample replacement rules
remain separate. Do not promote this point to an actual submission solely from self-predicted R.

## Load and reproduce

Both model folders are portable CLI model-run bundles; source binary bytes are preserved.
From the repository root:

```sh
env -u PYTHONPATH uv run --locked --group gui sdm predict \\
  --model-run analysis/stable_training/delivery/calibrated_reference \\
  --weights analysis/stable_training/delivery/calibrated_exploratory_weights.json
```

The source plan and scripts are under analysis/stable_training/. Execute run.py phases discovery,
audit,fit,calibration,search,calibration-search; then diagnostics.py regions/candidates and publish.py. Source hashes and model
versions are recorded. checksums.json covers the package except itself.
'''
    (out/'README.md').write_text(readme)
    (bundle/'README.md').write_text(f'# {profile} stable-stack bundle\n\nSource run: {training.id}.\nSee ../README.md for evidence and limitations.\n')
    (honest_bundle/'README.md').write_text(f'# Honestly calibrated reference stack\n\nSource run: {honest_run.id}.\nMean recipe unchanged; empirical nested residual calibration. See ../README.md.\n')
    sources = [p for p in BASE.glob('*.py')] + [BASE/'README.md',
        REPO_ROOT/'src/sdm/models/stable_stack.py', REPO_ROOT/'src/sdm/models/honest_stack.py',
        REPO_ROOT/'src/sdm/optimizers/neighborhood_trust.py',
        REPO_ROOT/'src/sdm/optimizers/_neighborhood.py', REPO_ROOT/'uv.lock',
        REPO_ROOT/'tests/test_stable_training.py', RESULTS/'test_result.json']
    write(out/'provenance.json', {'git': git_state(), 'dataset_sha256': ds.sha256,
        'packages': {p: version(p) for p in ('numpy', 'pandas', 'scikit-learn', 'catboost', 'xgboost')},
        'sources': {str(p): sha256_of(p) for p in sources}, 'simulator_calls': 0})
    write(out/'checksums.json', {str(p.relative_to(out)): sha256_of(p)
                              for p in sorted(out.rglob('*')) if p.is_file()})
    print(f'Published {out}: {profile}, gate={decision["replacement_gate_passed"]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=BASE/'delivery')
    publish(parser.parse_args().out.resolve())
