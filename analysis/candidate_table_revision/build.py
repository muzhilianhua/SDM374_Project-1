"""Build a revised candidate decision page from audited tables without changing model strategy."""
from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from sdm.data import sha256_of
from sdm.paths import REPO_ROOT
from sdm.runs import git_state
from sdm.simplex import check_weights

CHOICES = [
    ('challenger', '优先实测', '共识挑战者', 'first-challenger.json'),
    ('pool-0054', '稳健备选 A', '固定融合 2027 · 第 2 名', 'alt-fixed-r2-s2027.json'),
    ('stack-2027-r2', '稳健备选 B', '堆叠融合 2027 · 第 2 名', 'alt-stack-r2-s2027.json'),
    ('fixed-2027-r1', '探索候选', '固定融合 2027 · 第 1 名', 'explore-fixed-r1-s2027.json'),
    ('incumbent', '保留原默认', '原推荐 W', 'incumbent.json'),
    ('fixed-2026-r1', '历史首选', '固定融合 2026 · 第 1 名', 'legacy-fixed-r1-s2026.json'),
    ('cat-2026-r4', '方向探针', 'CatBoost RMSE 2026 · 第 4 名', 'probe-cat-r4-s2026.json'),
]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def build(source_html, out):
    audit = REPO_ROOT / 'analysis/perturbation_audit/delivery'
    retrain = REPO_ROOT / 'analysis/retrain_comparison/delivery'
    sources = [source_html, audit / 'panel.json', audit / 'summary.csv', audit / 'pool_scores.csv',
               audit / 'paired_comparisons.csv', retrain / 'verification.json',
               REPO_ROOT / 'analysis/robust_search/delivery/challenger_weights.json',
               REPO_ROOT / 'analysis/common_judge/delivery/inputs.json',
               Path(__file__), Path(__file__).with_name('README.md')]
    for folder in (audit, retrain):
        checks = json.loads((folder / 'checksums.json').read_text())
        assert all(sha256_of(folder / name) == checksum for name, checksum in checks.items())
    panel = {r['candidate']: r for r in json.loads(sources[1].read_text())}
    summary = pd.read_csv(sources[2])
    summary = summary[summary.method == 'project-exact'].set_index(['candidate', 'radius'])
    pool = pd.read_csv(sources[3]).set_index('pool_index')
    pairs = pd.read_csv(sources[4])
    proof = json.loads(sources[5].read_text())
    assert proof['max_candidate_prediction_abs_difference'] == 0
    assert proof['max_consensus_proposal_weight_abs_difference'] == 0
    out.mkdir(parents=True, exist_ok=False)
    weights_dir = out / 'weights'
    weights_dir.mkdir()
    records, all_weights = [], []
    for name, role, title, filename in CHOICES:
        point = panel[name]
        values = check_weights(point['weights'])[0]
        target = weights_dir / filename
        if name == 'challenger':
            shutil.copyfile(sources[6], target)
        else:
            write(target, values.reshape(3, 7).tolist())
        np.testing.assert_array_equal(check_weights(json.loads(target.read_text()))[0], values)
        row = pool.loc[point['pool_index']]
        small, large = summary.loc[(name, .02)], summary.loc[(name, .05)]
        records.append({'candidate': name, 'role': role, 'title': title, 'filename': filename,
                        'source_run': point['source_run'], 'source_rank': point['source_rank'],
                        'stack_mean': float(row.stack_mean), 'distance_ratio': float(row.distance_ratio),
                        'extratrees_delta': float(row.extratrees_delta),
                        'success_002': float(small.success_fraction),
                        'success_005': float(large.success_fraction),
                        'median_002': float(small.median_delta), 'median_005': float(large.median_delta),
                        'actual_R': None, 'status': 'pending_real_evaluation'})
        all_weights.append(values)
    assert records[0]['candidate'] == 'challenger'
    assert records[1]['candidate'] == 'pool-0054'
    pd.DataFrame(records).to_csv(out / 'candidate_plan.csv', index=False)
    raw_inputs = json.loads(sources[7].read_text())
    control_weights = check_weights(raw_inputs['weights'])[[0, 2]]
    for name, values in zip(('control-baseline.json', 'control-p01969.json'), control_weights):
        write(weights_dir / name, values.reshape(3, 7).tolist())
        all_weights.append(values)
    points = check_weights(np.array(all_weights))
    payload = json.dumps(points.tolist())
    cli_checks = []
    expected = [r['stack_mean'] for r in records] + [118.73160774401767, 118.17768368712012]
    for model in ('artifacts/best_model', 'analysis/retrain_comparison/delivery/retrained_stack'):
        cmd = [str(Path(sys.executable).with_name('sdm')), 'predict', '--model-run', model, '--weights', '-']
        result = subprocess.run(cmd, input=payload, text=True, capture_output=True, check=True, cwd=REPO_ROOT)
        predictions = json.loads(result.stdout)
        np.testing.assert_allclose(predictions['predicted_mean'], expected, rtol=1e-12, atol=1e-12)
        cli_checks.append({'command': cmd, 'predicted_mean': predictions['predicted_mean'],
                           'max_abs_difference': float(np.max(abs(np.array(expected) - predictions['predicted_mean'])))})
    css = '\n'.join(re.findall(r'<style>(.*?)</style>', source_html.read_text(), re.S))
    body_rows = []
    for r in records:
        body_rows.append(f'''<tr><td><b>{escape(r['role'])}</b><span class="sub">{escape(r['title'])}</span></td>
<td class="r">{r['stack_mean']:.4f}</td><td class="r">{100*r['success_002']:.1f}%</td>
<td class="r">{100*r['success_005']:.1f}%</td><td class="r">{r['median_002']:+.2f}</td>
<td class="r">{r['median_005']:+.2f}</td><td class="r">{r['extratrees_delta']:+.2f}</td>
<td><a href="weights/{escape(r['filename'])}" download>JSON</a></td></tr>''')
    comparison = pairs[(pairs.candidate == 'pool-0054') & (pairs.radius == .05)].iloc[0]
    ci = f"[{100*comparison.conditional_normal_ci_low:.2f}, {100*comparison.conditional_normal_ci_high:+.2f}]"
    document = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>离线候选表 · 复核修订版</title><style>{css}
a{{color:var(--ink);text-underline-offset:3px}}.revision-note{{border-left:4px solid var(--ochre);padding:14px 18px;background:var(--ochre-wash)}}
.tbl{{overflow-x:auto}}.tbl table{{min-width:850px}}.links{{display:flex;gap:18px;flex-wrap:wrap}}
</style></head><body><div class="wrap">
<header><div class="eyebrow">SDM374 · MEMBER C · 2026-11-01 · AUDITED REVISION</div>
<h1>先验证共识挑战者，第二次由实测决定</h1>
<p class="dek">候选表已按独立扰动复核与重训对照更新。当前证据支持优先实测挑战者；小扰动优势较强，大扰动下与近邻备选接近。</p>
<div class="revision-note">本页是成员 C 的提交建议，尚未提交真实评测。保留原操作默认 W 与原审计门槛，最终提交由团队确认。</div></header>

<section class="section"><div class="head"><span class="eyebrow">WHAT CHANGED</span><h2>修正证据，调整备选顺序</h2></div>
<div class="main"><p>首选仍是 <b>共识挑战者</b>。实际距离为 0.02 和 0.05 时，三个裁判都预测比 p01969 改善超过 3 的邻点比例分别为 <b>81.9% / 46.5%</b>。</p>
<p>原页的 82% / 62% 来自缺失脚本的 40 方向实验，不能与本次检查当作同一次实验。固定融合 2027 的<b>第 2 名</b>升为稳健备选；原第 1 名仅保留为探索候选。</p>
<p>删除“真实余量约为 3”的推断。所有新候选都只有预测值，扰动后的中位数也不能换算为实际收益。</p></div>
<aside class="aside"><b>复核范围</b><div>29 个重点候选；主检查每个步长共 1,536 个共享方向，来自三个新随机种子。</div><div>比较最终可行权重的实际欧氏距离，避免投影后步长缩小造成假稳定。</div></aside></section>

<section class="section"><div class="head"><span class="eyebrow">SHORTLIST</span><h2>按复核结果修订候选表</h2></div>
<div class="main full"><div class="tbl"><table><thead><tr><th>角色 / 候选</th><th class="r">堆叠预测 R</th><th class="r">0.02 成功比例</th><th class="r">0.05 成功比例</th><th class="r">0.02 中位差</th><th class="r">0.05 中位差</th><th class="r">ExtraTrees 差值</th><th>权重</th></tr></thead>
<tbody class="num">{''.join(body_rows)}</tbody></table></div>
<p>成功＝堆叠、HistGB、XGBoost 均预测比各自对 p01969 的预测低超过 3。中位差取三个差值中最保守者，负值更好；ExtraTrees 差值也相对于它自己的 p01969 预测。它仍反对这些候选。</p>
<p>0.05 下挑战者为 46.5%，固定融合第 2 名为 45.1%，只差约 1.5 个百分点。后者减前者的配对方向抽样区间为 {ci} 个百分点，且其中位差略好，因此没有一个候选在所有指标上占优。</p></div>
<aside class="aside full"><b>不把比例当保证</b><div>在另一种“截断后归一化”的 40 方向实验中，0.05 下挑战者为 35%，两组备选为 40%，排名会改变。</div><div>裁判有关联：堆叠包含 XGBoost、HistGB 类成员；同向候选也不等于风险分散。</div></aside></section>

<section class="section"><div class="head"><span class="eyebrow">TWO CHECKS</span><h2>保留两条稳定性证据</h2></div>
<div class="main"><p><b>训练数据稳定性：</b>原审计在五组不同的 90% 训练子样本上重拟合，挑战者仅 3/5 组改善，未通过原替换门槛。</p>
<p><b>输入邻域稳定性：</b>固定模型、扰动控制权重，本次主指标中挑战者在两个步长、三个种子下都排名第一。这支持优先实测，不覆盖原审计，也不证明真实最优。</p>
<p><b>重训对照：</b>相同 2,048 条数据、参数与种子下，融合模型仍是 75.1137 分、MAE 4.6933；1,554 个输入预测及 50 组共识提案均与旧结果一致。已有证据不足以期待反复同数据重训带来收益。</p></div>
<aside class="aside"><b>模型不替换</b><div>保留最佳 OOF 融合；新训练文件与旧模型预测一致。下一次训练更新应由可核实的新实测驱动。</div></aside></section>

<section class="section"><div class="head"><span class="eyebrow">SUBMISSION POLICY</span><h2>第二次不再自动提交原备选 A</h2></div>
<div class="main"><div class="tree">
<div class="branch first"><div class="if">首次建议<span>团队确认格式与预算后</span></div><p>提交 <a href="weights/first-challenger.json">共识挑战者</a>，记录确切文件、评测器版本、场景与实测 R。</p></div>
<div class="branch"><div class="if">结果达到目标<span>先确认对照口径</span></div><p>保留已验证的好点，再用新观测更新模型并比较稳健备选。固定融合第 2 名可作为备选；若希望测试更激进的第 1 名，应明确这是探索选择，而非更稳的保证。</p></div>
<div class="branch"><div class="if">结果低于预期<span>检查预测与实测偏差</span></div><p>先核对评测口径，再用新观测重训。在已有实测支持的区域缩小搜索范围、重新筛选；不自动换到同一区域的另一候选。</p></div>
<div class="branch"><div class="if">偏差很大或口径不符<span>暂停固定第二次候选</span></div><p>先排查场景与模型失准。CatBoost 第 4 名只作为方向探针，不作为未经验证的保底；一个坏结果也不足以证明整片区域无效。</p></div>
</div><p>现有共识封装的风险系数保持 1。若要调高风险，需要明确采用各成员“均值＋风险系数×误差代理”的最大值，不能直接把封装后的风险项当作独立标准差。</p></div>
<aside class="aside"><b>两个参照需区分</b><div>p01969 的历史实测 R＝115.7648，是训练表最优值。</div><div>原页所述 baseline 实测 119.0916 尚缺评测记录，暂不作为已核实门槛或新增训练标签。</div><div>是否超过课程 baseline，需使用同版本评测器的可核实 baseline 结果。</div></aside></section>

<section class="section"><div class="head"><span class="eyebrow">EVIDENCE & FILES</span><h2>可追溯文件</h2></div>
<div class="main"><div class="links"><a href="candidate_plan.csv">候选与来源 CSV</a><a href="decision_manifest.json">策略与证据清单</a><a href="../../perturbation_audit/delivery/README.md">独立扰动报告</a><a href="../../retrain_comparison/delivery/README.md">重训对照报告</a></div>
<p>权重 JSON 为 3×7：楼宇为行，列依次为 cost、carbon、peak、discomfort、degradation、ramping、resilience。所有权重非负、每行和为 1；导出后用新旧融合模型再次校验预测一致。</p>
<p><a href="weights/control-baseline.json">baseline 权重参照</a> · <a href="weights/control-p01969.json">p01969 权重参照</a>。参照文件不等于已授权提交；训练表点能否用于正式提交需遵循课程规则。</p></div>
<aside class="aside"><b>版本边界</b><div>本页修订保存在项目内部，原始 HTML 保留不动。未新增模型探索、未查询模拟器、未提交官方评测。</div></aside></section>
<footer><div>结论：优先实测挑战者；升级稳健备选；按首次真实反馈决定后续。</div><div>本页比例仅描述固定模型与指定扰动分布，不是实际成功概率。</div></footer>
</div></body></html>'''
    document = document.replace('<body>', '<body>' + '<aside style="margin:20px;padding:20px;background:#fff0d5;color:#532b00;border:2px solid #b8791d"><strong>历史报告：首次提交建议已失效。</strong> 共识挑战者 P1 已实测 R = 119.71206678869203，未达标，请勿重复提交。Baseline W* = 119.0916 已获助教确认。当前在线规则为最多 20 次顺序单点调用。下文保留原实验结论供追溯；当前状态请查看 <a href="../../../SUBMISSION_READY/README.md">交付说明</a>。</aside>', 1)
    (out / 'candidate_table.html').write_text(document)
    manifest = {'status': 'historical_proposal_superseded_after_submission_1',
                'current_status_source': 'docs/decisions.md',
                'current_P1_measured_R': 119.71206678869203,
                'historical_claims_below': True,
                'priority_candidate': 'challenger', 'robust_alternatives': ['pool-0054', 'stack-2027-r2'],
                'exploratory_alternative': 'fixed-2027-r1',
                'operational_incumbent_replaced': False, 'official_submissions': 0,
                'simulator_calls': 0, 'new_training_labels': 0,
                'baseline_reported_R': 119.0916, 'baseline_report_verified': False,
                'p01969_historical_measured_R': 115.76481925917324,
                'second_submission': 'conditional on verified first feedback; no automatic alternative',
                'candidates': records,
                'limitation': 'Post-audit ranking of a screened panel with correlated judges; no true-R guarantee.'}
    write(out / 'decision_manifest.json', manifest)
    write(out / 'verification.json', {'exported_weight_count': len(points),
                                     'all_feasible': True, 'old_and_new_cli_checks': cli_checks,
                                     'first_challenger_byte_identical': sha256_of(weights_dir / 'first-challenger.json') == sha256_of(sources[6])})
    write(out / 'provenance.json', {'git': git_state(), 'sources': {str(p.resolve()): sha256_of(p) for p in sources},
                                  'new_experiments': 0, 'simulator_calls': 0})
    (out / 'README.md').write_text('''# Archived offline-candidate decision package

> Historical report: P1 subsequently measured 119.71206678869203 (failed), and baseline W*
> 119.0916 is TA-confirmed. Current rules are in docs/decisions.md; do not use the old priority.

Open candidate_table.html for the Chinese user-facing review page. candidate_plan.csv and
decision_manifest.json record the proposed role changes, exact source ranks and pending status.
The first challenger is byte-identical to the audited source; all9 candidate/control files pass
simplex checks and old/new stack CLI prediction agreement. No true measurements are fabricated.

This post-audit proposal prioritizes the challenger, promotes fixed-blend2027/rank2 as an alternative,
and makes the second submission depend on verified feedback. It preserves the incumbent and
separate training-subsample gate. The reported baseline119.0916 remains unverified. Input-neighbor
medians are not actual-R estimates. Exact-radius audit and nominal-step alternatives are distinguished.

The existing consensus wrapper's risk1 identity is preserved; changing its risk coefficient does
not implement max(member mean+k*proxy). No model, dataset, scorer, or original HTML is changed.
Source hashes and export prediction checks are in provenance.json and verification.json.
''')
    write(out / 'checksums.json', {str(p.relative_to(out)): sha256_of(p)
                                  for p in sorted(out.rglob('*')) if p.is_file()})
    print(f'Created {out / "candidate_table.html"}; {len(records)} proposals +2 controls verified.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-html', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    build(args.source_html.resolve(), args.out.resolve())
