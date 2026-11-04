"""Generate current review guidance without rewriting historical experimental records."""
import hashlib,json
from pathlib import Path
import shutil
from sdm.runs import git_state
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'analysis/bayesian_search/delivery'
CURRENT=OUT/'current'
RECORDS=ROOT/'analysis/bayesian_search/team_records.json'
def write(path,data):path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def main():
 CURRENT.mkdir(exist_ok=True)
 records=json.loads(RECORDS.read_text(encoding='utf-8'))
 state=git_state()
 write(CURRENT/'known_results.json',records)
 write(CURRENT/'documentation_provenance.json',dict(git=state,team_records_sha256=hashlib.sha256(RECORDS.read_bytes()).hexdigest(),simulator_calls=0,official_submission2_selected=False))
 rows=json.loads((OUT/'evaluation.json').read_text())+json.loads((OUT/'extension/evaluation.json').read_text())
 byname={r['model']:r for r in rows}
 table='\n'.join('| '+name+' | '+f"{byname[name]['v32_score']:.4f} | {byname[name]['mae']:.4f} | {byname[name]['v32_top1_regret']:.4f} |" for name in ['method-blend-stack','gp-ard','gp-tree-blend'])
 decision_path=CURRENT/'decision.json'
 lead=json.loads(decision_path.read_text()).get('research_lead') if decision_path.exists() else None
 submission=records['submission_1']['R_decimal'];baseline=records['baseline_weights']['R_decimal'];bar=records['offline_bar_decimal']
 text=f'''# Bayesian exploration: current review guidance

Confirmed team results: submission1 **{submission}**, baseline W* **{baseline}**.
The offline bar is **R < {bar}** (decision17); reproducing training row p01969 only ties the bar.
Decision23 requires choosing offline submission2 from existing data, known real scores and surrogates.
**Do not use RampingSolver.exe to screen candidates. No official submission2 is selected here.**

## Historical model evidence

These are the original2026-11-02 frozen-v3.2 results, not newly recomputed scores for the reviewed source.

| Recipe | Historical v3.2 | MAE | Top1 regret |
|---|---:|---:|---:|
{table}

The mainline PR7 fixes are preserved. Historical per-row predictions, metrics and configs remain in
`cli_evidence/`; generated comparison CSVs and historical source hashes are preserved. The original
approximate119/118 report in old plans and CSV fields is historical, superseded by the confirmed
records in `current/known_results.json`. Do not reinterpret it as the team's current measurements.
The old recommendation and shortlist in `candidate_decision.json` and `shortlist/` are withdrawn as
submission guidance. The old r06 alias is archived in `historical/2026-11-02_r06_not_for_submission.json`.

## Review fixes and selection limitations

- EI is permitted only for the native raw-GP posterior. The blend's disagreement envelope remains
  heuristic and can be used only with mean/pessimistic acquisition, not as a Gaussian posterior SD.
- Current candidate execution reads and validates committed acquisition-specific TOML files, without
  rewriting configurations. Code/config must be committed and Git provenance available before execution.
- GP fitting is deterministic with zero random restarts. Changing seed does not vary it. The original
  three-seed blend comparison varies the tree half only; it does not measure GP uncertainty or total
  blend robustness. The historical mean score gain did not pass the top1-regret replacement gate.
- ARD honors each configured length start and deduplicates identical start pairs. Fresh metrics record
  seven fitted lengths, fitted noise, initial settings and optimizer iteration/status/convergence diagnostics.
  The18-iteration cap is retained for comparison, not claimed as convergence. Boundary values are visible.
- Current research candidates use ARD for ranking and a separate ExtraTrees eligibility veto, informed
  by only two real measurements. GP/tree disagreement or passing this veto does not guarantee real gain.
  The ray audit uses Euclidean distances.02 through.15 towards old ei-r10, r05 and r06; any projection
  and actual distance are recorded. It is offline research, not a solver-screened submission choice.
- Current research lead: **{lead or 'not generated'}**. Actual R is unknown and it is not official submission2.
  Current records are in `current/`. The historical r06 recommendation must not be used automatically.

## Reproduce

```sh
uv run --no-editable sdm train -c experiments/bayes-study-gp-ard.toml
uv run --no-editable sdm evaluate -c experiments/bayes-study-gp-ard.toml
uv run --no-editable python analysis/bayesian_search/candidates.py
uv run --no-editable python analysis/bayesian_search/finalize.py
uv run --no-editable pytest -q
```

Python3.11 and unchanged uv.lock are required. Commit source/configs before evidence runs. Git
provenance uses `sdm.runs.git_state`, which invokes Git and supports regular checkouts, packed refs and
worktrees; no script reads `.git/HEAD` or loose refs directly. Missing Git is reported as unknown rather
than guessed. The configured GP .5/.03 start is unchanged numerically by the diagnostic fix, but the
historical83.15 must not be presented as a fresh evaluation without a new complete CLI run.

The33.9MB fitted pickle is not tracked; retrain locally from the committed recipe. Historical checksum
and configuration records are retained. Never load an untrusted pickle. The reviewer independently
reported Windows-to-macOS refit agreement around1e-4; our earlier own-machine replay was1e-9. These
are separate checks, not an untested promise of identical cross-server floating-point results.

The GitHub branch and draft PR9 have already been published. This review revision is based on current
main after PR7. Role B's model-owner review remains required before merge; user authorization to explore
models is not represented as owner approval. No merge, solver call or official submission is performed.
'''
 history=OUT/'historical';history.mkdir(exist_ok=True)
 old=history/'README_2026-11-02.md'
 if not old.exists():shutil.copy2(OUT/'README.md',old)
 (OUT/'README.md').write_text(text,encoding='utf-8')
 (CURRENT/'README.md').write_text(text,encoding='utf-8')
 print('Current guidance regenerated; confirmed results and decision23 applied')
if __name__=='__main__':main()
