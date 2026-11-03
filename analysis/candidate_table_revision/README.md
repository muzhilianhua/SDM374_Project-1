# Audited offline-candidate decision page

Role C revision of the user-supplied HTML, after the independent perturbation audit and unchanged-data
retraining. The original HTML outside the designated repository remains read-only. The Chinese HTML
is a user-facing decision page; source documentation and code are English.

This is a post-audit decision proposal, not a new experiment, independent validation, settled team
decision, automatic replacement of the operational W, or actual submission.

Changes:

- Keep consensus challenger as the priority real-test proposal.
- Promote fixed-blend search2027/rank 2 to a robustness alternative; keep stack 2027/rank 2 as another.
- Demote fixed-blend search2027/rank 1 to an exploration alternative rather than an automatic second
  submission. The nearest0.05 runner-up is statistically unresolved under conditional direction
  sampling, and rankings change with perturbation geometry.
- Display actual-radius audit percentages and medians, separating them from original 40-direction
  HTML claims. Do not turn neighbor predictions into a true-R margin.
- Distinguish the known p01969 label from the unverified reported baseline 119.0916. Preserve
  ExtraTrees dissent and model dependence. Do not infer an entire region's response from one result.
- Make submission2 conditional on verified submission1 feedback, evaluator compatibility, new-data
  retraining and a fresh candidate comparison; no automatic fallback into a correlated region.
- Keep risk1 for the existing consensus envelope. If increasing risk is desired, explicitly
  implement max(member mean+k*member proxy); multiplying the wrapper envelope is a different rule.
- Stop repeating same-data retraining for hoped-for gains: the retraining comparison reproduced all
  evaluations, 1554-input predictions, and 50 proposals exactly.

Generate and verify from the repository root:

```sh
env -u PYTHONPATH uv run --locked --group gui python analysis/candidate_table_revision/build.py \
  --source-html ../离线候选表.html --out analysis/candidate_table_revision/delivery
```

The generator uses archived audit tables, exports exact audited W, verifies the JSON through the
old and newly retrained stack CLIs, records source hashes and writes a standalone revised HTML,
decision manifest, CSV and checksums. No dataset, model, scoring protocol or original W is edited.
