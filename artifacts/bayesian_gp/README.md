# Regenerate the untracked GP

Use the supported editable installation, from the repository root:

```sh
uv sync --locked
uv run --locked --no-sync sdm train -c experiments/bayes-study-gp-ard.toml
```

Use Python3.11 and the unchanged uv.lock. On Windows/Python3.11, use an ASCII-only checkout path
to avoid locale decoding of editable .pth paths. Clear PYTHONPATH for an installation check.
Wheel/non-editable installs do not support the current repository-root resolver.
The tracked metadata describes the reviewed refit; historical metrics are archived under
analysis/bayesian_search/delivery/historical/. The binary is deliberately untracked.
Kernel starts and optimizer diagnostics are in metrics.json. Do not load untrusted pickles.
