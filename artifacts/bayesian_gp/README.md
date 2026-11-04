Regenerate the ignored model.pkl with:

    uv run --no-editable sdm train -c experiments/bayes-study-gp-ard.toml

The tracked metadata describes the reviewed refit; historical metrics are archived under analysis/bayesian_search/delivery/historical/. The binary is deliberately untracked. Kernel starts and optimizer diagnostics are in metrics.json. Do not load untrusted pickles.
