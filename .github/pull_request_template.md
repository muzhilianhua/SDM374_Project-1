## What and why

<!-- One paragraph. Which task ID in docs/team.md (e.g. T08) does this address, and who owns it? -->

## Evidence

| Run id | Command | Key metrics |
|---|---|---|
|  | `uv run sdm evaluate -c experiments/...` |  |

<!-- Runs must come from committed code (no `*` after the commit in `uv run sdm runs`). -->

## Checklist

- [ ] `uv run pytest -q` passes (GUI tests too if `app/` changed: `uv sync --group gui` first)
- [ ] New plugins appear in `uv run sdm list` with sensible defaults
- [ ] No changes to read-only course files or to `tests/test_baseline_regression.py`
- [ ] `docs/interfaces.md` / `docs/decisions.md` updated if a shared interface or decision changed
- [ ] Task status and evidence updated in `docs/team.md` (status `待审`, PR / run ids)
