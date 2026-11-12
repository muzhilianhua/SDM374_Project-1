# T15/T16: recoverable online proposal strategy

## Design goal

The online evaluator permits at most 20 sequential calls and at most 50 weight vectors per call. The first proposal is trained only on the 2,048 released rows. After each successful call, the measured rows are appended and the model is refit; failed and lost calls consume quota and their sent points remain excluded from every later proposal.

## Batch proposal

`online-batch` first builds a deterministic local candidate pool around the best observed rows. The trust-region radius is `r * median_spacing(X)`, where `r` is bounded in `[0.25, 1.5]`. Dirichlet draws are supplemented by convex interpolation when a small radius would otherwise leave too few feasible points. Candidates are filtered against training rows, successful online rows, and failed/lost sent rows before screening.

The acquisition is split into two equal arms for a solver-sized batch:

- **Exploitation** ranks `mean + risk * std` and chooses low predicted ramping values with separation.
- **Exploration** ranks normalized mean, uncertainty, and distance from observed data, then chooses separated points.

The arms are interleaved before the online loop truncates an oversupply (`50 + candidate_margin`). Thus the first 50 candidates retain a 25/25 exploit/explore split even though the policy returns additional refill candidates. All generated vectors satisfy the 3-building simplex constraints and are unique after rounding to the screening tolerance.

## Adaptive trust region and recovery

A successful batch improves the incumbent when its measured minimum is below the previous best by more than the tolerance. After the configured success patience, `r` expands by `1.5`; a non-improving batch shrinks it by `0.7`. The bounds prevent unbounded expansion or a radius that cannot produce a full batch.

The policy is stateless between calls. On every invocation it reconstructs `r`, the proposal index, and the deterministic seed from `OnlineState`, successful call records, the attempt quota, and the config. This makes a resumed process produce the same batch as an uninterrupted process. A failed or lost attempt does not update `r`, because it has no measured objective; its points are still listed in `X_unmeasured` and cannot be sent again.

## Evidence and limits

The implementation has targeted tests for 75-candidate oversupply, a balanced 50-point prefix, radius expansion, failed-point exclusion, deterministic resume, strict batch logging, and journal quota accounting. The full suite passes with `318 passed, 2 skipped, 1 xfailed` in the pinned project environment. A two-call synthetic smoke run exercises the complete T08 path and confirms the second call uses the expanded radius.

Synthetic replay is a mechanism check, not a formal solver score. No real evaluator result is claimed here; the final online score must come from the course solver with the frozen configuration and its run folder.
