# T29 slide material: proposal and trust-region recovery

## Slide 1 — Why the batch is mixed

- 20 calls × at most 50 vectors makes each call a portfolio decision.
- Exploit: low `mean + risk·std` near promising rows.
- Explore: uncertainty and distance reward new local basins.
- Interleave 25 exploit + 25 explore in the solver prefix; keep extra candidates for screening/refill.

**Speaker note:** The first call uses only the released training table. Later calls add only successful measurements.

## Slide 2 — Trust-region control

- Radius: `r × median_spacing(X)` around the best observed centers.
- Improvement → `r ← min(1.5, 1.5r)`.
- No improvement → `r ← max(0.25, 0.7r)`.
- Small-radius interpolation fallback still creates 50 feasible points.

**Speaker note:** These bounds are a conservative policy choice. They are not a claim that replay establishes a globally optimal radius.

## Slide 3 — Resume and safety

- Rebuild model and radius from `OnlineState` and completed call records.
- Failed/lost batches count against quota and enter `X_unmeasured`.
- Screening rejects constraint violations, training duplicates, sent duplicates, and batch duplicates.
- Strict `call,point,n` snapshots and `attempts.jsonl` prevent the dashboard from understating quota.

**Demo:** Open the Online Console, select a run folder, and show the attempt ledger, per-call screening table, and best-so-far curve.
