# V4 conservative restart study

`policy_v4_restart.py` is an analysis-only subclass of V4. It keeps V4's full
proposal batch and activates only after one building has failed to produce a
single-building improvement for three consecutive calls. At most three points
for each stalled building are replaced. Restart heads are selected at runtime
from the best training rows and blocks measured in the current run, then given
two small local perturbations. The policy reads no historical evaluator log and
does not contain a known low point or a fixed weight vector.

## Checks completed

- Synthetic replay: four seeds, 12 calls, 50 points per call. Both V4 and the
  restart variant produced legal, non-repeated batches. Restart activated only
  after call 7 and replaced 54 points across each 12-call trajectory. The
  smooth stand-in reached the same final value as V4, so it does not establish
  an improvement.
- Real solver replay: seed 2026, calls 1–10. The solver executable matched the
  pinned SHA-256 in `solver/CHECKSUMS.sha256`. The running best was
  `113.2261, 110.6348, 110.4127, 109.6163, 109.4430, 109.4029, 109.2274,
  109.2130, 109.1219, 109.0317`, identical to the recorded V4 seed-2026
  trajectory through call 10. Restart first activated at call 9 for building 1
  and replaced three points; that call's minimum remained `109.1219`.

The result is currently a neutral robustness candidate rather than a proven
improvement. It stays under `analysis` and does not change `submission/`.

The synthetic harness is:

```powershell
python -X utf8 analysis/online_rehearsal/rehearse_restart.py `
  --seeds 1 7 11 2026 --calls 12 --local-rows 120
```

