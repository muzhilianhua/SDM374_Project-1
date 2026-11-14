# Validation and limits

The release was tested on Windows x64 / Python 3.11.18 in a newly created uv
environment, with no access to the original extracted modules on its import path.
The [recorded source-only report](../evidence/source_release_validation.json)
lists the versions and actual module paths. All task helpers are `.py` under
`source_view/`; CityLearn is the unmodified upstream source under `vendor/`.

## Original-evaluator comparison

The original input and stdout are preserved in
[continued_real_payload.json](../evidence/continued_real_payload.json) and
[continued_real_validation.json](../evidence/continued_real_validation.json).
The new release re-ran those exact weights locally; it did not call the EXE again.

| Case | Source-only release R | Original EXE R, parsed as float | Absolute difference |
| --- | ---: | ---: | ---: |
| Current best | 103.46086008194834 | 103.46086008194834 | 0 |
| Neighbour | 103.61721213231795 | 103.61721213231796 | 1.4210854715202004e-14 |
| Previous record | 103.80394676583819 | 103.80394676583820 | 1.4210854715202004e-14 |

The original stdout printed the last value as `103.80394676583821`; its parsed
Python float is shown above. The difference is consistent with double-precision
formatting and accumulation order. Reference verification uses an absolute
tolerance of `1e-10` and fails with exit code 1 if exceeded.

## Hourly regression and assets

`evidence/exact_local_best.npz` contains the independent complete evaluator's
8 by 3 by 168 net-load trace for the current best. The release test compares all
three buildings' week-one values exactly, rather than only checking the final R.
Asset verification checks all 67 original scenario/data/model files and 57 upstream
CityLearn distribution files against `assets_manifest.json` before evaluation.
Git attributes preserve their bytes on Windows checkouts.

The original checkpoints and data are included unchanged. The original executable
and the frozen Python bytecode are not runtime dependencies. The public CityLearn
package is vendored to avoid installing its large, unused autosizing dependencies;
the task explicitly disables PV/Battery autosizing, as the original worker did.

## Interpretation

These are finite, reproducible regression cases, not a formal equivalence proof.
Changing library versions, CPU numerical behavior, model/data bytes, unused
branches or scenarios requires new validation. Windows x64 is the supported uv
target; other operating systems have not been validated.

The best measured original score remains above 100 and is not a global optimum
claim. The release does not mix these reconstruction-derived observations into
the repository's frozen surrogate/online-policy data or alter official evaluator
usage counters. Approximate Numba search accelerators are not included in the
runtime or used to substantiate these scores.
