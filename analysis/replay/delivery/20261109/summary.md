# T18 replay study

20 calls x up to 50 points, 1600-row pool, 448 known rows, 5 seeds, two truth sources.

Curves score **submitted points only**; the best of the initial rows (training incumbent) is kept as a separate reference, never as a score. Under table truth a batch holds only in-radius rows: when fewer qualify than requested the batch is smaller (see mean pts/call), and max dist ratio audits that the radius held.

Training incumbent (reference, mean over seeds): semi-synthetic 128.371; table 117.437.

| truth | surrogate | exploit | radius | best after call 1 | median calls to the run's best | mean final R | std | mean pts/call | max dist ratio |
|---|---|---|---|---|---|---|---|---|---|
| semi-synthetic | ridge2 | 0.5 | 1.0 | 128.183 | 1 | 128.175 | 0.000 | — | — |
| semi-synthetic | histgb | 0.5 | 1.0 | 128.263 | 1 | 128.175 | 0.000 | — | — |
| semi-synthetic | histgb | 1.0 | 1.0 | 128.263 | 1 | 128.175 | 0.000 | — | — |
| semi-synthetic | histgb | 0.5 | 0.5 | 128.263 | 1 | 128.175 | 0.000 | — | — |
| semi-synthetic | ridge2 | 1.0 | 1.0 | 128.540 | 1 | 128.175 | 0.000 | — | — |
| semi-synthetic | ridge2 | 0.0 | 1.0 | 128.563 | 1 | 128.175 | 0.000 | — | — |
| semi-synthetic | histgb | 0.0 | 1.0 | 128.563 | 1 | 128.175 | 0.000 | — | — |
| table | ridge2 | 1.0 | 1.0 | 117.350 | 1 | 117.350 | 1.957 | 27.8 | 1.00 |
| table | histgb | 0.5 | 1.0 | 117.350 | 1 | 117.350 | 1.957 | 27.3 | 1.00 |
| table | histgb | 1.0 | 1.0 | 117.350 | 1 | 117.350 | 1.957 | 26.7 | 1.00 |
| table | ridge2 | 0.5 | 1.0 | 117.507 | 1 | 117.350 | 1.957 | 28.4 | 1.00 |
| table | histgb | 0.5 | 0.5 | 133.697 | 1 | 133.697 | 7.677 | 0.9 | 0.50 |
| table | ridge2 | 0.0 | 1.0 | 126.937 | 5 | 117.350 | 1.957 | 27.6 | 1.00 |
| table | histgb | 0.0 | 1.0 | 126.937 | 5 | 117.350 | 1.957 | 27.6 | 1.00 |
