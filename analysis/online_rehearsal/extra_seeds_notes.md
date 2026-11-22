# Extra seeds and the v4j ablation: run notes (2026-11-22)

These are the runs requested in `report/pending-experiments-2.md`. They were run with
`bash analysis/online_rehearsal/run_extra.sh` (PAR=3, WORKERS=6) from Git Bash, at commit ad07679, on the real
`solver.exe` (sha256 `293867b8…7247`).

Machine: Intel Core i5-14600KF, 20 logical CPUs, Windows 11 Pro 10.0.26200.

All 12 runs finished, each with 20 calls, `rc=0`, and no errors, rejections or quota messages in any
`console.log`. The runs took 00:36–06:49, about 1.5 h per run with three running at once.

## Driver log

```
start v2_s1 11-22 00:36
start v4j_s2026 11-22 00:36
start v4_s1 11-22 00:36
end   v2_s1 rc=0 11-22 02:03 | final best 105.64662
start v4j_s7 11-22 02:03
end   v4j_s2026 rc=0 11-22 02:10 | final best 108.97073
start v2_s2 11-22 02:10
end   v4_s1 rc=0 11-22 02:16 | final best 106.48761
start v4_s2 11-22 02:16
end   v2_s2 rc=0 11-22 03:36 | final best 107.19714
start v4j_s11 11-22 03:36
end   v4j_s7 rc=0 11-22 03:44 | final best 106.07574
start v2_s3 11-22 03:44
end   v4_s2 rc=0 11-22 03:48 | final best 105.87898
start v4_s3 11-22 03:48
end   v2_s3 rc=0 11-22 05:13 | final best 106.44160
start v4j_s1 11-22 05:13
end   v4_s3 rc=0 11-22 05:16 | final best 107.98259
start v4j_s2 11-22 05:16
end   v4j_s11 rc=0 11-22 05:17 | final best 107.41241
start v4j_s3 11-22 05:17
end   v4j_s2 rc=0 11-22 06:45 | final best 107.61872
end   v4j_s3 rc=0 11-22 06:47 | final best 108.40339
end   v4j_s1 rc=0 11-22 06:49 | final best 108.58129
```

## Summary (best R after 20 calls, lower is better)

| variant | n | mean | sd | seeds |
|---|---|---|---|---|
| v1 | 3 | 106.887 | 0.411 | 7: 107.211, 11: 106.425, 2026: 107.026 |
| v2 | 6 | 106.472 | 0.931 | 1: 105.647, 2: 107.197, 3: 106.442, 7: 105.107, 11: 107.490, 2026: 106.951 |
| v4 | 6 | 106.186 | 1.784 | 1: 106.488, 2: 105.879, 3: 107.983, 7: 103.275, 11: 105.440, 2026: 108.049 |
| v4j | 6 | 107.844 | 1.048 | 1: 108.581, 2: 107.619, 3: 108.403, 7: 106.076, 11: 107.412, 2026: 108.971 |

The seeds are treated as unpaired. The tests are an exact two-sided permutation test on the difference in means
and a two-sided Mann–Whitney U test:

- v4 − v2: −0.287, permutation p = 0.755, Mann–Whitney U p = 1.000 (not significant);
- v4 − v4j: −1.658, permutation p = 0.078, Mann–Whitney U p = 0.093. v4j is worse than v4 on every seed and
  worse than v2 on 5 of 6 seeds, but p is above 0.05 at n = 6.

Reproducibility: `v4_s1`, `v4_s2` and `v4_s3` reproduce the earlier `v4_s{1,2,3}_t1` runs on
`analysis/online-rehearsal` exactly (the same final best R to the last digit, and the same call-1 minima).
