# h7 on seeds 2026 and 11 (report Table II)

These are the pending runs from `report/pending-experiments.md`. The submitted policy h7 (package `submission/` on
`origin/feat/submission-v7s7`, commit f4333d7) was run with only `"seed"` changed in `config.json`. Each seed
used its own development copy of `solver.exe` (sha256 `293867b8…7247`, the pinned one) with a fresh 20-call quota.
Both seeds ran in parallel on one machine (`machine.txt`: i5-14600KF, 20 logical CPUs, Windows 11 Pro 10.0.26200),
and `check` passed for both (`h7_s<seed>/check_output.txt`: Haswell BLAS, 12 threads).

## Results

Best R after 20 calls. The h7 seed 7 and v4 numbers are the existing ones, and the v4 runs are in
`analysis/online_rehearsal/rehearsals/v4_s<seed>/`.

| seed | h7 | v4 | h7 − v4 |
|---|---|---|---|
| 7 | 102.806 | 103.275 | −0.469 |
| 2026 | 107.812467 | 108.048844 | −0.236 |
| 11 | 105.865767 | 105.440451 | +0.425 |
| mean | 105.495 | 105.588 | −0.093 |

For calls 1–11, h7 and v4 send effectively the same batches on both seeds, and every point gets the same R. So
the paired comparison holds even though v4 ran on another machine with the T37 scripts. The only differences
are on seed 11:

- call 6: 4 weight vectors differ by at most 2.2e-16 (last-bit float noise);
- call 10: two candidates with identical R swap places in the batch (a near-tie in their predicted values).

The two policies diverge from call 12, which is where h7's changes take effect. On seed 2026, h7 leads from
call 13 to the end. On seed 11, h7 leads during calls 12–18, but v4 then drops to 105.654 and 105.440 in calls
19–20, and h7 ends 0.425 behind.

`python analysis/online_v7_seeds/compare_v4.py` prints the per-call comparison.

## Files per seed (`h7_s<seed>/`)

- `online_log.jsonl`, `checkpoints/`: the run, exactly as `run.py` wrote it;
- `config.json`: the package config with the seed changed;
- `check_output.txt`: output of `run.py check`;
- `console_output.txt`: console of the completed run (20 `[k/20]` lines);
- `console_output_attempt1_console_bug.txt` and `checkpoints/call_01.rejected.<time>.json`: the first,
  failed attempt (see below).

## First attempt failed: solver.exe under an attached console

The first attempt was launched from Claude Code's PowerShell tool. On both seeds call 1 came back with exit 2
and the message `Expecting value: line 1 column 1 (char 0)`. Tests on a separate development copy then showed
the following:

- From that host, every *valid* batch fails this way. Invalid input still gets the normal validation message,
  so the package's stdin check passes.
- Starting the process with `CREATE_NO_WINDOW` or `CREATE_NEW_CONSOLE`, or from a Bash shell, works.
- Environment variables, a UTF-8 BOM and `PYTHONIOENCODING` were ruled out.
- Two solver instances in parallel are fine (two 50-set batches together took about 330 s, against 301 s alone).
- These failures **do spend solver quota** (`usage.count` advances), but `run.py` records them as "rejected,
  no call spent".

Both development copies' `usage.count` were deleted, with the coordinator's approval, to restore the full
20 calls. Both seeds were then rerun from Bash with `--after-rejection`. The policy is deterministic, so the
rerun matches a first-time run, and calls 1–11 match v4 as shown above.
