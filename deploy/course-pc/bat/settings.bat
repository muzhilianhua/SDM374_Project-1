@rem Settings for 03_run_online.bat. Edit with Notepad, save, then run 03_run_online.bat.
@rem
@rem SOLVER_EXE: full path to the course solver.exe. Leave empty to use repo\solver\solver.exe.
@rem   Its folder keeps usage.count (the solver's own 20-call quota); the kit never touches that file.
set "SOLVER_EXE="
@rem ONLINE_CONFIG: config name in repo\experiments, without .toml.
@rem   online-gp-wide  = additive ARD GP + pure-exploit pool (temporary T18 policy, about 3 min to refit per call)
@rem   online-baseline = ridge2 + trust-dirichlet (fast working default)
set "ONLINE_CONFIG=online-gp-wide"
@rem MAX_CALLS: stop after this many successful calls in this session; empty = run to 20.
set "MAX_CALLS="
