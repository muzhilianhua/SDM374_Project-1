@echo off
setlocal
call "%~dp0_env.bat" || exit /b 1
pushd "%REPO%"
echo [02] 1/4 test suite (a few minutes)...
"%PY%" -m pytest -q -p no:cacheprovider > "%RESULTS%\02_pytest.log" 2>&1
echo      exit code %ERRORLEVEL%
"%PY%" -c "import sys; lines=open(sys.argv[1], encoding='utf-8', errors='replace').read().splitlines(); print('     ', lines[-1] if lines else 'no output')" "%RESULTS%\02_pytest.log"
echo [02] 2/4 online loop against the semi-synthetic truth, 3 calls (no solver)...
"%PY%" -m sdm --runs-dir "%RESULTS%\runs" online -c experiments\online-baseline-synthetic.toml --max-calls 3 > "%RESULTS%\02_synthetic.log" 2>&1
echo      exit code %ERRORLEVEL%
echo [02] 3/4 first-batch dry runs of the solver configs (proposes and screens, calls nothing)...
"%PY%" -m sdm --runs-dir "%RESULTS%\runs" online -c experiments\online-baseline.toml --dry-run > "%RESULTS%\02_dryrun_baseline.log" 2>&1
echo      online-baseline exit code %ERRORLEVEL%
"%PY%" -m sdm --runs-dir "%RESULTS%\runs" online -c experiments\online-gp-wide.toml --dry-run > "%RESULTS%\02_dryrun_gp_wide.log" 2>&1
echo      online-gp-wide exit code %ERRORLEVEL%
echo [02] 4/4 surrogate fit timing and prediction fingerprints...
"%PY%" "%REPO%\deploy\course-pc\bench_models.py" --out "%RESULTS%" > "%RESULTS%\02_bench.log" 2>&1
echo      exit code %ERRORLEVEL%
type "%RESULTS%\02_bench.log"
popd
echo [02] Done. Logs are in %RESULTS%. Next: edit settings.bat, then 03_run_online.bat
endlocal
pause
