@echo off
setlocal
call "%~dp0_env.bat" || exit /b 1
set "RUN=%~1"
rem Latest run that has attempts and whose saved config sends batches to solver.exe (replays and dry runs are skipped).
if "%RUN%"=="" for /f "delims=" %%D in ('dir /b /ad /o-d "%RESULTS%\runs\*-online-*" 2^>nul') do (
  if not defined RUN if exist "%RESULTS%\runs\%%D\attempts.jsonl" findstr /b /r /c:"kind *= *.solver." "%RESULTS%\runs\%%D\config.toml" >nul && set "RUN=%%D"
)
if "%RUN%"=="" (
  echo [05] No solver run to continue in %RESULTS%\runs
  pause
  exit /b 1
)
echo [05] Resuming %RUN% (it calls solver.exe for the remaining calls)
pushd "%REPO%"
"%PY%" -m sdm --runs-dir "%RESULTS%\runs" online --resume "%RUN%"
echo [05] exit code %ERRORLEVEL%
popd
endlocal
pause
