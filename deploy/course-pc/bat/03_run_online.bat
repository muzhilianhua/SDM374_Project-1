@echo off
setlocal
call "%~dp0_env.bat" || exit /b 1
call "%~dp0settings.bat"
pushd "%REPO%"
"%PY%" "%REPO%\deploy\course-pc\with_solver.py" "%ONLINE_CONFIG%" "%SOLVER_EXE%" > "%RESULTS%\03_config_path.txt"
if errorlevel 1 (
  type "%RESULTS%\03_config_path.txt"
  popd
  pause
  exit /b 1
)
set /p CONFIG=<"%RESULTS%\03_config_path.txt"
set "LIMIT="
if not "%MAX_CALLS%"=="" set "LIMIT=--max-calls %MAX_CALLS%"
echo.
echo ================================================================
echo  REAL ONLINE RUN: calls solver.exe. Every call uses one of the
echo  solver's 20 calls (usage.count), even if it fails.
echo  config : %CONFIG%
echo  limit  : %LIMIT%
echo ================================================================
if not "%SDM_NO_CONFIRM%"=="1" (
  set /p ANSWER=Type YES to start: 
)
if not "%SDM_NO_CONFIRM%"=="1" if /i not "%ANSWER%"=="YES" (
  echo Cancelled.
  popd
  pause
  exit /b 1
)
"%PY%" -m sdm --runs-dir "%RESULTS%\runs" online -c "%CONFIG%" --name %ONLINE_CONFIG% %LIMIT%
echo [03] exit code %ERRORLEVEL%. Run folder: %RESULTS%\runs. To continue an interrupted run: 05_resume.bat
popd
endlocal
pause
