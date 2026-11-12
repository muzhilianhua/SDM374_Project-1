@echo off
setlocal
call "%~dp0_env.bat" || exit /b 1
echo [01] Installing dependencies offline into the bundled Python 3.11...
"%PY%" --version
"%PY%" -m pip install --disable-pip-version-check --no-index --find-links "%KIT%\wheels" -r "%REPO%\deploy\course-pc\requirements-win.txt" > "%RESULTS%\01_setup.log" 2>&1
if errorlevel 1 (
  type "%RESULTS%\01_setup.log"
  echo [01] pip install FAILED, see %RESULTS%\01_setup.log
  pause
  exit /b 1
)
echo [01] Installed. Checking the package and the training data...
pushd "%REPO%"
"%PY%" -m sdm check-data --json > "%RESULTS%\01_check_data.json" 2>&1
set "RC=%ERRORLEVEL%"
type "%RESULTS%\01_check_data.json"
"%PY%" -m sdm list > "%RESULTS%\01_sdm_list.txt" 2>&1
popd
if not "%RC%"=="0" (
  echo [01] check-data FAILED
  pause
  exit /b 1
)
echo [01] OK. Next: 02_selftest.bat
endlocal
pause
