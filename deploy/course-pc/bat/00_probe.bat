@echo off
setlocal
call "%~dp0_env.bat" || exit /b 1
echo [00] Recording this computer's environment (no solver call)...
"%PY%" "%REPO%\deploy\course-pc\probe_env.py" --out "%RESULTS%"
echo.
echo [00] Done. Send the probe-*.json in %RESULTS% back (04_collect.bat zips everything).
endlocal
pause
