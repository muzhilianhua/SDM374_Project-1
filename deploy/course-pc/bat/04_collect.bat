@echo off
setlocal
call "%~dp0_env.bat" || exit /b 1
echo [04] Zipping %RESULTS% ...
"%PY%" -c "import os, shutil, sys, time; base = os.path.join(sys.argv[1], 'results-%%s-%%s' %% (os.environ.get('COMPUTERNAME', 'pc'), time.strftime('%%Y%%m%%d-%%H%%M%%S'))); print(shutil.make_archive(base, 'zip', sys.argv[2]))" "%KIT%" "%RESULTS%"
echo [04] Copy the zip above back to your own computer.
endlocal
pause
