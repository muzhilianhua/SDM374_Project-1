@rem Shared environment for the kit's steps. Called by every NN_*.bat; not run directly.
chcp 65001 >nul
set "KIT=%~dp0"
set "KIT=%KIT:~0,-1%"
set "PY=%KIT%\python\python.exe"
set "REPO=%KIT%\repo"
set "RESULTS=%KIT%\results"
set "PYTHONPATH=%REPO%\src"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONDONTWRITEBYTECODE=1"
if not exist "%RESULTS%" mkdir "%RESULTS%"
if not exist "%PY%" (
  echo [kit] bundled Python not found at %PY%. Unzip the whole kit first.
  exit /b 1
)
exit /b 0
