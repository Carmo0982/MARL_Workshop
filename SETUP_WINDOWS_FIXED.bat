@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher not found. Install Python 3.12 or newer, then run this again.
  pause
  exit /b 1
)
if not exist .venv (
  py -3.12 -m venv .venv 2>nul || py -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements_fixed.txt
if errorlevel 1 (
  echo Installation failed. Check the error above.
  pause
  exit /b 1
)
echo.
echo Setup complete.
echo Next: double-click START_HERE.bat
pause
