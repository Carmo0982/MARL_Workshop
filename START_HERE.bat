@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Environment not found. Run SETUP_ONCE.bat first.
  pause
  exit /b 1
)
set PY=.venv\Scripts\python.exe
:menu
cls
echo ======================================================
echo MARL STOCK EXCHANGE - STUDENT WORKSHOP
echo ======================================================
echo MINI EXAMPLES - see the mechanics first
echo 1. Demo: inspect the 16-value market observation
echo 2. Demo: run a small rule-based trader
echo 3. Demo: see actions, cash, inventory, NAV and reward
echo 4. Demo: pass an observation through the neural network
echo 5. Demo: perform one real DQN learning update
echo.
echo PROJECT PIPELINE
echo 6. Check ABIDES-NG with a stable market
echo 7. Train starter DQN (5,000 steps)
echo 8. Test trained model on all practice scenarios
echo 9. Export final TEAM ZIP
echo V. Validate an existing TEAM ZIP
echo 0. Exit
echo.
set /p CH=Choose: 
if "%CH%"=="1" %PY% -m examples.demo_01_observe_market & pause & goto menu
if "%CH%"=="2" %PY% -m examples.demo_02_rule_based_agent & pause & goto menu
if "%CH%"=="3" %PY% -m examples.demo_03_actions_and_reward & pause & goto menu
if "%CH%"=="4" %PY% -m examples.demo_04_network_forward & pause & goto menu
if "%CH%"=="5" %PY% -m examples.demo_05_one_dqn_update & pause & goto menu
if "%CH%"=="6" %PY% -m shared.abides_smoke_test --scenario stable_day --seed 42 & pause & goto menu
if "%CH%"=="7" %PY% -m student_work.train --steps 5000 --seed 7 & pause & goto menu
if "%CH%"=="8" %PY% -m shared.test_policy --model outputs\model.pt --scenario all --seed 101 & pause & goto menu
if /I "%CH%"=="9" goto export
if /I "%CH%"=="V" goto validate
if "%CH%"=="0" exit /b 0
goto menu
:export
set /p TEAM=Team ID (example TEAM_05): 
set /p TNAME=Team name: 
set /p MEMBERS=Members separated by commas: 
set /p ALG=Training algorithm (example DQN): 
%PY% -m shared.export_submission --team-id "%TEAM%" --team-name "%TNAME%" --members "%MEMBERS%" --algorithm "%ALG%" --model outputs\model.pt
pause
goto menu
:validate
set /p ZFILE=ZIP path: 
%PY% -m shared.validate_submission "%ZFILE%"
pause
goto menu
