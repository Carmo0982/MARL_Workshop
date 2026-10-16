#!/usr/bin/env bash
# macOS / Linux equivalent of START_HERE.bat, plus our team tools.  Usage:  bash run.sh
cd "$(dirname "$0")"
PY=.venv/bin/python
if [ ! -x "$PY" ]; then echo "Environment not found. Run: bash setup_mac.sh"; exit 1; fi
export PYTHONWARNINGS=ignore

pause() { read -r -p "Press Enter to continue..." _; }

while true; do
  clear
  cat <<'MENU'
======================================================
MARL STOCK EXCHANGE - TEAM MENU (macOS)
======================================================
MINI EXAMPLES
 1. Demo: inspect the 16-value market observation
 2. Demo: run a small rule-based trader
 3. Demo: see actions, cash, inventory, NAV and reward
 4. Demo: pass an observation through the neural network
 5. Demo: perform one real DQN learning update

ORIGINAL PIPELINE
 6. Check ABIDES-NG with a stable market
 7. Quick train (5,000 steps) -> outputs/candidates/quick_5k.pt
 8. Test outputs/model.pt on the 3 practice scenarios
 9. Export final TEAM ZIP
 V. Validate an existing TEAM ZIP

TEAM TOOLS
 T. Full training run (200,000 steps, ~10-15 min)
 E. Evaluate outputs/model.pt vs baselines (all scenarios, validation seeds)
 A. Evaluate outputs/model.pt inside ABIDES (individual, 3 scenarios)
 M. ABIDES tournament: final model vs starter DQN vs baselines
 0. Exit
MENU
  read -r -p "Choose: " CH
  case "$CH" in
    1) $PY -m examples.demo_01_observe_market; pause ;;
    2) $PY -m examples.demo_02_rule_based_agent; pause ;;
    3) $PY -m examples.demo_03_actions_and_reward; pause ;;
    4) $PY -m examples.demo_04_network_forward; pause ;;
    5) $PY -m examples.demo_05_one_dqn_update; pause ;;
    6) $PY -m shared.abides_smoke_test --scenario stable_day --seed 42; pause ;;
    7) $PY -m student_work.train --steps 5000 --seed 7 --run-name quick_5k; pause ;;
    8) $PY -m shared.test_policy --model outputs/model.pt --scenario all --seed 101; pause ;;
    9)
      read -r -p "Team ID (example TEAM_05): " TEAM
      read -r -p "Team name: " TNAME
      read -r -p "Members separated by commas: " MEMBERS
      read -r -p "Training algorithm (example Double Dueling DQN): " ALG
      $PY -m shared.export_submission --team-id "$TEAM" --team-name "$TNAME" --members "$MEMBERS" \
          --algorithm "$ALG" --model outputs/model.pt
      pause ;;
    v|V)
      read -r -p "ZIP path (e.g. submission/TEAM_05.zip): " ZFILE
      $PY -m shared.validate_submission "$ZFILE"; pause ;;
    t|T)
      read -r -p "Run name (e.g. final_s7): " RN
      read -r -p "Seed [7]: " SEED
      $PY -m student_work.train --steps 200000 --seed "${SEED:-7}" --run-name "${RN:-final}" \
          --out "outputs/candidates/${RN:-final}.pt"
      echo "Model saved in outputs/candidates/${RN:-final}.pt"
      echo "If it is the one you want to submit:  cp outputs/candidates/${RN:-final}.pt outputs/model.pt"
      pause ;;
    e|E) $PY -m student_work.evaluate --policies outputs/model.pt outputs/baseline/model.pt hold momentum mean_reversion \
           --suite full --seeds val --out outputs/eval/menu; pause ;;
    a|A) $PY -m student_work.abides_eval --policies outputs/model.pt hold \
           --scenarios stable_day volatile_day stress_test --seeds 1 --out outputs/abides_eval/menu.csv 2>&1 | grep -v " INFO "; pause ;;
    m|M) $PY -m student_work.abides_eval --mode tournament \
           --policies outputs/model.pt outputs/baseline/model.pt momentum mean_reversion \
           --scenarios stable_day volatile_day --seeds 1 --out outputs/abides_eval/tournament_menu.csv 2>&1 | grep -v " INFO "; pause ;;
    0) exit 0 ;;
  esac
done
