"""Demo 03: see how actions change inventory, cash, NAV and reward."""
from shared.contract import ACTION_NAMES
from shared.practice_market import PracticeMarketEnv

# A short volatile market makes the consequences of positions visible.
env = PracticeMarketEnv(scenario="volatile", steps=12, seed=8)
obs = env.reset()

# Deliberately fixed sequence for demonstration, NOT a good trading strategy.
actions = [1, 1, 0, 0, 2, 0, 5, 0, 6, 0, 0, 0]

print("action                 price      inv       cash          NAV      reward")
print("-" * 79)
for action in actions:
    obs, reward, done, info = env.step(action)
    print(
        f"{ACTION_NAMES[action]:20s} "
        f"${info['mid_cents']/100:7.2f} "
        f"{info['inventory']:7d} "
        f"${info['cash_cents']/100:10,.2f} "
        f"${info['nav_cents']/100:11,.2f} "
        f"{reward:9.4f}"
    )
    if done:
        break
