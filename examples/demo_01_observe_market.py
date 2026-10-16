"""Demo 01: inspect one market observation and take a few HOLD actions."""
from shared.contract import OBSERVATION_NAMES, ACTION_NAMES
from shared.practice_market import PracticeMarketEnv


env = PracticeMarketEnv(scenario="stable", steps=10, seed=42)
obs = env.reset()

print("\n16-DIMENSION MARKET OBSERVATION")
print("-" * 46)
for name, value in zip(OBSERVATION_NAMES, obs):
    print(f"{name:20s} {value: .4f}")

print("\nTake action 0 =", ACTION_NAMES[0])
for step in range(3):
    obs, reward, done, info = env.step(0)  # 0 = HOLD
    print(
        f"step={step+1}  price=${info['mid_cents']/100:7.2f}  "
        f"inventory={info['inventory']:4d}  reward={reward: .4f}  "
        f"NAV=${info['nav_cents']/100:,.2f}"
    )
