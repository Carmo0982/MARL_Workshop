"""Demo 02: a transparent rule-based trader using the SAME 9-action contract."""
from shared.practice_market import PracticeMarketEnv


def choose_action(obs):
    # obs[1] is normalised 5-step return.
    momentum_5 = float(obs[1])
    inventory = float(obs[8])

    # Small market orders only, to keep the example easy to inspect.
    if momentum_5 > 0.015 and inventory < 0.8:
        return 1  # BUY_MARKET_SMALL
    if momentum_5 < -0.015 and inventory > -0.8:
        return 2  # SELL_MARKET_SMALL
    return 0      # HOLD


def run(scenario):
    env = PracticeMarketEnv(scenario=scenario, steps=150, seed=19)
    obs = env.reset()
    done = False
    while not done:
        action = choose_action(obs)
        obs, reward, done, info = env.step(action)

    pnl = (info["nav_cents"] - 10_000_000) / 100
    print(
        f"{scenario:9s}  final NAV=${info['nav_cents']/100:,.2f}  "
        f"PnL=${pnl:,.2f}  trades={info['trade_count']}  "
        f"inventory={info['inventory']}"
    )


for scenario_name in ("stable", "trending", "volatile"):
    run(scenario_name)
