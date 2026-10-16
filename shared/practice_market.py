from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from shared.contract import (
    STARTING_CASH_CENTS, POSITION_LIMIT, ACTION_SIZE,
    build_observation, action_to_order,
)

PRACTICE_SCENARIOS = {
    "stable": dict(drift=0.0, volatility=0.0007, spread_bps=4.0, depth=900, shock_prob=0.0, shock_scale=0.0),
    "trending": dict(drift=0.00035, volatility=0.0012, spread_bps=6.0, depth=750, shock_prob=0.0, shock_scale=0.0),
    "volatile": dict(drift=0.0, volatility=0.0040, spread_bps=15.0, depth=500, shock_prob=0.02, shock_scale=0.03),
}

class PracticeMarketEnv:
    """Fast training sandbox that implements the official observation/action contract.

    This is intentionally lightweight.  The official assessment market is ABIDES-NG.
    """
    def __init__(self, scenario="stable", steps=300, seed=0):
        if scenario not in PRACTICE_SCENARIOS:
            raise ValueError(f"Unknown scenario {scenario}. Choose {list(PRACTICE_SCENARIOS)}")
        self.scenario = scenario
        self.steps = int(steps)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.cfg = PRACTICE_SCENARIOS[scenario]
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.seed)
        self.t = 0
        self.mid = 10_000.0
        self.history = [self.mid]
        self.cash = STARTING_CASH_CENTS
        self.inventory = 0
        self.prev_nav = float(self.cash)
        self.trade_count = 0
        return self._obs()

    def _book(self):
        spread = max(1.0, self.mid * self.cfg["spread_bps"] / 10_000.0)
        bid = int(round(self.mid - spread / 2.0))
        ask = int(round(self.mid + spread / 2.0))
        if ask <= bid:
            ask = bid + 1
        noise = int(self.rng.integers(-100, 101))
        bid_depth = max(50, self.cfg["depth"] + noise)
        ask_depth = max(50, self.cfg["depth"] - noise)
        return bid, ask, bid_depth, ask_depth

    def _obs(self):
        bid, ask, bd, ad = self._book()
        return build_observation(
            self.history, bid, ask, bd, ad, self.inventory, self.cash,
            time_progress=self.t / max(1, self.steps), open_orders=0,
        )

    def step(self, action: int):
        if not 0 <= int(action) < ACTION_SIZE:
            raise ValueError("Action must be an integer 0..8")
        bid, ask, bd, ad = self._book()
        order = action_to_order(action, bid, ask)

        # Immediate execution model for workshop training. Limit orders have a
        # moderate probability of filling; market orders always cross the spread.
        filled = False
        qty = int(order["quantity"])
        if order["kind"] == "MARKET":
            filled = qty > 0
            price = ask if order["side"] == "BUY" else bid
        elif order["kind"] == "LIMIT":
            filled = self.rng.random() < 0.55
            price = int(order["limit_price"])
        else:
            price = int(round(self.mid))

        if filled:
            signed = qty if order["side"] == "BUY" else -qty
            proposed = self.inventory + signed
            if abs(proposed) <= POSITION_LIMIT:
                cost = signed * price
                # Allow shorting but do not allow long purchases beyond cash.
                if signed < 0 or self.cash >= cost:
                    self.inventory = proposed
                    self.cash -= cost
                    self.trade_count += 1

        shock = 0.0
        if self.rng.random() < self.cfg["shock_prob"]:
            shock = self.rng.choice([-1.0, 1.0]) * self.cfg["shock_scale"]
        ret = self.cfg["drift"] + self.rng.normal(0.0, self.cfg["volatility"]) + shock
        self.mid = max(100.0, self.mid * math.exp(ret))
        self.history.append(self.mid)
        self.t += 1

        nav = self.cash + self.inventory * self.mid
        pnl_change = nav - self.prev_nav
        self.prev_nav = nav
        # Dense reward: change in NAV, with a small inventory-risk penalty.
        reward = pnl_change / 1000.0 - 0.0005 * abs(self.inventory)
        done = self.t >= self.steps
        info = {
            "nav_cents": float(nav), "cash_cents": int(self.cash),
            "inventory": int(self.inventory), "mid_cents": float(self.mid),
            "trade_count": int(self.trade_count), "scenario": self.scenario,
        }
        return self._obs(), float(reward), done, info
