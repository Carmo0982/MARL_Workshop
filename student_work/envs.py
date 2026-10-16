"""Richer training/evaluation markets that keep the official 16/9 contract.

Why this file exists
--------------------
``shared/practice_market.py`` is fast but differs from ABIDES-NG in ways that
encourage overfitting:

* only three regimes, and the only trend is upward;
* limit orders fill instantly 55% of the time at a better price than market
  orders, so ``open_orders`` (observation 12) is always 0;
* no low-liquidity / crisis regimes, no market impact;
* fixed price level ($100) and fixed episode length.

``RobustMarketEnv`` keeps the *same* observation builder and action mapping
(imported from ``shared.contract``, which we never modify) but adds:

* more regimes (down-trend, mean-reverting, low liquidity, stress/crash);
* optional parameter jitter, random price level and random episode length;
* resting limit orders with a time-to-live, so ``open_orders`` is non-zero;
* market impact for large orders in thin books;
* a configurable reward (``RewardConfig``).

The ``step`` signature and ``info`` keys match ``PracticeMarketEnv`` so the
same training / evaluation code works with both.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from shared.contract import (
    ACTION_SIZE,
    POSITION_LIMIT,
    STARTING_CASH_CENTS,
    action_to_order,
    build_observation,
)

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """Per-step market dynamics. Returns are log-returns per decision step."""

    name: str
    drift: float = 0.0
    volatility: float = 0.001
    spread_bps: float = 5.0
    depth: int = 800
    shock_prob: float = 0.0
    shock_scale: float = 0.0
    shock_down_bias: float = 0.5   # probability that a shock is negative
    mean_reversion: float = 0.0    # pull of the mid towards a slow "fundamental"
    fundamental_vol: float = 0.0   # random walk of the fundamental
    momentum_flow: float = 0.0     # other trend-followers: ret += k * prev_ret
    limit_fill_prob: float = 0.30  # per-step fill probability of a resting order at the touch
    impact_bps: float = 2.0        # market-order price impact for qty == depth
    imbalance_noise: float = 0.125  # bid/ask depth noise as a fraction of depth (ABIDES ~0.7)


SCENARIOS: dict[str, Scenario] = {
    # Same three regimes as the practice market (so results are comparable).
    "stable": Scenario("stable", volatility=0.0007, spread_bps=4.0, depth=900, limit_fill_prob=0.35),
    "trend_up": Scenario("trend_up", drift=0.00035, volatility=0.0012, spread_bps=6.0, depth=750),
    "volatile": Scenario("volatile", volatility=0.0040, spread_bps=15.0, depth=500,
                         shock_prob=0.02, shock_scale=0.03, limit_fill_prob=0.35),
    # Regimes missing from the practice market but present in ABIDES / the brief.
    "trend_down": Scenario("trend_down", drift=-0.00035, volatility=0.0012, spread_bps=6.0, depth=750),
    "mean_revert": Scenario("mean_revert", volatility=0.0015, spread_bps=8.0, depth=700,
                            mean_reversion=0.05, fundamental_vol=0.0003),
    "low_liquidity": Scenario("low_liquidity", volatility=0.0015, spread_bps=40.0, depth=120,
                              limit_fill_prob=0.12, impact_bps=25.0, mean_reversion=0.02,
                              fundamental_vol=0.0003),
    # Crisis: roughly -10% to -20% over a day, with mostly-negative jumps.
    "stress": Scenario("stress", drift=-0.0002, volatility=0.0030, spread_bps=25.0, depth=300,
                       shock_prob=0.02, shock_scale=0.03, shock_down_bias=0.75,
                       momentum_flow=0.10, limit_fill_prob=0.20, impact_bps=12.0),
    "crowded": Scenario("crowded", drift=0.0, volatility=0.0018, spread_bps=10.0, depth=400,
                        momentum_flow=0.30, impact_bps=8.0),
    # ABIDES-calibrated regimes. Parameters chosen so the observation statistics
    # (std of returns, mean spread / depth, std of book imbalance) match what our
    # PolicyAgent observes in ABIDES-NG with a 60 s decision interval
    # (measured with student_work/abides_eval.py, see NOTAS_EQUIPO.md section 4).
    "abides_calm": Scenario("abides_calm", volatility=0.00018, spread_bps=0.6, depth=480,
                            mean_reversion=0.03, fundamental_vol=0.00005,
                            limit_fill_prob=0.30, impact_bps=1.0, imbalance_noise=0.65),
    "abides_trending": Scenario("abides_trending", volatility=0.00035, spread_bps=0.9, depth=440,
                                mean_reversion=0.01, fundamental_vol=0.0002, momentum_flow=0.10,
                                limit_fill_prob=0.30, impact_bps=1.5, imbalance_noise=0.75),
    "abides_volatile": Scenario("abides_volatile", volatility=0.0009, spread_bps=1.6, depth=470,
                                mean_reversion=0.03, fundamental_vol=0.0004,
                                shock_prob=1 / 360, shock_scale=0.03,
                                limit_fill_prob=0.30, impact_bps=2.0, imbalance_noise=0.6),
    "abides_stress": Scenario("abides_stress", volatility=0.014, spread_bps=70.0, depth=350,
                              mean_reversion=0.22, fundamental_vol=0.002,
                              shock_prob=1 / 120, shock_scale=0.05, shock_down_bias=0.6,
                              limit_fill_prob=0.15, impact_bps=20.0, imbalance_noise=0.75),
}

# Default splits. Training uses ALL scenarios; validation and test use seeds
# that are never used for training, so model selection is not done on data
# the network has seen.
TRAIN_SCENARIOS = list(SCENARIOS)
EVAL_SCENARIOS = list(SCENARIOS)
VALIDATION_SEEDS = [1001, 1002, 1003]
TEST_SEEDS = [5001, 5002, 5003, 5004, 5005]   # use ONLY for the final report


def jitter_scenario(sc: Scenario, rng: np.random.Generator, amount: float = 0.3) -> Scenario:
    """Randomly scale the scenario parameters by up to +/- ``amount``."""

    def j(x: float) -> float:
        return float(x * rng.uniform(1.0 - amount, 1.0 + amount))

    return replace(
        sc,
        drift=j(sc.drift),
        volatility=j(sc.volatility),
        spread_bps=j(sc.spread_bps),
        depth=max(20, int(j(sc.depth))),
        shock_prob=j(sc.shock_prob),
        shock_scale=j(sc.shock_scale),
        mean_reversion=j(sc.mean_reversion),
        momentum_flow=j(sc.momentum_flow),
        limit_fill_prob=min(0.9, j(sc.limit_fill_prob)),
        impact_bps=j(sc.impact_bps),
    )


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RewardConfig:
    """Reward = scaled NAV change minus risk / activity penalties.

    The starter reward is ``pnl/1000 - 0.0005*|inv|``: ``RewardConfig.starter()``.
    Every penalty is expressed in the same unit as the PnL term (1 unit = $10)
    so their magnitudes are easy to compare.
    """

    pnl_scale: float = 1.0 / 1000.0
    inventory_penalty: float = 0.0005   # per share held, per step
    drawdown_penalty: float = 0.0       # per $10 lost in the step: the drawdown-increase formula below
                                        # reduces to max(0, -pnl_change), i.e. every step loss is penalised
    trade_penalty: float = 0.0          # per filled order (turnover cost)
    terminal_inventory_penalty: float = 0.0  # per share still held at the end
    risk_free_hold_bonus: float = 0.0   # tiny bonus for HOLD when flat (anti-churn)

    @staticmethod
    def starter() -> "RewardConfig":
        return RewardConfig()

    @staticmethod
    def risk_aware() -> "RewardConfig":
        """Default for the improved trainer (see NOTAS_EQUIPO.md for the rationale)."""
        return RewardConfig(
            inventory_penalty=0.001,
            drawdown_penalty=0.5,
            trade_penalty=0.02,
            terminal_inventory_penalty=0.01,
        )


REWARD_PRESETS = {
    "starter": RewardConfig.starter(),
    "risk_aware": RewardConfig.risk_aware(),
    "pnl_only": RewardConfig(inventory_penalty=0.0),
    "strong_risk": RewardConfig(inventory_penalty=0.003, drawdown_penalty=1.0,
                                trade_penalty=0.05, terminal_inventory_penalty=0.02),
}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


@dataclass
class _RestingOrder:
    side: str          # "BUY" / "SELL"
    qty: int
    price: int
    ttl: int


@dataclass
class EnvConfig:
    scenario: str = "stable"
    steps: int = 300
    seed: int = 0
    randomize: bool = False            # jitter params, random price and length
    steps_range: tuple[int, int] = (200, 400)
    price_range: tuple[float, float] = (5_000.0, 150_000.0)  # cents: $50 .. $1500
    start_price: float = 10_000.0      # cents, used when randomize=False
    limit_model: str = "resting"       # "resting" (ABIDES-like) or "instant" (practice-like)
    limit_ttl: int = 5                 # steps a resting order lives before cancellation
    max_open_orders: int = 10
    reward: RewardConfig = field(default_factory=RewardConfig.starter)


class RobustMarketEnv:
    """Contract-compatible market with richer dynamics. See module docstring."""

    def __init__(self, scenario: str = "stable", steps: int = 300, seed: int = 0, **kwargs):
        self.cfg = EnvConfig(scenario=scenario, steps=steps, seed=seed, **kwargs)
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario {scenario}. Choose {list(SCENARIOS)}")
        self.scenario = scenario
        self.reset()

    # -- public API (same as PracticeMarketEnv) -----------------------------
    def reset(self):
        c = self.cfg
        self.rng = np.random.default_rng(c.seed)
        sc = SCENARIOS[c.scenario]
        if c.randomize:
            sc = jitter_scenario(sc, self.rng)
            lo, hi = c.price_range
            self.mid = float(math.exp(self.rng.uniform(math.log(lo), math.log(hi))))
            self.steps = int(self.rng.integers(c.steps_range[0], c.steps_range[1] + 1))
        else:
            self.mid = float(c.start_price)
            self.steps = int(c.steps)
        self.sc = sc
        self.fundamental = self.mid
        self.prev_ret = 0.0
        self.t = 0
        self.history = [self.mid]
        self.cash = STARTING_CASH_CENTS
        self.inventory = 0
        self.prev_nav = float(self.cash)
        self.peak_nav = float(self.cash)
        self.trade_count = 0
        self.orders: list[_RestingOrder] = []
        self._book_cache = self._make_book()
        return self._obs()

    def step(self, action: int):
        action = int(action)
        if not 0 <= action < ACTION_SIZE:
            raise ValueError("Action must be an integer 0..8")
        rc = self.cfg.reward
        bid, ask, bd, ad = self._book_cache
        order = action_to_order(action, bid, ask)
        fills = 0

        if order["kind"] == "MARKET":
            fills += self._market_order(order["side"], int(order["quantity"]), bid, ask, bd, ad)
        elif order["kind"] == "LIMIT":
            if self.cfg.limit_model == "instant":
                if self.rng.random() < 0.55:
                    fills += self._fill(order["side"], int(order["quantity"]), int(order["limit_price"]))
            elif len(self.orders) < self.cfg.max_open_orders:
                self.orders.append(_RestingOrder(order["side"], int(order["quantity"]),
                                                 int(order["limit_price"]), self.cfg.limit_ttl))

        # --- price dynamics ---------------------------------------------
        sc = self.sc
        shock = 0.0
        if self.rng.random() < sc.shock_prob:
            sign = -1.0 if self.rng.random() < sc.shock_down_bias else 1.0
            shock = sign * sc.shock_scale
        self.fundamental *= math.exp(self.rng.normal(0.0, sc.fundamental_vol)) if sc.fundamental_vol else 1.0
        pull = sc.mean_reversion * math.log(self.fundamental / self.mid) if sc.mean_reversion else 0.0
        ret = (sc.drift + pull + sc.momentum_flow * self.prev_ret
               + self.rng.normal(0.0, sc.volatility) + shock)
        self.mid = max(100.0, self.mid * math.exp(ret))
        self.prev_ret = ret
        self.history.append(self.mid)
        self.t += 1
        self._book_cache = self._make_book()

        # --- resting limit orders (filled against the NEW book) ------------
        if self.orders:
            fills += self._process_resting_orders()

        # --- accounting and reward -----------------------------------------
        nav = self.cash + self.inventory * self.mid
        pnl_change = nav - self.prev_nav
        self.prev_nav = nav
        old_dd = self.peak_nav - (nav - pnl_change)
        self.peak_nav = max(self.peak_nav, nav)
        new_dd = self.peak_nav - nav
        dd_increase = max(0.0, new_dd - max(0.0, old_dd))

        reward = (pnl_change * rc.pnl_scale
                  - rc.inventory_penalty * abs(self.inventory)
                  - rc.drawdown_penalty * dd_increase * rc.pnl_scale
                  - rc.trade_penalty * fills)
        if action == 0 and self.inventory == 0:
            reward += rc.risk_free_hold_bonus
        done = self.t >= self.steps
        if done:
            reward -= rc.terminal_inventory_penalty * abs(self.inventory)

        info = {
            "nav_cents": float(nav), "cash_cents": int(self.cash),
            "inventory": int(self.inventory), "mid_cents": float(self.mid),
            "trade_count": int(self.trade_count), "scenario": self.scenario,
            "open_orders": len(self.orders), "fills": fills,
        }
        return self._obs(), float(reward), done, info

    # -- internals ----------------------------------------------------------
    def _make_book(self):
        sc = self.sc
        spread = max(1.0, self.mid * sc.spread_bps / 10_000.0)
        # Spread widens after big moves (liquidity providers step back).
        spread *= 1.0 + min(3.0, abs(self.prev_ret) / max(1e-6, 3 * sc.volatility))
        bid = int(round(self.mid - spread / 2.0))
        ask = int(round(self.mid + spread / 2.0))
        if ask <= bid:
            ask = bid + 1
        k = int(sc.depth * sc.imbalance_noise)
        noise = int(self.rng.integers(-k - 1, k + 2))
        bid_depth = max(10, sc.depth + noise)
        ask_depth = max(10, sc.depth - noise)
        return bid, ask, bid_depth, ask_depth

    def _obs(self):
        bid, ask, bd, ad = self._book_cache
        return build_observation(
            self.history, bid, ask, bd, ad, self.inventory, self.cash,
            time_progress=self.t / max(1, self.steps), open_orders=len(self.orders),
        )

    def _fill(self, side: str, qty: int, price: int) -> int:
        signed = qty if side == "BUY" else -qty
        proposed = self.inventory + signed
        if abs(proposed) > POSITION_LIMIT:
            return 0
        cost = signed * price
        if signed > 0 and self.cash < cost:
            return 0  # same rule as the practice market: no buying beyond cash
        self.inventory = proposed
        self.cash -= cost
        self.trade_count += 1
        return 1

    def _market_order(self, side, qty, bid, ask, bd, ad) -> int:
        sc = self.sc
        depth = ad if side == "BUY" else bd
        impact = sc.impact_bps / 10_000.0 * (qty / max(1.0, depth))
        if side == "BUY":
            price = int(math.ceil(ask * (1.0 + impact)))
        else:
            price = int(math.floor(bid * (1.0 - impact)))
        filled = self._fill(side, qty, price)
        if filled:
            # Permanent impact: half of the temporary impact moves the mid.
            self.mid *= math.exp((impact / 2.0) * (1 if side == "BUY" else -1))
        return filled

    def _process_resting_orders(self) -> int:
        bid, ask, _, _ = self._book_cache
        fills = 0
        keep: list[_RestingOrder] = []
        for o in self.orders:
            crossed = (o.side == "BUY" and ask <= o.price) or (o.side == "SELL" and bid >= o.price)
            if crossed or self.rng.random() < self.sc.limit_fill_prob:
                fills += self._fill(o.side, o.qty, o.price)
                continue
            o.ttl -= 1
            if o.ttl > 0:
                keep.append(o)
        self.orders = keep
        return fills


# ---------------------------------------------------------------------------
# Helpers used by the trainer
# ---------------------------------------------------------------------------


class ScenarioSampler:
    """Produces a fresh randomized environment for every training episode."""

    def __init__(self, scenarios=None, weights=None, randomize=True,
                 reward: RewardConfig | None = None, limit_model="resting",
                 include_practice: float = 0.15, seed: int = 0):
        self.scenarios = list(scenarios or TRAIN_SCENARIOS)
        w = np.asarray(weights or [1.0] * len(self.scenarios), dtype=float)
        self.weights = w / w.sum()
        self.randomize = randomize
        self.reward = reward or RewardConfig.risk_aware()
        self.limit_model = limit_model
        self.include_practice = include_practice
        self.rng = np.random.default_rng(seed)

    def sample(self, episode_seed: int):
        """Return (env, scenario_name). A fraction of episodes use the official practice market."""
        if self.include_practice and self.rng.random() < self.include_practice:
            from shared.practice_market import PracticeMarketEnv
            name = str(self.rng.choice(["stable", "trending", "volatile"]))
            env = PracticeMarketEnv(scenario=name, steps=int(self.rng.integers(200, 401)), seed=episode_seed)
            return RewardWrapper(env, self.reward), f"practice_{name}"
        name = str(self.rng.choice(self.scenarios, p=self.weights))
        env = RobustMarketEnv(name, seed=episode_seed, randomize=self.randomize,
                              limit_model=self.limit_model, reward=self.reward)
        return env, name


class RewardWrapper:
    """Recomputes the reward of any contract env (e.g. PracticeMarketEnv) with a RewardConfig."""

    def __init__(self, env, reward: RewardConfig):
        self.env = env
        self.rc = reward
        self.scenario = getattr(env, "scenario", "?")

    def reset(self):
        obs = self.env.reset()
        self.prev_nav = float(STARTING_CASH_CENTS)
        self.peak_nav = float(STARTING_CASH_CENTS)
        self.prev_trades = 0
        return obs

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        rc = self.rc
        nav = info["nav_cents"]
        pnl_change = nav - self.prev_nav
        old_dd = self.peak_nav - self.prev_nav
        self.prev_nav = nav
        self.peak_nav = max(self.peak_nav, nav)
        dd_increase = max(0.0, (self.peak_nav - nav) - max(0.0, old_dd))
        fills = info["trade_count"] - self.prev_trades
        self.prev_trades = info["trade_count"]
        inv = info["inventory"]
        reward = (pnl_change * rc.pnl_scale - rc.inventory_penalty * abs(inv)
                  - rc.drawdown_penalty * dd_increase * rc.pnl_scale - rc.trade_penalty * fills)
        if action == 0 and inv == 0:
            reward += rc.risk_free_hold_bonus
        if done:
            reward -= rc.terminal_inventory_penalty * abs(inv)
        return obs, float(reward), done, info
