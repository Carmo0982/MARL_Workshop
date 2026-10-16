from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

CONTRACT_VERSION = "1.0"
OBSERVATION_SIZE = 16
ACTION_SIZE = 9
STARTING_CASH_CENTS = 10_000_000  # $100,000
POSITION_LIMIT = 500
SMALL_ORDER = 10
LARGE_ORDER = 50

ACTION_NAMES = {
    0: "HOLD",
    1: "BUY_MARKET_SMALL",
    2: "SELL_MARKET_SMALL",
    3: "BUY_MARKET_LARGE",
    4: "SELL_MARKET_LARGE",
    5: "BUY_LIMIT_SMALL",
    6: "SELL_LIMIT_SMALL",
    7: "BUY_LIMIT_LARGE",
    8: "SELL_LIMIT_LARGE",
}

OBSERVATION_NAMES = [
    "return_1",
    "return_5",
    "return_20",
    "volatility_10",
    "spread_bps",
    "book_imbalance",
    "bid_depth",
    "ask_depth",
    "inventory",
    "cash_ratio",
    "nav_return",
    "time_progress",
    "open_orders",
    "price_vs_ma5",
    "price_vs_ma20",
    "bias",
]


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def _ret(history: list[float], lag: int) -> float:
    if len(history) <= lag or history[-1-lag] <= 0:
        return 0.0
    return history[-1] / history[-1-lag] - 1.0


def _ma(history: list[float], n: int) -> float:
    if not history:
        return 1.0
    vals = history[-n:]
    return float(sum(vals) / len(vals))


def build_observation(
    mid_history: list[float],
    bid_cents: int,
    ask_cents: int,
    bid_depth: int,
    ask_depth: int,
    inventory: int,
    cash_cents: int,
    starting_cash_cents: int = STARTING_CASH_CENTS,
    position_limit: int = POSITION_LIMIT,
    time_progress: float = 0.0,
    open_orders: int = 0,
) -> np.ndarray:
    """Build the official 16-dimensional observation vector.

    The same function is used by the student practice environment and by the
    lecturer ABIDES adapter.  Students may read this file but must not change
    it for the submitted model contract.
    """
    if not mid_history:
        mid = (bid_cents + ask_cents) / 2.0
        mid_history = [mid]
    mid = float(mid_history[-1])
    if mid <= 0:
        mid = max(1.0, (bid_cents + ask_cents) / 2.0)

    # Returns: 5% move maps to magnitude 1.0.
    r1 = _clip(_ret(mid_history, 1) * 20.0, -1.0, 1.0)
    r5 = _clip(_ret(mid_history, 5) * 20.0, -1.0, 1.0)
    r20 = _clip(_ret(mid_history, 20) * 20.0, -1.0, 1.0)

    # Short-horizon realised volatility from log returns.
    recent = mid_history[-11:]
    if len(recent) >= 3:
        lr = np.diff(np.log(np.maximum(np.asarray(recent, dtype=np.float64), 1.0)))
        vol = float(np.std(lr))
    else:
        vol = 0.0
    vol_norm = _clip(vol / 0.01, 0.0, 1.0)

    spread_bps = ((ask_cents - bid_cents) / max(mid, 1.0)) * 10_000.0
    spread_norm = _clip(spread_bps / 200.0, 0.0, 1.0)

    depth_total = max(1.0, float(bid_depth + ask_depth))
    imbalance = _clip((bid_depth - ask_depth) / depth_total, -1.0, 1.0)
    bid_depth_norm = _clip(bid_depth / 1000.0, 0.0, 1.0)
    ask_depth_norm = _clip(ask_depth / 1000.0, 0.0, 1.0)

    inventory_norm = _clip(inventory / max(1.0, float(position_limit)), -1.0, 1.0)
    cash_ratio = _clip(cash_cents / max(1.0, float(starting_cash_cents)), 0.0, 2.0) / 2.0
    nav_cents = cash_cents + inventory * mid
    nav_return = nav_cents / max(1.0, float(starting_cash_cents)) - 1.0
    nav_return_norm = _clip(nav_return * 5.0, -1.0, 1.0)

    ma5 = _ma(mid_history, 5)
    ma20 = _ma(mid_history, 20)
    pv5 = _clip((mid / max(ma5, 1.0) - 1.0) * 20.0, -1.0, 1.0)
    pv20 = _clip((mid / max(ma20, 1.0) - 1.0) * 20.0, -1.0, 1.0)

    obs = np.asarray([
        r1, r5, r20, vol_norm, spread_norm, imbalance,
        bid_depth_norm, ask_depth_norm, inventory_norm, cash_ratio,
        nav_return_norm, _clip(time_progress, 0.0, 1.0),
        _clip(open_orders / 10.0, 0.0, 1.0), pv5, pv20, 1.0,
    ], dtype=np.float32)
    if obs.shape != (OBSERVATION_SIZE,):
        raise RuntimeError(f"Observation contract broken: {obs.shape}")
    return obs


def action_to_order(action: int, bid_cents: int, ask_cents: int) -> dict:
    """Map policy action index to a standardised order instruction."""
    action = int(action)
    if action not in ACTION_NAMES:
        raise ValueError(f"Invalid action {action}; expected 0..{ACTION_SIZE-1}")
    if action == 0:
        return {"kind": "HOLD", "side": None, "quantity": 0, "limit_price": None}

    is_buy = action in (1, 3, 5, 7)
    is_large = action in (3, 4, 7, 8)
    is_limit = action in (5, 6, 7, 8)
    qty = LARGE_ORDER if is_large else SMALL_ORDER
    side = "BUY" if is_buy else "SELL"
    if is_limit:
        price = int(bid_cents if is_buy else ask_cents)
        kind = "LIMIT"
    else:
        price = None
        kind = "MARKET"
    return {"kind": kind, "side": side, "quantity": qty, "limit_price": price}


def metadata_template(team_id: str, team_name: str = "", members: list[str] | None = None) -> dict:
    return {
        "team_id": team_id,
        "team_name": team_name or team_id,
        "members": members or [],
        "model_format": "torchscript",
        "contract_version": CONTRACT_VERSION,
        "observation_size": OBSERVATION_SIZE,
        "action_size": ACTION_SIZE,
        "training_algorithm": "REPLACE_ME",
        "notes": "",
    }
