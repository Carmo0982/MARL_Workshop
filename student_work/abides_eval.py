"""Run our exported policy INSIDE the ABIDES-NG exchange (the official simulator).

This is our own approximation of the lecturer-side adapter: a TradingAgent that,
every ``--wake`` seconds, queries the order book, builds the official 16-value
observation with ``shared.contract.build_observation``, runs the TorchScript
model, takes ``argmax`` and converts it with ``shared.contract.action_to_order``.

Two modes, mirroring the assessment:
  * individual  - each policy alone in its own simulation (same scenario & seed);
  * tournament  - all policies trade at the same time in ONE exchange.

Examples:
    .venv/bin/python -m student_work.abides_eval --policies outputs/model.pt hold momentum \
        --scenarios stable_day volatile_day --seeds 1 2 --end-time 11:00:00

    .venv/bin/python -m student_work.abides_eval --mode tournament \
        --policies outputs/model.pt outputs/baseline/model.pt momentum --scenarios stress_test

Assumptions we had to make (the real adapter may differ, see NOTAS_EQUIPO.md):
  * order-book depth = shares in the best 5 price levels of each side;
  * resting limit orders are cancelled after ``--limit-ttl`` decisions;
  * the 500-share position limit is enforced before sending an order;
  * buying beyond cash is allowed only with ``--enforce-cash`` off (ABIDES default).
"""
from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path

import numpy as np

from shared.contract import (
    ACTION_SIZE,
    POSITION_LIMIT,
    STARTING_CASH_CENTS,
    action_to_order,
    build_observation,
)

ABIDES_SCENARIOS = ["stable_day", "trending_day", "volatile_day", "low_liquidity", "stress_test", "rmsc04"]
RESULTS: dict[str, dict] = {}  # filled by PolicyAgent.kernel_stopping
RECORDS: dict[str, list] = {}  # per-decision (obs, action, nav, inventory, fills) for data collection


def _make_agent_class():
    """Defined lazily so the module can be imported without ABIDES installed."""
    from abides_core.utils import str_to_ns
    from abides_markets.agents.trading_agent import TradingAgent
    from abides_markets.messages.query import QuerySpreadResponseMsg
    from abides_markets.orders import Side

    class PolicyAgent(TradingAgent):
        def __init__(self, id, label, policy, symbol="ABM", wake="60s", random_state=None,
                     limit_ttl=5, depth_levels=5, enforce_cash=False):
            super().__init__(id, name=f"POLICY_{label}", type="PolicyAgent",
                             random_state=random_state or np.random.RandomState(0),
                             starting_cash=STARTING_CASH_CENTS, log_orders=False)
            self.label, self.policy, self.symbol = label, policy, symbol
            self.wake_ns = str_to_ns(wake)
            self.limit_ttl, self.depth_levels, self.enforce_cash = limit_ttl, depth_levels, enforce_cash
            self.mid_history: list[float] = []
            self.order_age: dict[int, int] = {}
            self.actions = np.zeros(ACTION_SIZE, dtype=np.int64)
            self.navs: list[float] = []
            self.inventories: list[int] = []
            self.fills = 0
            self.blocked = 0
            self.records: list[tuple] = []
            self.state = "AWAITING_WAKEUP"

        # -- scheduling ---------------------------------------------------
        def get_wake_frequency(self):
            # Offset of the first wake-up after market open (TradingAgent schedules it
            # once market hours are known). Random so several agents do not act in lockstep.
            return int(self.random_state.randint(1, max(2, self.wake_ns)))

        def wakeup(self, current_time):
            can_trade = super().wakeup(current_time)
            if self.mkt_closed or not can_trade:
                return  # before open: TradingAgent will wake us at mkt_open + offset
            self.get_current_spread(self.symbol, depth=self.depth_levels)
            self.state = "AWAITING_SPREAD"

        def receive_message(self, current_time, sender_id, message):
            super().receive_message(current_time, sender_id, message)
            if self.state == "AWAITING_SPREAD" and isinstance(message, QuerySpreadResponseMsg):
                self.state = "AWAITING_WAKEUP"
                if not self.mkt_closed:
                    self.decide(current_time)
                    self.set_wakeup(current_time + self.wake_ns)

        def order_executed(self, order):
            super().order_executed(order)
            self.fills += 1

        # -- decision -------------------------------------------------------
        def decide(self, now):
            bids = self.known_bids.get(self.symbol, [])
            asks = self.known_asks.get(self.symbol, [])
            if not bids or not asks:
                return  # one-sided book: HOLD
            bid, ask = int(bids[0][0]), int(asks[0][0])
            bid_depth = int(sum(q for _, q in bids[: self.depth_levels]))
            ask_depth = int(sum(q for _, q in asks[: self.depth_levels]))
            mid = (bid + ask) / 2.0
            self.mid_history.append(mid)
            inv = int(self.holdings.get(self.symbol, 0))
            cash = int(self.holdings["CASH"])

            # Age and cancel stale resting orders.
            for oid in list(self.orders):
                self.order_age[oid] = self.order_age.get(oid, 0) + 1
                if self.order_age[oid] >= self.limit_ttl:
                    self.cancel_order(self.orders[oid])
            open_orders = len(self.orders)

            span = max(1, self.mkt_close - self.mkt_open)
            obs = build_observation(
                self.mid_history[-60:], bid, ask, bid_depth, ask_depth, inv, cash,
                time_progress=(now - self.mkt_open) / span, open_orders=open_orders,
            )
            action = int(self.policy(obs))
            self.actions[action] += 1
            self.navs.append(cash + inv * mid)
            self.inventories.append(inv)
            self.records.append((obs, action, cash + inv * mid, inv, self.fills))

            order = action_to_order(action, bid, ask)
            if order["kind"] == "HOLD":
                return
            qty = int(order["quantity"])
            signed = qty if order["side"] == "BUY" else -qty
            # Count resting orders on the same side so fills can never breach +/-500.
            pending = sum(o.quantity for o in self.orders.values()
                          if o.side.is_bid() == (signed > 0))
            worst_case = inv + signed + (pending if signed > 0 else -pending)
            if abs(worst_case) > POSITION_LIMIT or (self.enforce_cash and signed > 0 and cash < qty * ask):
                self.blocked += 1
                return
            side = Side.BID if order["side"] == "BUY" else Side.ASK
            if order["kind"] == "MARKET":
                self.place_market_order(self.symbol, qty, side)
            else:
                self.place_limit_order(self.symbol, qty, side, int(order["limit_price"]))

        def kernel_stopping(self):
            super().kernel_stopping()
            final = self.mark_to_market(self.holdings) / 100.0
            final_inv = int(self.holdings.get(self.symbol, 0))
            # Include the starting capital and the closing state, so the drawdown and the
            # max inventory also cover fills that happen after the last decision.
            navs = np.asarray([STARTING_CASH_CENTS, *self.navs], dtype=float) / 100.0
            navs = np.append(navs, final)
            peak = np.maximum.accumulate(navs)
            n = max(1, int(self.actions.sum()))
            RECORDS[self.label] = self.records
            RESULTS[self.label] = {
                "pnl": final - STARTING_CASH_CENTS / 100.0,
                "max_drawdown": float(np.max(peak - navs)),
                "decisions": int(self.actions.sum()),
                "fills": int(self.fills),
                "blocked": int(self.blocked),
                "final_inventory": final_inv,
                "max_abs_inventory": int(max([abs(final_inv), *map(abs, self.inventories)])),
                "hold_pct": 100.0 * self.actions[0] / n,
                "limit_pct": 100.0 * self.actions[5:].sum() / n,
                "market_pct": 100.0 * self.actions[1:5].sum() / n,
            }

    return PolicyAgent


def run_abides(policies: list[tuple[str, object]], scenario: str, seed: int, wake="60s",
               end_time: str | None = None, limit_ttl=5, enforce_cash=False) -> tuple[dict, dict]:
    """Run ONE ABIDES simulation with the given (label, policy) agents. Returns (our results, market info)."""
    from abides_markets.config_system import SimulationBuilder
    from abides_markets.simulation import run_simulation

    PolicyAgent = _make_agent_class()
    b = SimulationBuilder().apply_template(scenario).ticker("ABM").seed(seed)
    if end_time:
        b = b.end_time(end_time)
    config = b.build()
    rng = np.random.RandomState(seed)
    agents = [PolicyAgent(-1 - i, label, pol, wake=wake, limit_ttl=limit_ttl, enforce_cash=enforce_cash,
                          random_state=np.random.RandomState(rng.randint(0, 2**31 - 1)))
              for i, (label, pol) in enumerate(policies)]
    RESULTS.clear()
    RECORDS.clear()
    result = run_simulation(config, runtime_agents=agents, log_dir=str(Path("outputs/abides_logs") / f"{scenario}_{seed}"))
    ours = {k: dict(v) for k, v in RESULTS.items()}
    # Background agents for context: best and median PnL of the other traders.
    others = [a.pnl_cents / 100 for a in result.agents if a.agent_type not in ("PolicyAgent", "ExchangeAgent")]
    market = {"others_median_pnl": float(np.median(others)) if others else 0.0,
              "others_best_pnl": float(np.max(others)) if others else 0.0}
    return ours, market


def main():
    from student_work.evaluate import load_policy

    p = argparse.ArgumentParser(description="Evaluate policies inside ABIDES-NG.")
    p.add_argument("--policies", nargs="+", default=["outputs/model.pt", "hold"])
    p.add_argument("--scenarios", nargs="+", default=["stable_day", "volatile_day", "stress_test"],
                   choices=ABIDES_SCENARIOS)
    p.add_argument("--seeds", nargs="+", type=int, default=[1])
    p.add_argument("--mode", choices=["individual", "tournament"], default="individual")
    p.add_argument("--wake", default="60s", help="decision interval (ABIDES duration string)")
    p.add_argument("--end-time", default=None, help="e.g. 11:00:00 to shorten the day (faster)")
    p.add_argument("--limit-ttl", type=int, default=5)
    p.add_argument("--enforce-cash", action="store_true")
    p.add_argument("--out", type=Path, default=Path("outputs/abides_eval/results.csv"))
    a = p.parse_args()
    import torch
    torch.set_num_threads(1)  # tiny network: extra threads only slow the simulation down
    logging.getLogger("abides").setLevel(logging.WARNING)
    logging.getLogger().setLevel(logging.WARNING)

    loaded = [load_policy(s) for s in a.policies]
    rows = []
    for sc in a.scenarios:
        for seed in a.seeds:
            groups = [loaded] if a.mode == "tournament" else [[lp] for lp in loaded]
            for g in groups:
                t0 = time.time()
                ours, market = run_abides(g, sc, seed, a.wake, a.end_time, a.limit_ttl, a.enforce_cash)
                for label, r in ours.items():
                    row = {"mode": a.mode, "scenario": sc, "seed": seed, "policy": label, **r, **market}
                    rows.append(row)
                    print(f"{a.mode:10s} {sc:13s} seed={seed:<4d} {label:28s} PnL=${r['pnl']:10,.2f} "
                          f"maxDD=${r['max_drawdown']:9,.2f} fills={r['fills']:4d} inv={r['final_inventory']:5d} "
                          f"hold={r['hold_pct']:5.1f}% ({time.time() - t0:.0f}s)", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("\nMean PnL by policy:")
    for label in dict.fromkeys(r["policy"] for r in rows):
        pn = [r["pnl"] for r in rows if r["policy"] == label]
        print(f"  {label:28s} mean=${np.mean(pn):10,.2f}  min=${np.min(pn):10,.2f}  n={len(pn)}")
    print(f"Saved: {a.out}")


if __name__ == "__main__":
    main()
