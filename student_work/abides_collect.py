"""Collect real ABIDES-NG transitions to reduce the simulator-to-ABIDES gap.

The fast training markets are only approximations of ABIDES. This script runs
an epsilon-greedy version of a trained policy inside ABIDES and stores the
(state, action, reward, next_state, done) transitions in an ``.npz`` file.
``train.py --abides-data outputs/abides_data/*.npz`` then mixes them into every
DQN mini-batch (off-policy learning can reuse data from any behaviour policy).

Example (5 scenarios x 4 seeds, 5 parallel workers, ~5-10 min):
    .venv/bin/python -m student_work.abides_collect --policy outputs/model.pt \
        --seeds 11 12 13 14 --epsilon 0.3 --out outputs/abides_data/round1.npz

Use seeds different from the ones you use for ABIDES evaluation (1, 2, ...).
"""
from __future__ import annotations

import argparse
import logging
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from shared.contract import ACTION_SIZE, STARTING_CASH_CENTS

COLLECT_SCENARIOS = ["stable_day", "trending_day", "volatile_day", "low_liquidity", "stress_test"]


def records_to_transitions(records, final_nav: float, final_inv: int, final_fills: int, reward_cfg):
    """Turn per-decision records into 1-step transitions with the training reward."""
    rc = reward_cfg
    S, A, R, NS, D = [], [], [], [], []
    peak = float(STARTING_CASH_CENTS)
    for t in range(len(records)):
        obs, action, nav, _, fills = records[t]
        last = t == len(records) - 1
        if last:
            next_obs, next_nav, next_inv, next_fills = obs, final_nav, final_inv, final_fills
        else:
            next_obs, next_nav, next_inv, next_fills = records[t + 1][0], records[t + 1][2], records[t + 1][3], records[t + 1][4]
        old_dd = max(0.0, peak - nav)
        peak = max(peak, nav, next_nav)
        dd_inc = max(0.0, (peak - next_nav) - old_dd)
        r = ((next_nav - nav) * rc.pnl_scale - rc.inventory_penalty * abs(next_inv)
             - rc.drawdown_penalty * dd_inc * rc.pnl_scale - rc.trade_penalty * (next_fills - fills))
        if last:
            r -= rc.terminal_inventory_penalty * abs(next_inv)
        S.append(obs); A.append(action); R.append(r); NS.append(next_obs); D.append(float(last))
    return (np.asarray(S, np.float32), np.asarray(A, np.int64), np.asarray(R, np.float32),
            np.asarray(NS, np.float32), np.asarray(D, np.float32))


def _job(args):
    policy_path, scenario, seed, epsilon, reward_name, wake = args
    logging.disable(logging.INFO)
    import torch
    torch.set_num_threads(1)
    from student_work import abides_eval
    from student_work.envs import REWARD_PRESETS
    from student_work.evaluate import load_policy

    _, greedy = load_policy(policy_path)
    rng = np.random.default_rng(seed)

    def behaviour(obs):
        return int(rng.integers(0, ACTION_SIZE)) if rng.random() < epsilon else greedy(obs)

    t0 = time.time()
    ours, _ = abides_eval.run_abides([("collector", behaviour)], scenario, seed, wake=wake)
    recs = abides_eval.RECORDS["collector"]
    res = ours["collector"]
    final_nav = res["pnl"] * 100 + STARTING_CASH_CENTS
    tr = records_to_transitions(recs, final_nav, res["final_inventory"], res["fills"], REWARD_PRESETS[reward_name])
    print(f"{scenario:13s} seed={seed:<4d} transitions={len(tr[0]):4d} pnl=${res['pnl']:9,.2f} ({time.time() - t0:.0f}s)", flush=True)
    return tr


def main():
    p = argparse.ArgumentParser(description="Collect ABIDES transitions for off-policy training.")
    p.add_argument("--policy", default="outputs/model.pt", help="TorchScript path or baseline name")
    p.add_argument("--scenarios", nargs="+", default=COLLECT_SCENARIOS)
    p.add_argument("--seeds", nargs="+", type=int, default=[11, 12, 13, 14])
    p.add_argument("--epsilon", type=float, default=0.3)
    p.add_argument("--reward", default="risk_aware")
    p.add_argument("--wake", default="60s")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--out", type=Path, default=Path("outputs/abides_data/collected.npz"))
    a = p.parse_args()
    jobs = [(a.policy, sc, s, a.epsilon, a.reward, a.wake) for sc in a.scenarios for s in a.seeds]
    with get_context("spawn").Pool(a.workers) as pool:
        parts = pool.map(_job, jobs)
    arrays = [np.concatenate([part[i] for part in parts]) for i in range(5)]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, s=arrays[0], a=arrays[1], r=arrays[2], ns=arrays[3], d=arrays[4])
    print(f"Saved {len(arrays[0])} transitions to {a.out}")
    print(f"reward mean={arrays[2].mean():.4f} std={arrays[2].std():.4f}; action counts={np.bincount(arrays[1], minlength=9)}")


if __name__ == "__main__":
    main()
