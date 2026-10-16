"""Multi-scenario, multi-seed evaluation of frozen policies.

Examples (run from the project root):

    # Compare the final model with the starter DQN and the rule baselines
    .venv/bin/python -m student_work.evaluate \
        --policies outputs/model.pt outputs/baseline/model.pt hold momentum mean_reversion \
        --suite full --seeds val --out outputs/eval/compare

    # Final report numbers (use the held-out TEST seeds only once!)
    .venv/bin/python -m student_work.evaluate --policies outputs/model.pt hold --seeds test

A policy is either a path to a TorchScript ``.pt`` file (loaded exactly like the
lecturer does: ``torch.jit.load`` + ``argmax``) or a baseline name from
``student_work/baselines.py``.
"""
from __future__ import annotations

import argparse
import csv
import json
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from shared.contract import ACTION_SIZE, STARTING_CASH_CENTS
from shared.practice_market import PracticeMarketEnv
from student_work.baselines import BASELINES
from student_work.envs import EVAL_SCENARIOS, TEST_SEEDS, VALIDATION_SEEDS, RobustMarketEnv

Policy = Callable[[np.ndarray], int]

SUITES = {
    # Our richer market, every regime, randomized price level / length / params.
    "full": [("robust", s) for s in EVAL_SCENARIOS],
    # Exactly what shared/test_policy.py runs (official practice market, 300 steps).
    "practice": [("practice", s) for s in ("stable", "trending", "volatile")],
    # Small and fast, useful during training.
    "quick": [("robust", s) for s in ("stable", "trend_up", "trend_down", "volatile", "low_liquidity", "stress")],
}


# ---------------------------------------------------------------------------
# Loading policies
# ---------------------------------------------------------------------------


def torchscript_policy(path: str | Path) -> Policy:
    model = torch.jit.load(str(path), map_location="cpu").eval()

    @torch.inference_mode()
    def policy(obs: np.ndarray) -> int:
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        return int(model(x).argmax(dim=1).item())

    return policy


def module_policy(module: torch.nn.Module) -> Policy:
    """Wrap an in-memory network (used by the trainer for fast validation)."""

    @torch.inference_mode()
    def policy(obs: np.ndarray) -> int:
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        return int(module(x).argmax(dim=1).item())

    return policy


def load_policy(spec: str) -> tuple[str, Policy]:
    if spec in BASELINES:
        return spec, BASELINES[spec]()
    p = Path(spec)
    if not p.exists():
        raise SystemExit(f"Policy '{spec}' is neither a baseline {list(BASELINES)} nor an existing file")
    name = p.parent.name + "/" + p.stem if p.parent.name not in ("", ".") else p.stem
    return name, torchscript_policy(p)


# ---------------------------------------------------------------------------
# Running episodes
# ---------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    policy: str
    env: str
    scenario: str
    seed: int
    steps: int
    pnl: float              # dollars
    return_pct: float
    max_drawdown: float     # dollars
    max_drawdown_pct: float
    sharpe: float           # per-episode mean/std of step PnL * sqrt(steps)
    trades: int
    final_inventory: int
    mean_abs_inventory: float
    max_abs_inventory: int
    hold_pct: float
    limit_pct: float
    market_pct: float


EVAL_PRICE_CENTS = 10_000.0  # $100, the official practice-market price level


def make_env(kind: str, scenario: str, seed: int):
    if kind == "practice":
        return PracticeMarketEnv(scenario=scenario, steps=300, seed=seed)
    # Each scenario gets its own RNG stream (otherwise every scenario of a given seed
    # shares the same jitter, length and price), and the price level is fixed: with a
    # share-based position limit and a dollar-based return, a random price level would
    # let one high-price seed dominate the averages.
    return RobustMarketEnv(scenario, seed=seed * 1000 + zlib.crc32(scenario.encode()) % 1000,
                           randomize=True, price_range=(EVAL_PRICE_CENTS, EVAL_PRICE_CENTS))


def run_episode(policy: Policy, kind: str, scenario: str, seed: int, name: str = "policy") -> EpisodeResult:
    env = make_env(kind, scenario, seed)
    obs = env.reset()
    done = False
    navs = [float(STARTING_CASH_CENTS)]
    invs = []
    actions = np.zeros(ACTION_SIZE, dtype=np.int64)
    info = {"trade_count": 0, "inventory": 0}
    while not done:
        a = int(policy(obs))
        actions[a] += 1
        obs, _, done, info = env.step(a)
        navs.append(info["nav_cents"])
        invs.append(abs(info["inventory"]))
    navs_arr = np.asarray(navs) / 100.0
    peak = np.maximum.accumulate(navs_arr)
    dd = float(np.max(peak - navs_arr))
    step_pnl = np.diff(navs_arr)
    sd = float(np.std(step_pnl))
    sharpe = float(np.mean(step_pnl) / sd * np.sqrt(len(step_pnl))) if sd > 1e-9 else 0.0
    n = max(1, int(actions.sum()))
    start = STARTING_CASH_CENTS / 100.0
    pnl = float(navs_arr[-1] - start)
    return EpisodeResult(
        policy=name, env=kind, scenario=scenario, seed=seed, steps=len(step_pnl),
        pnl=pnl, return_pct=100.0 * pnl / start,
        max_drawdown=dd, max_drawdown_pct=100.0 * dd / start, sharpe=sharpe,
        trades=int(info["trade_count"]), final_inventory=int(info["inventory"]),
        mean_abs_inventory=float(np.mean(invs)) if invs else 0.0,
        max_abs_inventory=int(max(invs)) if invs else 0,
        hold_pct=100.0 * actions[0] / n,
        limit_pct=100.0 * actions[5:].sum() / n,
        market_pct=100.0 * actions[1:5].sum() / n,
    )


def evaluate_policy(policy: Policy, suite: str = "quick", seeds=None, name: str = "policy") -> list[EpisodeResult]:
    seeds = list(seeds if seeds is not None else VALIDATION_SEEDS)
    out = []
    for kind, scenario in SUITES[suite]:
        for s in seeds:
            out.append(run_episode(policy, kind, scenario, s, name))
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def summarize(results: list[EpisodeResult]) -> dict:
    """Aggregate one policy's episodes into a per-scenario table and a scalar score.

    score = mean return %  - 0.5 * mean max-drawdown %  + 0.25 * worst-scenario mean return %

    Rationale: reward profit, punish risk, and punish policies that only work in
    one regime (the brief explicitly asks for robustness, not single-seed luck).
    """
    by_sc: dict[str, list[EpisodeResult]] = {}
    for r in results:
        by_sc.setdefault(f"{r.env}:{r.scenario}", []).append(r)
    table = {}
    for k, rs in by_sc.items():
        table[k] = {
            "pnl_mean": float(np.mean([r.pnl for r in rs])),
            "pnl_std": float(np.std([r.pnl for r in rs])),
            "ret_mean": float(np.mean([r.return_pct for r in rs])),
            "dd_mean": float(np.mean([r.max_drawdown_pct for r in rs])),
            "trades_mean": float(np.mean([r.trades for r in rs])),
            "abs_final_inv": float(np.mean([abs(r.final_inventory) for r in rs])),
            "win_rate": float(np.mean([r.pnl > 0 for r in rs])),
        }
    rets = [v["ret_mean"] for v in table.values()]
    dds = [v["dd_mean"] for v in table.values()]
    mean_ret, mean_dd, worst = float(np.mean(rets)), float(np.mean(dds)), float(np.min(rets))
    return {
        "per_scenario": table,
        "mean_return_pct": mean_ret,
        "mean_drawdown_pct": mean_dd,
        "worst_scenario_return_pct": worst,
        "mean_pnl": float(np.mean([r.pnl for r in results])),
        "mean_trades": float(np.mean([r.trades for r in results])),
        "score": mean_ret - 0.5 * mean_dd + 0.25 * worst,
    }


def print_comparison(summaries: dict[str, dict]) -> str:
    names = list(summaries)
    scenarios = list(next(iter(summaries.values()))["per_scenario"])
    lines = []
    w = max(12, max(len(n) for n in names) + 1)
    lines.append("Mean PnL ($) per scenario  [mean max-drawdown %, mean trades]")
    lines.append(f"{'scenario':24s}" + "".join(f"{n:>{w + 14}s}" for n in names))
    for sc in scenarios:
        row = f"{sc:24s}"
        for n in names:
            v = summaries[n]["per_scenario"][sc]
            row += f"{v['pnl_mean']:>{w}.0f} [{v['dd_mean']:4.1f},{v['trades_mean']:4.0f}]".rjust(w + 14)
        lines.append(row)
    lines.append("")
    lines.append(f"{'overall':24s}" + "".join(f"{n:>{w + 14}s}" for n in names))
    for key, label in [("mean_pnl", "mean PnL $"), ("mean_return_pct", "mean return %"),
                       ("mean_drawdown_pct", "mean max DD %"), ("worst_scenario_return_pct", "worst scen. ret %"),
                       ("mean_trades", "mean trades"), ("score", "SCORE")]:
        lines.append(f"{label:24s}" + "".join(f"{summaries[n][key]:>{w + 14}.3f}" for n in names))
    text = "\n".join(lines)
    print(text)
    return text


def save_outputs(out_dir: Path, results: list[EpisodeResult], summaries: dict[str, dict], text: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "episodes.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(asdict(results[0])))
        wr.writeheader()
        for r in results:
            wr.writerow(asdict(r))
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    (out_dir / "summary.txt").write_text(text + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = list(summaries)
        scenarios = list(summaries[names[0]]["per_scenario"])
        x = np.arange(len(scenarios))
        width = 0.8 / len(names)
        fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(scenarios)), 4.5))
        for i, n in enumerate(names):
            vals = [summaries[n]["per_scenario"][s]["pnl_mean"] for s in scenarios]
            ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=n)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([s.split(":")[1] for s in scenarios], rotation=30, ha="right")
        ax.set_ylabel("Mean PnL ($)")
        ax.set_title("Mean PnL by scenario")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "pnl_by_scenario.png", dpi=130)
        plt.close(fig)
    except Exception as e:  # plotting is optional
        print("Plot skipped:", e)
    print(f"\nSaved: {out_dir}/episodes.csv, summary.json, summary.txt, pnl_by_scenario.png")


def parse_seeds(text: str) -> list[int]:
    if text == "val":
        return VALIDATION_SEEDS
    if text == "test":
        return TEST_SEEDS
    return [int(x) for x in text.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser(description="Evaluate policies over scenarios and seeds.")
    p.add_argument("--policies", nargs="+", default=["outputs/model.pt", "hold", "momentum"])
    p.add_argument("--suite", choices=list(SUITES), default="full")
    p.add_argument("--seeds", default="val", help="'val', 'test' or comma-separated ints")
    p.add_argument("--out", type=Path, default=Path("outputs/eval/latest"))
    a = p.parse_args()
    seeds = parse_seeds(a.seeds)
    all_results, summaries = [], {}
    for spec in a.policies:
        name, pol = load_policy(spec)
        res = evaluate_policy(pol, a.suite, seeds, name)
        all_results += res
        summaries[name] = summarize(res)
    print(f"suite={a.suite} seeds={seeds}\n")
    text = print_comparison(summaries)
    save_outputs(a.out, all_results, summaries, text)


if __name__ == "__main__":
    main()
