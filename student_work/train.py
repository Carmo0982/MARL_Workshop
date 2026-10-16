"""Improved DQN trainer: Double DQN + Dueling + n-step returns + domain randomization.

What changed with respect to the starter (kept as ``train_starter.py``):

* **Double DQN** - the online net picks the next action, the target net
  evaluates it; reduces the over-estimation bias of ``max_a Q_target``.
* **Dueling head + LayerNorm** (``model.DuelingQNetwork``).
* **n-step returns** (default 3) - faster reward propagation in a noisy market.
* **Soft target updates** (Polyak, tau) instead of a hard copy every 250 steps.
* **Domain randomization** - every episode samples a regime from
  ``envs.SCENARIOS`` (incl. down-trend, low liquidity, crisis), jitters its
  parameters, the price level and the episode length; a fraction of episodes
  uses the official practice market.
* **Risk-aware reward** (``envs.RewardConfig``) - inventory, drawdown,
  turnover and end-of-day inventory penalties.
* **Model selection on held-out validation seeds**: every ``--eval-every``
  steps the greedy policy is evaluated on ``envs.VALIDATION_SEEDS`` and the best
  checkpoint (by ``evaluate.summarize`` score) is exported to ``--out``.

Everything is logged to ``outputs/runs/<run-name>/`` for the report.

Examples:
    .venv/bin/python -m student_work.train --steps 150000 --run-name ddqn_main
    .venv/bin/python -m student_work.train --steps 5000 --seed 7          # quick smoke run
    .venv/bin/python -m student_work.train --reward starter --no-randomize --run-name ablation_no_dr
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from shared.contract import ACTION_SIZE, OBSERVATION_SIZE
from student_work.envs import REWARD_PRESETS, SCENARIOS, VALIDATION_SEEDS, ScenarioSampler
from student_work.evaluate import evaluate_policy, module_policy, summarize
from student_work.model import build_model


class Replay:
    """Fixed-size ring buffer stored in numpy arrays (fast sampling)."""

    def __init__(self, capacity: int):
        self.cap = capacity
        self.s = np.zeros((capacity, OBSERVATION_SIZE), np.float32)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.ns = np.zeros((capacity, OBSERVATION_SIZE), np.float32)
        self.d = np.zeros(capacity, np.float32)
        self.g = np.zeros(capacity, np.float32)  # discount to apply to the bootstrap (gamma^n)
        self.i = 0
        self.n = 0

    def add(self, s, a, r, ns, d, g):
        i = self.i
        self.s[i], self.a[i], self.r[i], self.ns[i], self.d[i], self.g[i] = s, a, r, ns, d, g
        self.i = (i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, k: int):
        idx = np.random.randint(0, self.n, size=k)
        t = torch.from_numpy
        return t(self.s[idx]), t(self.a[idx]), t(self.r[idx]), t(self.ns[idx]), t(self.d[idx]), t(self.g[idx])

    def __len__(self):
        return self.n


class NStep:
    """Turns 1-step transitions into n-step transitions before they enter the replay."""

    def __init__(self, n: int, gamma: float):
        self.n, self.gamma = n, gamma
        self.buf: deque = deque()

    def push(self, s, a, r, ns, d):
        self.buf.append((s, a, r, ns, d))
        out = []
        if len(self.buf) >= self.n:
            out.append(self._make())
            self.buf.popleft()
        if d:  # flush the tail of the episode
            while self.buf:
                out.append(self._make())
                self.buf.popleft()
        return out

    def _make(self):
        R, g = 0.0, 1.0
        s0, a0 = self.buf[0][0], self.buf[0][1]
        ns, d = self.buf[0][3], self.buf[0][4]
        for (_, _, r, ns_k, d_k) in self.buf:
            R += g * r
            g *= self.gamma
            ns, d = ns_k, d_k
            if d_k:
                break
        return s0, a0, R, ns, float(d), g


def load_abides_data(paths: list[str], gamma: float, clip: float = 50.0) -> Replay:
    """Load .npz files written by student_work/abides_collect.py into a separate replay.

    ABIDES rewards are heavy-tailed (in stress_test the mid price jumps when one side of
    the book empties), so they are clipped to +/-clip to keep TD targets stable while
    preserving the sign of the lesson (holding inventory in a crisis is very bad).
    """
    arrays = [np.load(p) for p in paths]
    n = sum(len(x["a"]) for x in arrays)
    rb = Replay(n)
    for x in arrays:
        for i in range(len(x["a"])):
            rb.add(x["s"][i], x["a"][i], float(np.clip(x["r"][i], -clip, clip)), x["ns"][i], x["d"][i], gamma)
    print(f"Loaded {n} ABIDES transitions from {len(paths)} file(s)")
    return rb


def export_torchscript(net: nn.Module, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    net = net.eval()
    scripted = torch.jit.script(net.cpu())
    scripted.save(str(path))
    # Round-trip check: exactly what the lecturer / validator will do.
    m = torch.jit.load(str(path), map_location="cpu").eval()
    with torch.inference_mode():
        y = m(torch.zeros((2, OBSERVATION_SIZE), dtype=torch.float32))
    assert tuple(y.shape) == (2, ACTION_SIZE) and torch.isfinite(y).all(), "exported model broke the contract"


def train(a: argparse.Namespace):
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    run_dir = Path("outputs/runs") / a.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(
        {**vars(a), "reward_config": asdict(REWARD_PRESETS[a.reward])}, indent=2, default=str))

    q = build_model(a.arch, a.hidden)
    if a.init:  # warm start (e.g. fine-tune a model with ABIDES data)
        src = torch.jit.load(str(a.init), map_location="cpu")
        q.load_state_dict(src.state_dict())
        print(f"Warm-started from {a.init}")
    target = build_model(a.arch, a.hidden)
    target.load_state_dict(q.state_dict())
    target.eval()
    opt = torch.optim.Adam(q.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.1, 1.0 - s / max(1, a.steps)))
    replay = Replay(a.buffer)
    nstep = NStep(a.n_step, a.gamma)
    abides_replay = load_abides_data(a.abides_data, a.gamma, a.abides_reward_clip) if a.abides_data else None
    k_abides = int(round(a.batch * a.abides_frac)) if abides_replay is not None else 0
    scenarios = a.scenarios or list(SCENARIOS)
    weights = [a.abides_weight if n.startswith("abides") else 1.0 for n in scenarios]
    sampler = ScenarioSampler(
        scenarios=scenarios, weights=weights, randomize=not a.no_randomize, reward=REWARD_PRESETS[a.reward],
        limit_model=a.limit_model, include_practice=a.practice_frac, seed=a.seed,
    )

    train_log = open(run_dir / "train_log.csv", "w", newline="")
    tw = csv.writer(train_log)
    tw.writerow(["step", "episode", "scenario", "ep_reward", "ep_pnl", "trades", "epsilon", "loss"])
    val_hist = []
    best_score, best_step = -float("inf"), -1
    global_step, episode, t0 = 0, 0, time.time()
    losses: deque = deque(maxlen=500)

    def validate():
        nonlocal best_score, best_step
        q.eval()
        res = evaluate_policy(module_policy(q), a.eval_suite, VALIDATION_SEEDS, name="val")
        q.train()
        s = summarize(res)
        val_hist.append({"step": global_step, **{k: v for k, v in s.items() if k != "per_scenario"},
                         "per_scenario": {k: v["pnl_mean"] for k, v in s["per_scenario"].items()}})
        export_torchscript(q, run_dir / f"ckpt_{global_step:07d}.pt")
        tag = ""
        if s["score"] > best_score:
            best_score, best_step = s["score"], global_step
            shutil.copy(run_dir / f"ckpt_{global_step:07d}.pt", run_dir / "model_best.pt")
            tag = "  <- new best"
        print(f"[val] step={global_step:7d} score={s['score']:7.3f} ret={s['mean_return_pct']:6.2f}% "
              f"dd={s['mean_drawdown_pct']:5.2f}% worst={s['worst_scenario_return_pct']:6.2f}% "
              f"trades={s['mean_trades']:5.0f}{tag}", flush=True)
        (run_dir / "val_log.json").write_text(json.dumps(val_hist, indent=2))

    while global_step < a.steps:
        env, sc_name = sampler.sample(episode_seed=a.seed * 100_000 + episode)
        state = env.reset()
        done, ep_reward = False, 0.0
        info = {"nav_cents": 10_000_000, "trade_count": 0}
        while not done and global_step < a.steps:
            eps = max(a.eps_end, a.eps_start - (a.eps_start - a.eps_end) * global_step / max(1, a.steps * a.eps_frac))
            if random.random() < eps:
                action = random.randrange(ACTION_SIZE)
            else:
                with torch.no_grad():
                    action = int(q(torch.from_numpy(state).unsqueeze(0)).argmax(dim=1).item())
            next_state, reward, done, info = env.step(action)
            for tr in nstep.push(state, action, reward, next_state, done):
                replay.add(*tr)
            state = next_state
            ep_reward += reward
            global_step += 1

            if len(replay) >= a.learning_starts and global_step % a.train_every == 0:
                s, act, r, ns, d, g = replay.sample(a.batch - k_abides)
                if k_abides:  # mix real ABIDES transitions into every mini-batch
                    extra = abides_replay.sample(k_abides)
                    s, act, r, ns, d, g = (torch.cat([x, y]) for x, y in zip((s, act, r, ns, d, g), extra))
                pred = q(s).gather(1, act.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    if a.double:
                        na = q(ns).argmax(dim=1, keepdim=True)
                        nxt = target(ns).gather(1, na).squeeze(1)
                    else:
                        nxt = target(ns).max(dim=1).values
                    y = r + g * (1.0 - d) * nxt
                loss = nn.functional.smooth_l1_loss(pred, y)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                opt.step(); sched.step()
                losses.append(float(loss.item()))
                with torch.no_grad():
                    for p_t, p in zip(target.parameters(), q.parameters()):
                        p_t.mul_(1.0 - a.tau).add_(p, alpha=a.tau)

            if a.eval_every and global_step % a.eval_every == 0:
                validate()

        episode += 1
        pnl = (info["nav_cents"] - 10_000_000) / 100
        tw.writerow([global_step, episode, sc_name, round(ep_reward, 4), round(pnl, 2), info["trade_count"],
                     round(eps, 4), round(float(np.mean(losses)), 6) if losses else ""])
        if episode % a.print_every == 0:
            train_log.flush()
            print(f"ep={episode:5d} step={global_step:7d}/{a.steps} eps={eps:.3f} sc={sc_name:18s} "
                  f"R={ep_reward:9.2f} pnl=${pnl:10,.0f} trades={info['trade_count']:4d} "
                  f"loss={np.mean(losses) if losses else 0:.4f} ({time.time() - t0:.0f}s)", flush=True)
    train_log.close()

    if not val_hist or val_hist[-1]["step"] != global_step:
        validate()
    export_torchscript(q, run_dir / "model_last.pt")
    chosen = run_dir / ("model_best.pt" if a.select == "best" else "model_last.pt")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(chosen, a.out)
    plot_training(run_dir)
    print(f"\nBest validation score {best_score:.3f} at step {best_step}.")
    print(f"Saved TorchScript policy to: {a.out}  (copied from {chosen})")
    if a.out.resolve() != Path("outputs/model.pt").resolve():
        print(f"To submit it:  cp {a.out} outputs/model.pt")
    print(f"Run artefacts: {run_dir}")


def plot_training(run_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rows = list(csv.DictReader(open(run_dir / "train_log.csv")))
        val = json.loads((run_dir / "val_log.json").read_text())
        steps = np.array([int(r["step"]) for r in rows])
        rew = np.array([float(r["ep_reward"]) for r in rows])
        k = max(1, len(rew) // 50)
        smooth = np.convolve(rew, np.ones(k) / k, mode="valid")
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(steps, rew, alpha=0.25, label="episode reward")
        ax[0].plot(steps[k - 1:], smooth, label=f"moving avg ({k})")
        ax[0].set_xlabel("step"); ax[0].set_title("Training reward"); ax[0].legend()
        vs = [v["step"] for v in val]
        ax[1].plot(vs, [v["score"] for v in val], marker="o", label="validation score")
        ax[1].plot(vs, [v["mean_return_pct"] for v in val], marker=".", label="mean return %")
        ax[1].plot(vs, [v["mean_drawdown_pct"] for v in val], marker=".", label="mean max DD %")
        ax[1].set_xlabel("step"); ax[1].set_title("Validation (held-out seeds)"); ax[1].legend()
        fig.tight_layout(); fig.savefig(run_dir / "training_curve.png", dpi=130); plt.close(fig)
    except Exception as e:
        print("Plot skipped:", e)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train the improved DQN agent.")
    p.add_argument("--steps", type=int, default=150_000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=Path, default=None,
                   help="where to copy the selected model (default: outputs/candidates/<run-name>.pt). "
                        "Never defaults to outputs/model.pt, so a quick run cannot overwrite the model to submit.")
    p.add_argument("--run-name", default=None, help="folder under outputs/runs (default: auto)")
    # model / algorithm
    p.add_argument("--arch", choices=["dueling", "mlp"], default="dueling")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--no-double", dest="double", action="store_false", help="disable Double DQN")
    p.add_argument("--n-step", type=int, default=3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--buffer", type=int, default=200_000)
    p.add_argument("--learning-starts", type=int, default=2_000)
    p.add_argument("--train-every", type=int, default=1)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.02)
    p.add_argument("--eps-frac", type=float, default=0.4, help="fraction of training used to decay epsilon")
    # environment / reward
    p.add_argument("--reward", choices=list(REWARD_PRESETS), default="risk_aware")
    p.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS), default=None)
    p.add_argument("--abides-weight", type=float, default=2.0,
                   help="sampling weight of the ABIDES-calibrated scenarios relative to the others")
    p.add_argument("--no-randomize", action="store_true", help="disable parameter/price/length jitter")
    p.add_argument("--practice-frac", type=float, default=0.15, help="share of episodes on the official practice market")
    p.add_argument("--limit-model", choices=["resting", "instant"], default="resting")
    p.add_argument("--abides-data", nargs="*", default=None,
                   help="npz files from student_work/abides_collect.py to mix into training")
    p.add_argument("--abides-frac", type=float, default=0.25, help="share of each mini-batch taken from --abides-data")
    p.add_argument("--abides-reward-clip", type=float, default=50.0, help="clip ABIDES rewards to +/- this value")
    p.add_argument("--init", type=Path, default=None, help="TorchScript/state_dict to warm-start from (fine-tuning)")
    # evaluation / selection
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--eval-suite", choices=["quick", "full", "practice"], default="full")
    p.add_argument("--select", choices=["best", "last"], default="best")
    p.add_argument("--print-every", type=int, default=20)
    p.add_argument("--threads", type=int, default=2)
    a = p.parse_args(argv)
    if a.run_name is None:
        a.run_name = time.strftime("run_%Y%m%d_%H%M%S")
    if a.out is None:
        a.out = Path("outputs/candidates") / f"{a.run_name}.pt"
    a.eval_every = min(a.eval_every, max(1000, a.steps // 5)) if a.eval_every else 0
    return a


def main(argv=None):
    train(parse_args(argv))


if __name__ == "__main__":
    main()
