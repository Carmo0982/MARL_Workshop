from __future__ import annotations

import argparse
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from shared.contract import OBSERVATION_SIZE, ACTION_SIZE
from shared.practice_market import PracticeMarketEnv
from student_work.model import QNetwork

SCENARIOS = ["stable", "trending", "volatile"]

class Replay:
    def __init__(self, capacity=50_000):
        self.data = deque(maxlen=capacity)
    def add(self, *row): self.data.append(row)
    def sample(self, n):
        batch = random.sample(self.data, n)
        s, a, r, ns, d = zip(*batch)
        return (
            torch.tensor(np.asarray(s), dtype=torch.float32),
            torch.tensor(a, dtype=torch.int64),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(np.asarray(ns), dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32),
        )
    def __len__(self): return len(self.data)


def train(steps: int, seed: int, out: Path):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cpu")  # assessment inference is CPU-safe and local
    q = QNetwork().to(device)
    target = QNetwork().to(device)
    target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=1e-3)
    replay = Replay()
    gamma = 0.995
    batch_size = 64
    global_step = 0
    episode = 0

    while global_step < steps:
        scenario = SCENARIOS[episode % len(SCENARIOS)]
        env = PracticeMarketEnv(scenario=scenario, steps=250, seed=seed + episode)
        state = env.reset()
        done = False
        ep_reward = 0.0
        while not done and global_step < steps:
            epsilon = max(0.05, 1.0 - global_step / max(1, steps * 0.75))
            if random.random() < epsilon:
                action = random.randrange(ACTION_SIZE)
            else:
                with torch.no_grad():
                    x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    action = int(q(x).argmax(dim=1).item())
            next_state, reward, done, info = env.step(action)
            replay.add(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            global_step += 1

            if len(replay) >= batch_size:
                s, a, r, ns, d = replay.sample(batch_size)
                pred = q(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    nxt = target(ns).max(dim=1).values
                    y = r + gamma * (1.0 - d) * nxt
                loss = nn.functional.smooth_l1_loss(pred, y)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()

            if global_step % 250 == 0:
                target.load_state_dict(q.state_dict())
        episode += 1
        print(f"episode={episode:03d} scenario={scenario:9s} steps={global_step:6d}/{steps} "
              f"reward={ep_reward:9.3f} nav=${info['nav_cents']/100:,.2f}")

    out.parent.mkdir(parents=True, exist_ok=True)
    q.eval()
    scripted = torch.jit.script(q.cpu())
    scripted.save(str(out))
    print(f"Saved TorchScript policy to: {out}")


def main():
    p = argparse.ArgumentParser(description="Train the starter DQN in the practice market.")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=Path, default=Path("outputs/model.pt"))
    args = p.parse_args()
    train(args.steps, args.seed, args.out)

if __name__ == "__main__":
    main()
