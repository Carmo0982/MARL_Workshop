from __future__ import annotations
import argparse
from pathlib import Path
import torch
from shared.practice_market import PracticeMarketEnv

@torch.inference_mode()
def run(model_path: Path, scenario: str, seed: int):
    model = torch.jit.load(str(model_path), map_location="cpu").eval()
    env = PracticeMarketEnv(scenario=scenario, steps=300, seed=seed)
    obs = env.reset(); done = False
    while not done:
        x = torch.tensor(obs).unsqueeze(0)
        action = int(model(x).argmax(dim=1).item())
        obs, reward, done, info = env.step(action)
    print(f"{scenario}: NAV=${info['nav_cents']/100:,.2f}, "
          f"PnL=${(info['nav_cents']-10_000_000)/100:,.2f}, trades={info['trade_count']}")

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument('--model',type=Path,default=Path('outputs/model.pt'))
    p.add_argument('--scenario',choices=['stable','trending','volatile','all'],default='all')
    p.add_argument('--seed',type=int,default=101); a=p.parse_args()
    scenarios=['stable','trending','volatile'] if a.scenario=='all' else [a.scenario]
    for i,s in enumerate(scenarios): run(a.model,s,a.seed+i)
