"""Demo 05: collect transitions and perform ONE real DQN optimisation update."""
import random
import numpy as np
import torch
from torch import nn

from shared.contract import ACTION_SIZE
from shared.practice_market import PracticeMarketEnv
from student_work.model import QNetwork

random.seed(5)
np.random.seed(5)
torch.manual_seed(5)

env = PracticeMarketEnv(scenario="stable", steps=80, seed=5)
state = env.reset()
transitions = []

# Collect a small replay sample with random exploration.
for _ in range(64):
    action = random.randrange(ACTION_SIZE)
    next_state, reward, done, _ = env.step(action)
    transitions.append((state, action, reward, next_state, done))
    state = next_state
    if done:
        break

states = torch.tensor(np.asarray([t[0] for t in transitions]), dtype=torch.float32)
actions = torch.tensor([t[1] for t in transitions], dtype=torch.int64)
rewards = torch.tensor([t[2] for t in transitions], dtype=torch.float32)
next_states = torch.tensor(np.asarray([t[3] for t in transitions]), dtype=torch.float32)
dones = torch.tensor([t[4] for t in transitions], dtype=torch.float32)

q = QNetwork()
target = QNetwork()
target.load_state_dict(q.state_dict())
optimizer = torch.optim.Adam(q.parameters(), lr=1e-3)
gamma = 0.995

predicted_q = q(states).gather(1, actions.unsqueeze(1)).squeeze(1)
with torch.no_grad():
    next_q = target(next_states).max(dim=1).values
    target_q = rewards + gamma * (1.0 - dones) * next_q

loss = nn.functional.smooth_l1_loss(predicted_q, target_q)
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"Transitions collected: {len(transitions)}")
print(f"One DQN update completed. Loss = {loss.item():.6f}")
print("The full student_work/train.py repeats this process with replay, exploration and target updates.")
