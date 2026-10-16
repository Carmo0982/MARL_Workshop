"""Demo 04: pass one observation through the starter neural network."""
import torch
from shared.contract import ACTION_NAMES
from shared.practice_market import PracticeMarketEnv
from student_work.model import QNetwork


env = PracticeMarketEnv(scenario="trending", steps=10, seed=4)
obs = env.reset()

model = QNetwork().eval()
x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # [1, 16]

with torch.no_grad():
    q_values = model(x)  # [1, 9]
    action = int(q_values.argmax(dim=1).item())

print("Input tensor shape :", tuple(x.shape))
print("Output tensor shape:", tuple(q_values.shape))
print("\nQ-values from an UNTRAINED network:")
for i, q in enumerate(q_values[0].tolist()):
    print(f"{i}: {ACTION_NAMES[i]:22s} {q: .5f}")
print("\nArgmax action:", action, ACTION_NAMES[action])
print("Note: this network is untrained, so this action has no trading meaning yet.")
