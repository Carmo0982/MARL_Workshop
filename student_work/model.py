from __future__ import annotations
import torch
from torch import nn
from shared.contract import OBSERVATION_SIZE, ACTION_SIZE

class QNetwork(nn.Module):
    """Starter network. You may redesign the internal architecture.

    The only fixed requirement is:
      input  : float32 tensor [batch, 16]
      output : float32 tensor [batch, 9]
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, ACTION_SIZE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuelingQNetwork(nn.Module):
    """Dueling DQN head (Wang et al., 2016): Q(s,a) = V(s) + A(s,a) - mean_a A(s,a).

    Many market states are "nothing to do" states where the choice of action
    barely matters; the separate value stream lets the network learn V(s)
    from every transition instead of only from the action that was taken.
    LayerNorm keeps activations stable across very different regimes
    (stable vs. crisis) without needing running statistics at inference time.
    Same contract: [batch, 16] float32 -> [batch, 9] float32.
    """
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
        self.advantage = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, ACTION_SIZE))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        v = self.value(h)
        a = self.advantage(h)
        return v + a - a.mean(dim=1, keepdim=True)


def build_model(arch: str = "dueling", hidden: int = 128) -> nn.Module:
    if arch == "mlp":
        return QNetwork()
    if arch == "dueling":
        return DuelingQNetwork(hidden)
    raise ValueError(f"Unknown architecture '{arch}' (use 'mlp' or 'dueling')")
