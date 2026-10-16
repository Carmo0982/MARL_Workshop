"""Transparent rule-based policies used as reference points.

Every policy is a function ``obs (np.ndarray[16]) -> action (int 0..8)`` using the
official observation indices from ``shared.contract.OBSERVATION_NAMES``.
An RL model is only interesting if it beats these under the same seeds.
"""
from __future__ import annotations

import numpy as np

# Observation indices (see shared/contract.py)
R1, R5, R20, VOL, SPREAD, IMB, BID_D, ASK_D, INV, CASH, NAV, TIME, OPEN, PV5, PV20, BIAS = range(16)


def hold(obs) -> int:
    return 0


def make_random(seed: int = 0):
    rng = np.random.default_rng(seed)

    def random_policy(obs) -> int:
        return int(rng.integers(0, 9))

    return random_policy


def momentum(obs, threshold: float = 0.015) -> int:
    """The workshop demo_02 rule: follow the 5-step return."""
    if obs[R5] > threshold and obs[INV] < 0.8:
        return 1
    if obs[R5] < -threshold and obs[INV] > -0.8:
        return 2
    return 0


def mean_reversion(obs, threshold: float = 0.03) -> int:
    """Fade deviations from the 20-step moving average using passive limit orders."""
    if obs[PV20] < -threshold and obs[INV] < 0.5:
        return 5  # BUY_LIMIT_SMALL
    if obs[PV20] > threshold and obs[INV] > -0.5:
        return 6  # SELL_LIMIT_SMALL
    # Flatten slowly near the end of the day.
    if obs[TIME] > 0.9 and abs(obs[INV]) > 0.01:
        return 2 if obs[INV] > 0 else 1
    return 0


def buy_and_hold(obs) -> int:
    """Buy up to ~40% of the position limit once, then hold."""
    return 1 if obs[INV] < 0.4 else 0


BASELINES = {
    "hold": lambda: hold,
    "random": lambda: make_random(0),
    "momentum": lambda: momentum,
    "mean_reversion": lambda: mean_reversion,
    "buy_and_hold": lambda: buy_and_hold,
}
