"""Ornstein-Uhlenbeck process fitting and simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OUParams:
    theta: float
    mu: float
    sigma: float
    dt: float

    @property
    def half_life(self) -> float:
        """Mean-reversion half-life; inf if theta <= 0."""
        if self.theta <= 0 or math.isnan(self.theta):
            return math.inf
        return math.log(2) / self.theta

    @property
    def stationary_std(self) -> float:
        """Stationary standard deviation sigma / sqrt(2*theta); nan if theta <= 0."""
        if self.theta <= 0 or math.isnan(self.theta):
            return math.nan
        return self.sigma / math.sqrt(2 * self.theta)


def fit_ou(spread: pd.Series | np.ndarray, dt: float = 1.0) -> OUParams:
    """Fit an OU process via the AR(1) regression x_{t+1} = a + b x_t + eps.

    If the estimated AR coefficient is outside (0, 1) the process is not
    mean-reverting; theta is then set to 0 or nan rather than raising.
    """
    x = np.asarray(spread, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 10:
        raise ValueError("Need at least 10 observations to fit OU parameters")

    x_prev, x_next = x[:-1], x[1:]
    b, a = np.polyfit(x_prev, x_next, 1)
    eps = x_next - (a + b * x_prev)

    if b <= 0:
        return OUParams(theta=math.nan, mu=math.nan, sigma=math.nan, dt=dt)
    if b >= 1:
        return OUParams(theta=0.0, mu=math.nan, sigma=math.nan, dt=dt)

    theta = -math.log(b) / dt
    mu = a / (1 - b)
    sigma = float(np.std(eps, ddof=1)) * math.sqrt(
        -2 * math.log(b) / (dt * (1 - b**2))
    )
    return OUParams(theta=theta, mu=mu, sigma=sigma, dt=dt)


def half_life(spread: pd.Series | np.ndarray, dt: float = 1.0) -> float:
    """Convenience: half-life implied by an OU fit."""
    return fit_ou(spread, dt=dt).half_life


def simulate_ou(
    params: OUParams,
    n: int,
    x0: float | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate an OU path using the exact discretisation."""
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = params.mu if x0 is None else x0
    decay = math.exp(-params.theta * params.dt)
    vol = params.sigma * math.sqrt(
        (1 - math.exp(-2 * params.theta * params.dt)) / (2 * params.theta)
    )
    eps = rng.normal(0.0, vol, n)
    for t in range(1, n):
        x[t] = params.mu + (x[t - 1] - params.mu) * decay + eps[t]
    return x
