"""Spread construction, z-scores, and stateful trading signals."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_spread(
    y: pd.Series, x: pd.Series, beta: float, alpha: float = 0.0
) -> pd.Series:
    """Residual spread ``y - alpha - beta * x``."""
    return y - alpha - beta * x


def zscore(spread: pd.Series, window: int | None = None) -> pd.Series:
    """Z-score using rolling statistics, or full-sample if window is None."""
    if window is None:
        return (spread - spread.mean()) / spread.std()
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std()
    return (spread - mean) / std


def generate_signals(
    z: pd.Series,
    entry: float = 2.0,
    exit: float = 0.5,
    stop: float | None = 4.0,
) -> pd.Series:
    """Convert a z-score series into positions in {-1, 0, 1}.

    Enter short the spread when z > ``entry``, long when z < -``entry``.
    Exit when |z| < ``exit``. If |z| exceeds ``stop``, go flat and stay flat
    until |z| first comes back inside ``exit``.
    """
    values = z.to_numpy(dtype=float)
    out = np.zeros(len(values))
    position = 0
    stopped = False
    for i, zt in enumerate(values):
        if np.isnan(zt):
            out[i] = position
            continue
        az = abs(zt)
        if stopped:
            if az < exit:
                stopped = False
            else:
                out[i] = 0
                continue
        if stop is not None and az > stop:
            position = 0
            stopped = True
        elif position == 0:
            if zt > entry:
                position = -1
            elif zt < -entry:
                position = 1
        elif az < exit:
            position = 0
        out[i] = position
    return pd.Series(out, index=z.index, name="position")
