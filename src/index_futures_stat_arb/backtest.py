"""Spread backtest, performance metrics, and walk-forward evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cointegration import estimate_hedge_ratio
from .ou import fit_ou
from .signals import compute_spread, generate_signals, zscore


@dataclass
class BacktestResult:
    positions: pd.Series
    pnl: pd.Series
    equity: pd.Series
    trades: int
    metrics: dict[str, float]


def performance_metrics(
    pnl: pd.Series, periods_per_year: int = 252
) -> dict[str, float]:
    """Total PnL, annualised Sharpe, max drawdown, hit rate, n_periods."""
    pnl = pnl.dropna()
    equity = pnl.cumsum()
    peak = equity.cummax()
    max_dd = float((peak - equity).max()) if len(equity) else 0.0
    std = float(pnl.std())
    sharpe = (
        float(pnl.mean() / std * np.sqrt(periods_per_year)) if std > 0 else 0.0
    )
    nonzero = pnl[pnl != 0]
    hit_rate = float((nonzero > 0).mean()) if len(nonzero) else 0.0
    return {
        "total_pnl": float(pnl.sum()),
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "hit_rate": hit_rate,
        "n_periods": len(pnl),
    }


def backtest_spread(
    spread: pd.Series,
    positions: pd.Series,
    cost_per_unit: float = 0.0,
    lag: int = 1,
) -> BacktestResult:
    """Backtest positions on a spread.

    pnl_t = position_{t-lag} * (spread_t - spread_{t-1}) - cost * |d_position|.
    """
    positions = positions.reindex(spread.index).fillna(0.0)
    delta_spread = spread.diff()
    held = positions.shift(lag).fillna(0.0)
    trades_cost = cost_per_unit * positions.diff().fillna(positions).abs()
    pnl = (held * delta_spread).fillna(0.0) - trades_cost
    equity = pnl.cumsum()
    changes = positions.diff().fillna(positions)
    trades = int(((changes != 0) & (positions != 0)).sum())
    return BacktestResult(
        positions=positions,
        pnl=pnl,
        equity=equity,
        trades=trades,
        metrics=performance_metrics(pnl),
    )


def train_test_split_by_date(
    df: pd.DataFrame, split: str | float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time-indexed frame by a date string or a fraction."""
    if isinstance(split, str):
        split_date = pd.Timestamp(split)
    elif isinstance(split, float):
        if not 0.0 < split < 1.0:
            raise ValueError("Fractional split must be in (0, 1)")
        split_date = df.index[int(len(df) * split)]
    else:
        raise TypeError("split must be a date string or a float fraction")
    return df.loc[df.index < split_date], df.loc[df.index >= split_date]


def walk_forward_backtest(
    prices: pd.DataFrame,
    y_col: str = "ES",
    x_col: str = "NQ",
    split: str | float = 0.7,
    z_window: int | None = 60,
    entry: float = 2.0,
    exit: float = 0.5,
    stop: float | None = 4.0,
    cost_per_unit: float = 0.0,
) -> dict:
    """Fit the hedge ratio on the train window, backtest on the test window.

    The hedge ratio and OU fit use train data only; the spread/z-score are
    computed on the full sample with the train beta and the backtest is
    restricted to the test slice.
    """
    train, test = train_test_split_by_date(prices, split)
    logp = np.log(prices)
    log_train = np.log(train)

    hedge = estimate_hedge_ratio(log_train[y_col], log_train[x_col], method="ols")
    spread = compute_spread(logp[y_col], logp[x_col], hedge.beta, hedge.alpha)
    train_spread = compute_spread(
        log_train[y_col], log_train[x_col], hedge.beta, hedge.alpha
    )
    ou = fit_ou(train_spread)

    z = zscore(spread, window=z_window)
    signals = generate_signals(z, entry=entry, exit=exit, stop=stop)

    train_bt = backtest_spread(
        spread.loc[train.index], signals.loc[train.index],
        cost_per_unit=cost_per_unit,
    )
    test_bt = backtest_spread(
        spread.loc[test.index], signals.loc[test.index],
        cost_per_unit=cost_per_unit,
    )
    return {
        "hedge": hedge,
        "ou": ou,
        "train": train_bt,
        "test": test_bt,
        "spread": spread,
        "zscore": z,
    }
