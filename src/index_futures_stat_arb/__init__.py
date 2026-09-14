"""Research scaffold for ES vs NQ index-futures statistical arbitrage."""

from .backtest import (
    BacktestResult,
    backtest_spread,
    performance_metrics,
    train_test_split_by_date,
    walk_forward_backtest,
)
from .cointegration import (
    CointegrationResult,
    HedgeRatio,
    adf_test,
    engle_granger,
    estimate_hedge_ratio,
    rolling_hedge_ratio,
)
from .data import (
    DEFAULT_TICKERS,
    PROXY_TICKERS,
    align_and_clean,
    fetch_databento,
    fetch_yfinance,
    generate_synthetic_pair,
    load_prices,
    log_prices,
)
from .ou import OUParams, fit_ou, half_life, simulate_ou
from .signals import compute_spread, generate_signals, zscore

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_TICKERS",
    "PROXY_TICKERS",
    "BacktestResult",
    "CointegrationResult",
    "HedgeRatio",
    "OUParams",
    "adf_test",
    "align_and_clean",
    "backtest_spread",
    "compute_spread",
    "engle_granger",
    "estimate_hedge_ratio",
    "fetch_databento",
    "fetch_yfinance",
    "fit_ou",
    "generate_signals",
    "generate_synthetic_pair",
    "half_life",
    "load_prices",
    "log_prices",
    "performance_metrics",
    "rolling_hedge_ratio",
    "simulate_ou",
    "train_test_split_by_date",
    "walk_forward_backtest",
    "zscore",
]
