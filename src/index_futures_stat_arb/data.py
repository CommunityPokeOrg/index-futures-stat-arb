"""Data ingestion and cleaning for the ES/NQ stat-arb research project."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

DEFAULT_TICKERS: dict[str, str] = {"ES": "ES=F", "NQ": "NQ=F"}
PROXY_TICKERS: dict[str, str] = {"ES": "SPY", "NQ": "QQQ"}


def fetch_yfinance(
    tickers: dict[str, str],
    start: str,
    end: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download adjusted-close prices via yfinance.

    Columns are the dict keys; values are ``Adj Close`` (falling back to
    ``Close`` when adjusted prices are unavailable). Rows that are entirely
    NaN are dropped.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "yfinance is required for fetch_yfinance; install with "
            '`pip install "index_futures_stat_arb[data]"` or `pip install yfinance`'
        ) from exc

    raw = yf.download(
        list(tickers.values()), start=start, end=end, interval=interval,
        auto_adjust=False, progress=False,
    )
    if raw.empty:
        return pd.DataFrame(columns=list(tickers.keys()))

    if isinstance(raw.columns, pd.MultiIndex):
        field = "Adj Close" if "Adj Close" in raw.columns.get_level_values(0) else "Close"
        prices = raw[field]
    else:  # single ticker
        field = "Adj Close" if "Adj Close" in raw.columns else "Close"
        prices = raw[[field]]
        prices.columns = list(tickers.values())

    inverse = {v: k for k, v in tickers.items()}
    prices = prices.rename(columns=inverse)
    prices = prices.reindex(columns=list(tickers.keys()))
    return prices.dropna(how="all")


def fetch_databento(
    symbols: dict[str, str],
    start: str,
    end: str | None,
    dataset: str | None = None,
    api_key: str | None = None,
    schema: str = "ohlcv-1d",
) -> pd.DataFrame:
    """Fetch daily OHLCV from Databento for continuous CME contracts.

    This path requires a valid ``DATABENTO_API_KEY`` and the optional
    ``databento`` extra, and is not covered by the offline test suite.
    """
    api_key = api_key or os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DATABENTO_API_KEY is not set; add it to .env or pass api_key="
        )
    dataset = dataset or os.environ.get("DATABENTO_DATASET") or "GLBX.MDP3"
    try:
        import databento as db
    except ImportError as exc:
        raise ImportError(
            "databento is required for fetch_databento; install with "
            '`pip install "index_futures_stat_arb[databento]"`'
        ) from exc

    try:
        client = db.Historical(api_key)
        df = client.timeseries.get_range(
            dataset=dataset,
            symbols=list(symbols.values()),
            schema=schema,
            start=start,
            end=end,
            stype_in="continuous",
        ).to_df()
    except Exception as exc:
        raise RuntimeError(f"Databento request failed: {exc}") from exc

    prices = df.reset_index().pivot_table(
        index="ts_event", columns="symbol", values="close"
    )
    inverse = {v: k for k, v in symbols.items()}
    prices = prices.rename(columns=inverse).reindex(columns=list(symbols.keys()))
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    return prices.dropna(how="all")


def load_prices(
    source: str | None = None,
    start: str = "2018-01-01",
    end: str | None = None,
    **kw,
) -> pd.DataFrame:
    """Load an ES/NQ price panel from the configured source.

    ``source`` defaults to env ``DATA_SOURCE`` or ``"yfinance"``. The yfinance
    path tries ``DEFAULT_TICKERS`` (ES=F/NQ=F) and falls back to the SPY/QQQ
    proxies before giving up.
    """
    source = source or os.environ.get("DATA_SOURCE") or "yfinance"

    if source == "databento":
        return fetch_databento(DEFAULT_TICKERS, start=start, end=end, **kw)
    if source != "yfinance":
        raise ValueError(f"Unknown data source: {source!r}")

    last_exc: Exception | None = None
    for tickers in (DEFAULT_TICKERS, PROXY_TICKERS):
        try:
            df = fetch_yfinance(tickers, start=start, end=end, **kw)
            if not df.empty:
                return df
        except Exception as exc:  # noqa: BLE001 - any fetch failure falls through to proxies
            last_exc = exc
    raise RuntimeError(
        "Could not load prices from yfinance (futures and ETF proxies both "
        "failed). For offline work, use generate_synthetic_pair() instead."
    ) from last_exc


def generate_synthetic_pair(
    n: int = 1500,
    beta: float = 1.3,
    alpha: float = 100.0,
    theta: float = 0.05,
    sigma_spread: float = 1.0,
    sigma_common: float = 0.01,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Generate a synthetic cointegrated price pair.

    log(NQ) follows a random walk; log(ES) = alpha + beta * log(NQ) + s, where
    ``s`` is an Ornstein-Uhlenbeck (mean-reverting) spread with speed ``theta``.
    """
    rng = np.random.default_rng(seed)
    log_nq = np.log(100.0) + np.cumsum(rng.normal(0.0, sigma_common, n))
    spread = np.empty(n)
    spread[0] = 0.0
    eps = rng.normal(0.0, sigma_spread * np.sqrt(2 * theta), n)
    for t in range(1, n):
        spread[t] = spread[t - 1] * (1 - theta) + eps[t]
    log_es = alpha + beta * log_nq + spread
    index = pd.bdate_range("2018-01-01", periods=n)
    return pd.DataFrame(
        {"ES": np.exp(log_es), "NQ": np.exp(log_nq)}, index=index
    )


def align_and_clean(
    df: pd.DataFrame,
    dropna: bool = True,
    ffill_limit: int = 2,
    min_price: float = 0.0,
) -> pd.DataFrame:
    """Sort index, drop duplicate timestamps, clean non-positive prices."""
    out = df.copy().sort_index()
    out = out[~out.index.duplicated(keep="first")]
    out = out.mask(out <= min_price)
    out = out.ffill(limit=ffill_limit)
    if dropna:
        out = out.dropna()
    return out


def log_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Natural log of prices."""
    return np.log(df)
