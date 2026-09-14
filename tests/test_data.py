import numpy as np
import pandas as pd
import pytest

from index_futures_stat_arb import data


def test_synthetic_pair_shape_and_columns():
    df = data.generate_synthetic_pair(n=500, seed=0)
    assert df.shape == (500, 2)
    assert list(df.columns) == ["ES", "NQ"]
    assert (df > 0).all().all()
    assert not df.isna().any().any()
    assert isinstance(df.index, pd.DatetimeIndex)


def test_align_and_clean_removes_dupes_and_nonpositive():
    idx = pd.bdate_range("2024-01-01", periods=4)
    df = pd.DataFrame(
        {"ES": [100.0, -1.0, 102.0, 103.0], "NQ": [50.0, 51.0, 0.0, 53.0]},
        index=idx,
    )
    dup = pd.concat([df, df.iloc[[0]]])
    clean = data.align_and_clean(dup, ffill_limit=2)
    assert clean.index.is_unique
    assert clean.index.is_monotonic_increasing
    assert (clean > 0).all().all()


def test_align_and_clean_limits_ffill():
    idx = pd.bdate_range("2024-01-01", periods=6)
    df = pd.DataFrame(
        {"ES": [100.0, np.nan, np.nan, np.nan, np.nan, 105.0], "NQ": [50.0] * 6},
        index=idx,
    )
    clean = data.align_and_clean(df, ffill_limit=2)
    # ES has 4 consecutive NaN, only 2 can be filled -> row dropped by dropna
    assert clean["ES"].isna().sum() == 0


def test_load_prices_falls_back_to_proxy(monkeypatch):
    calls = []

    def fake_fetch(tickers, start, end=None, **kw):
        calls.append(tickers)
        if tickers is data.DEFAULT_TICKERS:
            raise RuntimeError("no network")
        return pd.DataFrame({"ES": [1.0], "NQ": [2.0]})

    monkeypatch.setattr(data, "fetch_yfinance", fake_fetch)
    out = data.load_prices(source="yfinance", start="2020-01-01")
    assert calls == [data.DEFAULT_TICKERS, data.PROXY_TICKERS]
    assert out.shape == (1, 2)


def test_load_prices_raises_with_synthetic_hint(monkeypatch):
    def fake_fetch(tickers, start, end=None, **kw):
        raise RuntimeError("no network")

    monkeypatch.setattr(data, "fetch_yfinance", fake_fetch)
    with pytest.raises(RuntimeError, match="generate_synthetic_pair"):
        data.load_prices(source="yfinance", start="2020-01-01")


def test_log_prices():
    df = data.generate_synthetic_pair(n=50, seed=0)
    lp = data.log_prices(df)
    assert np.allclose(np.exp(lp), df)
