from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from index_futures_stat_arb.cli import main
from index_futures_stat_arb.config import load_simulation_config, load_yahoo_config
from index_futures_stat_arb.execution.engine import SimulationConfig, run_simulation
from index_futures_stat_arb.ingest.yahoo import (
    YAHOO_SYMBOLS,
    YahooError,
    build_pair_bars,
    clip_to_retention,
    fetch_yahoo_bars,
    fetch_yahoo_rate,
    iter_request_windows,
    normalize_yahoo,
)


def _daily_frame(start: date, end: date, offset: float = 0.0) -> pd.DataFrame:
    index = pd.date_range(start, end - timedelta(days=1), freq="B")
    close = 5000.0 + np.arange(len(index), dtype=float) + offset
    return pd.DataFrame(
        {
            "Open": close - 1,
            "High": close + 2,
            "Low": close - 2,
            "Close": close,
            "Adj Close": close,
            "Volume": np.full(len(index), 1000),
        },
        index=index,
    )


def test_normalize_daily_flat_and_multiindex() -> None:
    flat = normalize_yahoo(_daily_frame(date(2026, 1, 1), date(2026, 1, 5)), "ES", "1d")
    assert list(flat.columns) == [
        "ts_event",
        "ts_recv",
        "source",
        "product",
        "contract",
        "instrument_id",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "interval",
        "session_date",
        "is_rth",
    ]
    assert flat["ts_event"].dt.tz is not None
    assert flat.iloc[0]["ts_event"].hour == 21
    multi = _daily_frame(date(2026, 1, 1), date(2026, 1, 5))
    multi.columns = pd.MultiIndex.from_product([["ES=F"], multi.columns])
    normalized = normalize_yahoo(multi, "ES", "1d")
    assert normalized["contract"].unique().tolist() == [YAHOO_SYMBOLS["ES"]]


def test_normalize_intraday_converts_timezone_and_session() -> None:
    index = pd.date_range("2026-01-05 09:30", periods=2, freq="5min", tz="America/New_York")
    raw = pd.DataFrame(
        {
            "Open": [5000.0, 5001.0],
            "High": [5001.0, 5002.0],
            "Low": [4999.0, 5000.0],
            "Close": [5000.5, 5001.5],
            "Volume": [10, 12],
        },
        index=index,
    )
    normalized = normalize_yahoo(raw, "ES", "5m")
    assert str(normalized.iloc[0]["ts_event"].tz) == "UTC"
    assert normalized["session_date"].tolist() == [date(2026, 1, 5)] * 2
    assert normalized["is_rth"].all()


def test_retention_clipping_and_expired_range() -> None:
    today = date(2026, 9, 14)
    assert clip_to_retention(date(2026, 8, 1), today, "1m", today) == (
        date(2026, 8, 16),
        today,
        True,
    )
    assert clip_to_retention(date(2015, 1, 1), today, "1d", today)[2] is False
    with pytest.raises(YahooError):
        clip_to_retention(date(2026, 8, 1), date(2026, 8, 5), "1m", today)


def test_request_windows_cover_range() -> None:
    windows = list(iter_request_windows(date(2026, 1, 1), date(2026, 1, 21), "1m"))
    assert windows == [
        (date(2026, 1, 1), date(2026, 1, 8)),
        (date(2026, 1, 8), date(2026, 1, 15)),
        (date(2026, 1, 15), date(2026, 1, 21)),
    ]


def test_retry_empty_download_and_cache(tmp_path: Path) -> None:
    calls = 0
    sleeps: list[float] = []

    def flaky(symbol: str, start: date, end: date, interval: str) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ConnectionError("temporary")
        return _daily_frame(start, end)

    frame, metadata = fetch_yahoo_bars(
        "ES",
        date(2026, 1, 1),
        date(2026, 1, 10),
        "1d",
        cache_dir=tmp_path,
        downloader=flaky,
        sleep=sleeps.append,
        backoff_base_s=0,
        backoff_max_s=0,
    )
    assert len(frame) > 0
    assert calls == 3
    assert len(sleeps) == 2
    cached, cached_meta = fetch_yahoo_bars(
        "ES",
        date(2026, 1, 1),
        date(2026, 1, 10),
        "1d",
        cache_dir=tmp_path,
        downloader=lambda *_: pytest.fail("cache miss"),
    )
    pd.testing.assert_frame_equal(frame, cached)
    assert cached_meta["cache_hit"] is True
    assert metadata["sha256"]

    with pytest.raises(YahooError):
        fetch_yahoo_bars(
            "NQ",
            date(2026, 1, 1),
            date(2026, 1, 10),
            "1d",
            cache_dir=tmp_path,
            downloader=lambda *_: pd.DataFrame(),
            sleep=lambda _: None,
            backoff_base_s=0,
            backoff_max_s=0,
        )


def test_invalid_ohlc_is_rejected(tmp_path: Path) -> None:
    def invalid(*_: object) -> pd.DataFrame:
        frame = _daily_frame(date(2026, 1, 1), date(2026, 1, 4))
        frame.loc[frame.index[0], "High"] = frame.loc[frame.index[0], "Low"] - 1
        return frame

    with pytest.raises(ValueError, match="bar validation failed"):
        fetch_yahoo_bars(
            "ES",
            date(2026, 1, 1),
            date(2026, 1, 4),
            "1d",
            cache_dir=tmp_path,
            downloader=invalid,
        )


def test_fetch_yahoo_rate_preserves_nonpositive_rates(tmp_path: Path) -> None:
    def rate_downloader(symbol: str, start: date, end: date, interval: str) -> pd.DataFrame:
        frame = _daily_frame(start, end)
        frame["Close"] = np.linspace(0.0, -0.02, len(frame))
        return frame

    values, metadata = fetch_yahoo_rate(
        "IRX",
        date(2026, 1, 1),
        date(2026, 1, 10),
        cache_dir=tmp_path,
        downloader=rate_downloader,
        backoff_base_s=0,
        backoff_max_s=0,
    )
    assert values.iloc[0] == 0.0
    assert values.iloc[-1] == pytest.approx(-0.0002)
    assert metadata["kind"] == "rate"
    assert "kind=rate" in metadata["cache_path"]


def test_yahoo_carry_fallback_records_reason(tmp_path: Path, monkeypatch) -> None:
    from index_futures_stat_arb.ingest import yahoo

    def fake_downloader(symbol: str, start: date, end: date, interval: str) -> pd.DataFrame:
        return _daily_frame(start, end, 1000.0 if symbol == "SPY" else 0.0)

    def failed_rate(*_: object, **__: object) -> tuple[pd.Series, dict]:
        raise yahoo.YahooError("invalid rate cache")

    monkeypatch.setattr(yahoo, "default_downloader", fake_downloader)
    monkeypatch.setattr(yahoo, "fetch_yahoo_rate", failed_rate)
    monkeypatch.setattr(
        yahoo,
        "fetch_yahoo_dividends",
        lambda *_args, **_kwargs: (
            pd.Series([0.1], index=pd.DatetimeIndex(["2025-01-02"])),
            {"cache_hit": True},
        ),
    )
    config = tmp_path / "config.toml"
    config.write_text(
        """
[simulation]
products = ["ES", "SPY"]
start = "2025-01-01"
end = "2026-03-01"
bar_minutes = 1
rth_only = false
z_window = 20
hedge_lookback_sessions = 10
min_hedge_sessions = 5
z_reset_each_session = false
adjust = "none"

[yahoo]
interval = "1d"
cache_dir = "CACHE"
carry_adjust = true
fallback_risk_free_rate = 0.04
fallback_dividend_yield = 0.012
""".replace("CACHE", str(tmp_path / "cache"))
    )
    assert main(["simulate-yahoo", "--config", str(config), "--out", str(tmp_path)]) == 0
    report = json.loads(next(tmp_path.glob("*/report.json")).read_text())
    assert report["data_meta"]["carry_source"] == "fallback_constant"
    assert "invalid rate cache" in report["data_meta"]["fallback_reason"]


def test_build_pair_bars_daily_join() -> None:
    es = normalize_yahoo(_daily_frame(date(2026, 1, 1), date(2026, 1, 10)), "ES", "1d")
    nq = normalize_yahoo(_daily_frame(date(2026, 1, 1), date(2026, 1, 10), 13000), "NQ", "1d")
    pair = build_pair_bars(es, nq)
    assert len(pair) == len(es)
    assert set(pair.columns) == {
        "ts_event",
        "session_date",
        "es_open",
        "es_high",
        "es_low",
        "es_close",
        "es_volume",
        "nq_open",
        "nq_high",
        "nq_low",
        "nq_close",
        "nq_volume",
        "es_contract",
        "nq_contract",
        "es_roll",
        "nq_roll",
        "es_close_raw",
        "nq_close_raw",
    }


def test_daily_engine_keeps_z_history_when_not_reset() -> None:
    es = normalize_yahoo(_daily_frame(date(2025, 1, 1), date(2026, 3, 1)), "ES", "1d")
    nq = normalize_yahoo(_daily_frame(date(2025, 1, 1), date(2026, 3, 1), 13000), "NQ", "1d")
    oscillation = 20.0 * np.sin(np.arange(len(es)) / 8.0)
    es["close"] = 5000.0 + oscillation
    es["open"] = es["close"]
    es["high"] = es["close"] + 1.0
    es["low"] = es["close"] - 1.0
    pair = build_pair_bars(es, nq)
    result = run_simulation(
        pair,
        SimulationConfig(
            start="2025-01-01",
            end="2026-03-01",
            bar_minutes=1,
            rth_only=False,
            z_window=60,
            hedge_lookback_sessions=20,
            min_hedge_sessions=5,
            z_reset_each_session=False,
            adjust="none",
        ),
    )
    assert result.signals["z"].iloc[30:].notna().any()
    assert result.metrics["n_trades"] > 0


def test_load_yahoo_configs() -> None:
    daily = load_yahoo_config("configs/sim_yahoo_daily.toml")
    simulation = load_simulation_config("configs/sim_yahoo_daily.toml")
    assert daily.interval == "1d"
    assert simulation.z_reset_each_session is False


def test_simulate_yahoo_cli_with_monkeypatched_downloader(tmp_path: Path, monkeypatch) -> None:
    from index_futures_stat_arb.ingest import yahoo

    def fake_downloader(symbol: str, start: date, end: date, interval: str) -> pd.DataFrame:
        offset = 0 if symbol == "ES=F" else 13000
        return _daily_frame(start, end, offset)

    monkeypatch.setattr(yahoo, "default_downloader", fake_downloader)
    config = tmp_path / "config.toml"
    config.write_text(
        """
[simulation]
start = "2025-01-01"
end = "2026-03-01"
bar_minutes = 1
rth_only = false
z_window = 20
hedge_lookback_sessions = 10
min_hedge_sessions = 5
z_reset_each_session = false
adjust = "none"

[yahoo]
interval = "1d"
cache_dir = "CACHE"
""".replace("CACHE", str(tmp_path / "cache"))
    )
    assert main(["simulate-yahoo", "--config", str(config), "--out", str(tmp_path)]) == 0
    reports = list(tmp_path.glob("*/report.json"))
    report = json.loads(reports[0].read_text())
    assert report["data_source"] == "yahoo"
    assert (reports[0].parent / "trades.csv").exists()


@pytest.mark.network
def test_live_yahoo_daily_smoke() -> None:
    frame, metadata = fetch_yahoo_bars(
        "ES",
        date.today() - timedelta(days=14),
        date.today(),
        "1d",
        cache_dir=Path("data/yahoo"),
        use_cache=False,
    )
    assert not frame.empty
    assert metadata["symbol"] == "ES=F"
