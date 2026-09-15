"""Keyless Yahoo Finance ingestion for futures, ETFs, rates, and dividends."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..schema import BARS_SCHEMA, sha256_file
from ..sessions import is_rth, session_date
from .databento import (
    raise_on_invalid,
    validate_bars,
    with_retries,
)

YAHOO_SYMBOLS: dict[str, str] = {
    "ES": "ES=F",
    "NQ": "NQ=F",
    "SPY": "SPY",
    "QQQ": "QQQ",
    "IRX": "^IRX",
}


@dataclass(frozen=True)
class IntervalLimit:
    max_history_days: int | None
    max_span_days: int | None


YAHOO_INTERVAL_LIMITS: dict[str, IntervalLimit] = {
    "1m": IntervalLimit(30, 7),
    "2m": IntervalLimit(60, 60),
    "5m": IntervalLimit(60, 60),
    "15m": IntervalLimit(60, 60),
    "30m": IntervalLimit(60, 60),
    "60m": IntervalLimit(730, 730),
    "1h": IntervalLimit(730, 730),
    "1d": IntervalLimit(None, None),
}


class YahooError(RuntimeError):
    """Raised when Yahoo returns unusable or unavailable data."""


Downloader = Callable[[str, date, date, str], pd.DataFrame]


def default_downloader(symbol: str, start: date, end: date, interval: str) -> pd.DataFrame:
    """Download one Yahoo symbol without requiring an API key."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "yfinance is required for Yahoo ingestion; install with "
            '`pip install "index_futures_stat_arb[data]"` or pip install yfinance'
        ) from exc
    return yf.download(
        symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )


def _limits(interval: str) -> IntervalLimit:
    try:
        return YAHOO_INTERVAL_LIMITS[interval]
    except KeyError as exc:
        raise ValueError(f"unsupported Yahoo interval: {interval!r}") from exc


def clip_to_retention(
    start: date, end: date, interval: str, today: date
) -> tuple[date, date, bool]:
    """Clip a request to Yahoo's documented history retention window."""
    limit = _limits(interval)
    effective_start = start
    if limit.max_history_days is not None:
        effective_start = max(start, today - timedelta(days=limit.max_history_days - 1))
    if end <= effective_start:
        raise YahooError(
            f"Yahoo {interval} retention has no data for {start.isoformat()}..{end.isoformat()}"
        )
    return effective_start, end, effective_start != start


def iter_request_windows(start: date, end: date, interval: str) -> Iterator[tuple[date, date]]:
    """Yield half-open requests within Yahoo's maximum span."""
    span = _limits(interval).max_span_days
    if span is None:
        yield start, end
        return
    current = start
    while current < end:
        window_end = min(current + timedelta(days=span), end)
        yield current, window_end
        current = window_end


def _empty_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            name: pd.Series(dtype=_dtype_for_schema(field.type))
            for name, field in zip(BARS_SCHEMA.names, BARS_SCHEMA, strict=True)
        }
    )


def _dtype_for_schema(dtype: Any) -> str:
    text = str(dtype)
    if text.startswith("timestamp"):
        return "datetime64[ns, UTC]"
    if text == "date32[day]":
        return "object"
    if text == "bool":
        return "bool"
    if text.startswith(("int", "uint")):
        return "int64"
    if text.startswith(("float", "double")):
        return "float64"
    return "string"


def _field_columns(raw: pd.DataFrame) -> dict[str, pd.Series]:
    fields = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    if not isinstance(raw.columns, pd.MultiIndex):
        lowered = {str(column).lower(): column for column in raw.columns}
        return {
            key: raw[lowered[value.lower()]]
            for key, value in fields.items()
            if value.lower() in lowered
        }
    selected: dict[str, pd.Series] = {}
    for field, display in fields.items():
        for column in raw.columns:
            parts = {str(part).lower() for part in column}
            if display.lower() in parts:
                selected[field] = raw[column]
                break
    return selected


def normalize_yahoo(raw: pd.DataFrame, product: str, interval: str) -> pd.DataFrame:
    """Normalize flat or MultiIndex Yahoo output to the canonical bar schema."""
    if product not in YAHOO_SYMBOLS:
        raise ValueError(f"unsupported Yahoo product: {product!r}")
    _limits(interval)
    if raw.empty:
        return _empty_bars()
    columns = _field_columns(raw)
    missing = {"open", "high", "low", "close", "volume"} - set(columns)
    if missing:
        raise YahooError(f"Yahoo response is missing columns: {sorted(missing)}")
    index = pd.DatetimeIndex(pd.to_datetime(raw.index))
    if interval == "1d":
        days = pd.Index(index.date)
        timestamps = pd.DatetimeIndex(days).tz_localize("UTC") + pd.Timedelta(hours=21)
        sessions: object = days.to_numpy()
    else:
        if index.tz is None:
            index = index.tz_localize("America/New_York")
        timestamps = index.tz_convert("UTC")
        sessions = session_date(timestamps)
    out = pd.DataFrame(
        {
            "ts_event": timestamps,
            "ts_recv": pd.NaT,
            "source": "yahoo",
            "product": product,
            "contract": YAHOO_SYMBOLS[product],
            "instrument_id": 0,
            "open": pd.to_numeric(columns["open"], errors="coerce").to_numpy(),
            "high": pd.to_numeric(columns["high"], errors="coerce").to_numpy(),
            "low": pd.to_numeric(columns["low"], errors="coerce").to_numpy(),
            "close": pd.to_numeric(columns["close"], errors="coerce").to_numpy(),
            "volume": pd.to_numeric(columns["volume"], errors="coerce")
            .fillna(0)
            .astype("int64")
            .to_numpy(),
            "interval": interval,
            "session_date": sessions,
            "is_rth": is_rth(timestamps) if interval != "1d" else False,
        }
    )
    out = out.dropna(subset=["close"])
    out = out.drop_duplicates("ts_event").sort_values("ts_event").reset_index(drop=True)
    out["ts_event"] = pd.to_datetime(out["ts_event"], utc=True)
    out["ts_recv"] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    out["source"] = out["source"].astype("string")
    out["product"] = out["product"].astype("string")
    out["contract"] = out["contract"].astype("string")
    out["interval"] = out["interval"].astype("string")
    out["session_date"] = pd.to_datetime(out["session_date"]).dt.date
    out["is_rth"] = out["is_rth"].astype(bool)
    out["instrument_id"] = out["instrument_id"].astype("int64")
    return out[list(BARS_SCHEMA.names)]


def fetch_yahoo_bars(
    product: str,
    start: date,
    end: date,
    interval: str,
    *,
    cache_dir: Path,
    downloader: Downloader | None = None,
    use_cache: bool = True,
    max_retries: int = 3,
    backoff_base_s: float = 1.0,
    backoff_max_s: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
    today: date | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch, validate, and cache Yahoo bars with retention-aware metadata."""
    if product not in YAHOO_SYMBOLS:
        raise ValueError(f"unsupported Yahoo product: {product!r}")
    effective_start, effective_end, clipped = clip_to_retention(
        start, end, interval, today or datetime.now(timezone.utc).date()
    )
    destination = (
        Path(cache_dir)
        / "yahoo"
        / f"interval={interval}"
        / f"product={product}"
        / f"{effective_start}_{effective_end}.parquet"
    )
    sidecar = destination.with_suffix(".json")
    if use_cache and destination.exists() and sidecar.exists():
        frame = pd.read_parquet(destination)
        cached_metadata = json.loads(sidecar.read_text())
        cached_metadata["cache_hit"] = True
        return frame, cached_metadata
    frames: list[pd.DataFrame] = []
    active_downloader = downloader or default_downloader
    for window_start, window_end in iter_request_windows(effective_start, effective_end, interval):

        def fetch_window(
            window_start: date = window_start, window_end: date = window_end
        ) -> pd.DataFrame:
            result = active_downloader(YAHOO_SYMBOLS[product], window_start, window_end, interval)
            if result.empty:
                raise YahooError(
                    f"Yahoo returned no data for {product} {window_start}..{window_end}"
                )
            return result

        try:
            frames.append(
                with_retries(
                    fetch_window,
                    max_retries,
                    backoff_base_s,
                    backoff_max_s,
                    sleep=sleep,
                    retry_on=(ConnectionError, TimeoutError, RuntimeError, YahooError),
                )
            )
        except Exception as exc:
            if isinstance(exc, YahooError):
                raise
            raise YahooError(f"Yahoo request failed for {product}") from exc
    frame = normalize_yahoo(pd.concat(frames), product, interval)
    report = validate_bars(frame, interval)
    raise_on_invalid(report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
    metadata: dict[str, Any] = {
        "symbol": YAHOO_SYMBOLS[product],
        "product": product,
        "interval": interval,
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "effective_start": effective_start.isoformat(),
        "effective_end": effective_end.isoformat(),
        "clipped": clipped,
        "rows": len(frame),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "cache_hit": False,
        "cache_path": str(destination),
        "sha256": sha256_file(destination),
        "n_gaps": len(report.gaps),
    }
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return frame, metadata


def fetch_yahoo_rate(
    symbol_key: str,
    start: date,
    end: date,
    *,
    cache_dir: Path,
    downloader: Downloader | None = None,
    use_cache: bool = True,
    max_retries: int = 3,
    backoff_base_s: float = 1.0,
    backoff_max_s: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
    today: date | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    """Fetch a daily Yahoo rate series without applying price validation."""
    if symbol_key not in YAHOO_SYMBOLS:
        raise ValueError(f"unsupported Yahoo product: {symbol_key!r}")
    interval = "1d"
    effective_start, effective_end, clipped = clip_to_retention(
        start, end, interval, today or datetime.now(timezone.utc).date()
    )
    destination = (
        Path(cache_dir)
        / "yahoo"
        / f"interval={interval}"
        / f"product={symbol_key}"
        / "kind=rate"
        / f"{effective_start}_{effective_end}.parquet"
    )
    sidecar = destination.with_suffix(".json")
    if use_cache and destination.exists() and sidecar.exists():
        frame = pd.read_parquet(destination)
        values = pd.Series(
            frame["rate"].to_numpy(dtype=float),
            index=pd.to_datetime(frame["session_date"]).dt.date,
        )
        return values, json.loads(sidecar.read_text()) | {"cache_hit": True}

    frames: list[pd.DataFrame] = []
    active_downloader = downloader or default_downloader
    for window_start, window_end in iter_request_windows(effective_start, effective_end, interval):

        def fetch_window(
            window_start: date = window_start, window_end: date = window_end
        ) -> pd.DataFrame:
            result = active_downloader(
                YAHOO_SYMBOLS[symbol_key], window_start, window_end, interval
            )
            if result.empty:
                raise YahooError(
                    f"Yahoo returned no data for {symbol_key} {window_start}..{window_end}"
                )
            return result

        try:
            frames.append(
                with_retries(
                    fetch_window,
                    max_retries,
                    backoff_base_s,
                    backoff_max_s,
                    sleep=sleep,
                    retry_on=(ConnectionError, TimeoutError, RuntimeError, YahooError),
                )
            )
        except Exception as exc:
            if isinstance(exc, YahooError):
                raise
            raise YahooError(f"Yahoo request failed for {symbol_key}") from exc

    raw = pd.concat(frames)
    columns = _field_columns(raw)
    if "close" not in columns:
        raise YahooError("Yahoo response is missing the close column")
    index = pd.DatetimeIndex(pd.to_datetime(raw.index))
    if index.has_duplicates:
        raise YahooError(f"Yahoo response contains duplicate timestamps for {symbol_key}")
    if not index.is_monotonic_increasing:
        raise YahooError(f"Yahoo response is not sorted for {symbol_key}")
    close = pd.to_numeric(columns["close"], errors="coerce")
    if close.isna().any():
        raise YahooError(f"Yahoo response contains NaN rates for {symbol_key}")
    frame = normalize_yahoo(raw, symbol_key, interval)
    values = frame.set_index("session_date")["close"].astype(float).groupby(level=0).last() / 100.0
    rate_frame = pd.DataFrame({"session_date": values.index, "rate": values.to_numpy()})
    destination.parent.mkdir(parents=True, exist_ok=True)
    rate_frame.to_parquet(destination, index=False)
    metadata: dict[str, Any] = {
        "symbol": YAHOO_SYMBOLS[symbol_key],
        "product": symbol_key,
        "interval": interval,
        "kind": "rate",
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "effective_start": effective_start.isoformat(),
        "effective_end": effective_end.isoformat(),
        "clipped": clipped,
        "rows": len(values),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "cache_hit": False,
        "cache_path": str(destination),
        "sha256": sha256_file(destination),
        "rate_min": float(values.min()),
        "rate_max": float(values.max()),
    }
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return values, metadata


def fetch_yahoo_dividends(
    product: str,
    *,
    cache_dir: Path,
    downloader: Callable[[str], pd.Series] | None = None,
    use_cache: bool = True,
) -> tuple[pd.Series, dict[str, Any]]:
    """Fetch and cache Yahoo per-share dividends by ex-date."""
    if product not in YAHOO_SYMBOLS:
        raise ValueError(f"unsupported Yahoo product: {product!r}")
    destination = Path(cache_dir) / "yahoo" / "dividends" / f"product={product}.parquet"
    sidecar = destination.with_suffix(".json")
    if use_cache and destination.exists() and sidecar.exists():
        frame = pd.read_parquet(destination)
        values = pd.Series(frame["dividend"].to_numpy(), index=pd.to_datetime(frame["date"]))
        return values, json.loads(sidecar.read_text()) | {"cache_hit": True}
    if downloader is None:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover
            raise ImportError("yfinance is required to fetch dividends") from exc
        values = yf.Ticker(YAHOO_SYMBOLS[product]).dividends
    else:
        values = downloader(YAHOO_SYMBOLS[product])
    values = pd.Series(values, dtype=float)
    values.index = pd.to_datetime(values.index).tz_localize(None).normalize()
    values = values[~values.index.duplicated(keep="last")].sort_index()
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": values.index, "dividend": values.to_numpy()}).to_parquet(
        destination, index=False
    )
    metadata = {
        "symbol": YAHOO_SYMBOLS[product],
        "product": product,
        "rows": len(values),
        "cache_hit": False,
        "cache_path": str(destination),
    }
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return values, metadata


def build_pair_bars(
    a: pd.DataFrame,
    b: pd.DataFrame,
    *,
    bar_minutes: int | None = None,
    rth_only: bool = False,
    carry: pd.Series | None = None,
) -> pd.DataFrame:
    """Build the execution engine's aligned pair-bar frame from Yahoo bars."""
    from ..execution.engine import _resample

    def prepare(frame: pd.DataFrame, product: str) -> pd.DataFrame:
        out = frame.copy()
        if "volume" not in out:
            out["volume"] = 0
        out["roll_flag"] = False
        is_intraday = not out.empty and str(out["interval"].iloc[0]) != "1d"
        if is_intraday and bar_minutes is not None:
            out = _resample(out, bar_minutes, rth_only, product)
        elif rth_only and is_intraday:
            out = out[out["is_rth"]]
        return out

    product_a = str(a["product"].iloc[0])
    product_b = str(b["product"].iloc[0])
    a_frame = prepare(a, product_a)
    b_frame = prepare(b, product_b)
    a_frame["a_close_raw"] = a_frame["close"]
    b_frame["b_close_raw"] = b_frame["close"]
    merged = a_frame.merge(b_frame, on="ts_event", suffixes=("_a", "_b"))
    merged["session_date"] = merged["session_date_a"]
    merged["carry"] = 0.0
    if carry is not None:
        carry_values = pd.Series(carry, dtype=float)
        carry_values.index = pd.to_datetime(carry_values.index).date
        merged["carry"] = merged["session_date"].map(carry_values).fillna(0.0)
    result = (
        merged[
            [
                "ts_event",
                "session_date",
                "open_a",
                "high_a",
                "low_a",
                "close_a",
                "volume_a",
                "open_b",
                "high_b",
                "low_b",
                "close_b",
                "volume_b",
                "contract_a",
                "contract_b",
                "roll_flag_a",
                "roll_flag_b",
                "a_close_raw",
                "b_close_raw",
                "carry",
            ]
        ]
        .rename(
            columns={
                "open_a": "a_open",
                "high_a": "a_high",
                "low_a": "a_low",
                "close_a": "a_close",
                "volume_a": "a_volume",
                "contract_a": "a_contract",
                "roll_flag_a": "a_roll",
                "open_b": "b_open",
                "high_b": "b_high",
                "low_b": "b_low",
                "close_b": "b_close",
                "volume_b": "b_volume",
                "contract_b": "b_contract",
                "roll_flag_b": "b_roll",
            }
        )
        .sort_values("ts_event")
        .reset_index(drop=True)
    )
    if (product_a, product_b) == ("ES", "NQ"):
        result = result.rename(
            columns={
                "a_open": "es_open",
                "a_high": "es_high",
                "a_low": "es_low",
                "a_close": "es_close",
                "a_volume": "es_volume",
                "b_open": "nq_open",
                "b_high": "nq_high",
                "b_low": "nq_low",
                "b_close": "nq_close",
                "b_volume": "nq_volume",
                "a_contract": "es_contract",
                "b_contract": "nq_contract",
                "a_roll": "es_roll",
                "b_roll": "nq_roll",
                "a_close_raw": "es_close_raw",
                "b_close_raw": "nq_close_raw",
            }
        )
        return result.drop(columns=["carry"])
    return result
