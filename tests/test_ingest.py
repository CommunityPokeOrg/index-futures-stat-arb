import random
import sys
import types
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from index_futures_stat_arb.config import IngestConfig
from index_futures_stat_arb.fixtures import FlakyClient, SyntheticBarClient
from index_futures_stat_arb.ingest.databento import (
    DatabentoClient,
    DataValidationError,
    IngestError,
    iter_chunks,
    normalize_bars,
    raise_on_invalid,
    read_partitioned,
    run_ingest,
    validate_bars,
    with_retries,
)
from index_futures_stat_arb.schema import sha256_file
from index_futures_stat_arb.sessions import session_bounds_utc


def _raw(ts: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_event": pd.to_datetime(ts, utc=True),
            "symbol": ["ESH6"] * len(ts),
            "instrument_id": [1] * len(ts),
            "open": [4567250000000] * len(ts),
            "high": [4567500000000] * len(ts),
            "low": [4567000000000] * len(ts),
            "close": [4567250000000] * len(ts),
            "volume": [1] * len(ts),
        }
    )


def test_databento_client_adapter_uses_wire_options(monkeypatch):
    calls: dict[str, object] = {}

    class FakeResponse:
        def to_df(self, **kwargs):
            calls["to_df"] = kwargs
            return _raw(["2026-01-05 14:30Z"])

    class FakeTimeseries:
        def get_range(self, **kwargs):
            calls["get_range"] = kwargs
            return FakeResponse()

    class FakeHistorical:
        def __init__(self, key):
            calls["key"] = key
            self.timeseries = FakeTimeseries()

    fake_module = types.ModuleType("databento")
    fake_module.__version__ = "fake-version"
    fake_module.Historical = FakeHistorical
    monkeypatch.setitem(sys.modules, "databento", fake_module)

    client = DatabentoClient("test-key")
    frame = client.get_range("GLBX.MDP3", "ohlcv-1m", ["ESH6"], "raw_symbol", "a", "b")
    assert len(frame) == 1
    assert calls["key"] == "test-key"
    assert calls["get_range"] == {
        "dataset": "GLBX.MDP3",
        "schema": "ohlcv-1m",
        "symbols": ["ESH6"],
        "stype_in": "raw_symbol",
        "start": "a",
        "end": "b",
    }
    assert calls["to_df"] == {"pretty_px": False, "pretty_ts": True}


def test_normalize_validate_and_gap_detection():
    raw = _raw(["2026-01-05 14:30Z", "2026-01-05 14:34Z"])
    normalized = normalize_bars(raw, 1e-9, "synthetic", "1m")
    assert normalized.iloc[0]["open"] == 4567.25
    assert validate_bars(normalized, "1m").n_duplicates == 0
    assert validate_bars(pd.concat([normalized, normalized]), "1m").n_duplicates == 2
    bad = normalized.copy()
    bad.loc[0, "low"] = bad.loc[0, "open"] + 1
    report = validate_bars(bad, "1m")
    assert report.n_ohlc_violations == 1
    with pytest.raises(DataValidationError):
        raise_on_invalid(report)
    assert validate_bars(normalized, "1m").gaps[0][3] == 3


def test_chunks_and_retries():
    assert iter_chunks("2026-01-01", "2026-01-05", 2) == [
        ("2026-01-01", "2026-01-03"),
        ("2026-01-03", "2026-01-05"),
    ]
    attempts = 0
    sleeps: list[float] = []

    def flaky():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError
        return "ok"

    assert with_retries(flaky, 3, 1, 30, sleeps.append, rng=random.Random(0)) == "ok"
    assert attempts == 3
    assert len(sleeps) == 2
    assert sleeps[1] >= sleeps[0]
    with pytest.raises(IngestError):
        with_retries(lambda: (_ for _ in ()).throw(ConnectionError()), 2, 0, 0, lambda _: None)


def test_run_ingest_resume_and_parquet_round_trip(tmp_path):
    cfg = IngestConfig(
        symbols=("ESH6",),
        start="2026-01-05",
        end="2026-01-08",
        data_root=tmp_path,
        source="synthetic",
    )
    underlying = SyntheticBarClient(seed=1)
    client = FlakyClient(underlying, 2)
    manifest = run_ingest(cfg, client, sleep=lambda _: None)
    assert manifest.row_count > 0
    assert all(Path(entry.path).exists() for entry in manifest.files)
    assert all(sha256_file(entry.path) == entry.sha256 for entry in manifest.files)
    first_call_count = client.calls
    next_manifest = run_ingest(cfg, client, sleep=lambda _: None)
    assert client.calls == first_call_count
    assert next_manifest.row_count == manifest.row_count
    out = read_partitioned(tmp_path, "synthetic", cfg.dataset, cfg.schema, products=["ES"])
    assert out["ts_event"].dt.tz is not None
    assert out["ts_event"].dtype == "datetime64[ns, UTC]"
    filtered = read_partitioned(
        tmp_path,
        "synthetic",
        cfg.dataset,
        cfg.schema,
        start="2026-01-07",
        end="2026-01-08",
    )
    assert filtered["session_date"].nunique() == 1
    assert filtered["session_date"].iloc[0].isoformat() == "2026-01-07"
    run_ingest(cfg, client, resume=False, sleep=lambda _: None)
    assert client.calls == first_call_count + 3


def test_session_aligned_chunks_preserve_partition_rows(tmp_path):
    cfg = IngestConfig(
        symbols=("ESH6", "ESM6"),
        start="2026-01-05",
        end="2026-01-10",
        chunk_days=1,
        data_root=tmp_path,
        source="synthetic",
    )
    chunked = SyntheticBarClient(seed=11)
    manifest = run_ingest(cfg, chunked, resume=False, sleep=lambda _: None)
    start, _ = session_bounds_utc(pd.Timestamp(cfg.start).date())
    _, end = session_bounds_utc(pd.Timestamp(cfg.end).date() - timedelta(days=1))
    direct = SyntheticBarClient(seed=11).get_range(
        cfg.dataset,
        cfg.schema,
        list(cfg.symbols),
        cfg.stype_in,
        start.isoformat(),
        end.isoformat(),
    )
    assert manifest.row_count == len(direct)
    chunk_keys = {(row.symbol, row.ts_event) for row in direct.itertuples(index=False)}
    normalized = normalize_bars(direct, cfg.price_scale, cfg.source, "1m")
    assert chunk_keys == {
        (row.contract, row.ts_event) for row in normalized.itertuples(index=False)
    }
    for entry in manifest.files:
        partition_date = Path(entry.path).parent.name.removeprefix("date=")
        partition = pq.read_table(entry.path).to_pandas()
        assert partition["session_date"].astype(str).eq(partition_date).all()
