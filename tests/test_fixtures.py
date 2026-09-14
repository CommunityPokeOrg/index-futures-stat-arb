import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from index_futures_stat_arb.fixtures import SyntheticBarClient
from index_futures_stat_arb.ingest.databento import normalize_bars


def test_synthetic_fixture_is_deterministic_and_respects_sessions():
    client_a = SyntheticBarClient(seed=7)
    client_b = SyntheticBarClient(seed=7)
    kwargs = dict(
        dataset="GLBX.MDP3",
        schema="ohlcv-1m",
        symbols=["ESH6", "ESM6", "NQH6", "NQM6"],
        stype_in="raw_symbol",
        start="2026-03-10",
        end="2026-03-20",
    )
    left = client_a.get_range(**kwargs)
    right = client_b.get_range(**kwargs)
    pd.testing.assert_frame_equal(left, right)
    eastern = left["ts_event"].dt.tz_convert("America/New_York")
    assert not (eastern.dt.hour == 17).any()
    assert not ((eastern.dt.weekday >= 5) & (eastern.dt.hour < 18)).any()
    assert (left["high"] >= left[["open", "close"]].max(axis=1)).all()
    assert (left["low"] <= left[["open", "close"]].min(axis=1)).all()


def test_synthetic_volume_migrates_to_next_contract():
    client = SyntheticBarClient(seed=7)
    kwargs = dict(
        dataset="GLBX.MDP3",
        schema="ohlcv-1m",
        symbols=["ESH6", "ESM6"],
        stype_in="raw_symbol",
        start="2026-03-16",
        end="2026-03-20",
    )
    frame = client.get_range(**kwargs)
    totals = frame.groupby("symbol")["volume"].sum()
    assert totals["ESH6"] < totals["ESM6"]


def test_contracts_share_product_path_and_es_nq_spread_is_stationary():
    client = SyntheticBarClient(seed=19)
    kwargs = dict(
        dataset="GLBX.MDP3",
        schema="ohlcv-1m",
        symbols=["ESH6", "ESM6", "NQH6"],
        stype_in="raw_symbol",
        start="2026-01-05",
        end="2026-01-20",
    )
    raw = client.get_range(**kwargs)
    normalized = normalize_bars(raw, 1e-9, "synthetic", "1m")
    es = normalized[normalized["product"] == "ES"]
    matched = es[es["ts_event"].isin(es.loc[es["contract"] == "ESM6", "ts_event"])]
    front = matched[matched["contract"] == "ESH6"].set_index("ts_event")["close"]
    next_contract = matched[matched["contract"] == "ESM6"].set_index("ts_event")["close"]
    ratio = (next_contract / front).dropna()
    assert ratio.std() < 1e-10
    rth = (
        normalized[normalized["is_rth"]]
        .groupby(["session_date", "product"])["close"]
        .last()
        .unstack()
    )
    spread = (rth["ES"].apply(np.log) - 0.75 * rth["NQ"].apply(np.log)).dropna()
    assert adfuller(spread, autolag="AIC")[1] < 0.05
