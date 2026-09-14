from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from index_futures_stat_arb.execution.engine import (
    LookaheadError,
    SignalState,
    SimulationConfig,
    _BarCursor,
    run_simulation,
)
from index_futures_stat_arb.execution.lookahead import assert_no_lookahead


def make_bars(
    sessions: int = 3,
    bars_per_session: int = 6,
    roll_es_at: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(sessions * bars_per_session):
        session_index = index // bars_per_session
        session = pd.Timestamp("2026-01-05").date() + pd.Timedelta(days=session_index)
        es_contract = "ESH6" if roll_es_at is None or index < roll_es_at else "ESM6"
        nq_contract = "NQH6"
        es_close = 5000.0 + index
        nq_close = 18000.0 + index
        rows.append(
            {
                "ts_event": pd.Timestamp("2026-01-05 14:30", tz="UTC")
                + pd.Timedelta(minutes=5 * index),
                "session_date": session,
                "es_open": es_close - 0.25,
                "es_high": es_close + 0.25,
                "es_low": es_close - 0.5,
                "es_close": es_close,
                "es_volume": 1,
                "nq_open": nq_close - 0.25,
                "nq_high": nq_close + 0.25,
                "nq_low": nq_close - 0.5,
                "nq_close": nq_close,
                "nq_volume": 1,
                "es_contract": es_contract,
                "nq_contract": nq_contract,
                "es_roll": roll_es_at is not None
                and index >= roll_es_at
                and index < roll_es_at + bars_per_session,
                "nq_roll": False,
                "es_close_raw": es_close,
                "nq_close_raw": nq_close,
            }
        )
    return pd.DataFrame(rows)


def config(**kwargs: object) -> SimulationConfig:
    values: dict[str, object] = {
        "start": "2026-01-05",
        "end": "2026-01-10",
        "min_hedge_sessions": 5,
    }
    values.update(kwargs)
    return SimulationConfig(**values)


def test_signal_state_entry_exit_and_flip() -> None:
    state = SignalState(2.0, 0.5, 4.0)
    assert state.update(-3.0) == 1
    assert state.last_event == "entry"
    assert state.update(3.0) == 1
    assert state.last_event is None
    state.position = 1
    assert state.update(0.0) == 0
    assert state.last_event == "exit"


def test_signal_state_stop_event() -> None:
    state = SignalState(2.0, 0.5, 4.0, position=1)
    assert state.update(5.0) == 0
    assert state.last_event == "stop"
    assert state.stopped


def test_cursor_rejects_future_access() -> None:
    bars = make_bars(1, 2)
    cursor = _BarCursor(bars, index=0)
    with pytest.raises(LookaheadError):
        cursor.at(1)


def test_entry_fills_on_next_bar_with_two_legs() -> None:
    bars = make_bars()
    signals = [0, 0, 0, 1] + [1] * (len(bars) - 4)
    result = run_simulation(bars, config(), signal_override=signals)
    entries = result.trades[result.trades["reason"] == "entry"]
    assert len(entries) == 2
    assert entries["ts_event"].nunique() == 1
    assert set(entries["product"]) == {"ES", "NQ"}


def test_fill_bar_and_next_bar_pnl() -> None:
    bars = make_bars()
    result = run_simulation(bars, config(), signal_override=[0, 0, 0, 1] + [1] * 14)
    entries = result.trades[result.trades["reason"] == "entry"].set_index("product")
    fill_index = 4
    es = entries.loc["ES"]
    nq = entries.loc["NQ"]
    expected_fill = (bars.iloc[fill_index]["es_close"] - es["fill_px"]) * 50 + (-1) * (
        bars.iloc[fill_index]["nq_close"] - nq["fill_px"]
    ) * 20
    expected_fill -= entries["fees_usd"].sum()
    assert result.pnl.iloc[fill_index] == pytest.approx(expected_fill)
    expected_next = (bars.iloc[fill_index + 1]["es_close"] - bars.iloc[fill_index]["es_close"]) * 50
    expected_next -= (
        bars.iloc[fill_index + 1]["nq_close"] - bars.iloc[fill_index]["nq_close"]
    ) * 20
    assert result.pnl.iloc[fill_index + 1] == pytest.approx(expected_next)


def test_exit_creates_one_round_trip() -> None:
    signals = [0, 0, 0, 1, 1, 1, 0] + [0] * 11
    result = run_simulation(make_bars(), config(), signal_override=signals)
    assert (result.trades["reason"] == "exit").sum() == 2
    assert result.metrics["n_round_trips"] == 1


def test_roll_only_changed_leg_and_contract_labels() -> None:
    roll_at = 6
    result = run_simulation(
        make_bars(3, 6, roll_at),
        config(),
        signal_override=[0, 0, 0, 1] + [1] * 14,
    )
    rolls = result.trades[result.trades["reason"] == "roll"]
    assert len(rolls) == 2
    assert set(rolls["product"]) == {"ES"}
    assert set(rolls["contract"]) == {"ESH6", "ESM6"}
    assert rolls.iloc[0]["contract"] == "ESH6"
    assert rolls.iloc[1]["contract"] == "ESM6"


def test_final_bar_is_flattened() -> None:
    result = run_simulation(
        make_bars(),
        config(),
        signal_override=[0, 0, 0, 1] + [1] * 14,
    )
    assert result.positions.iloc[-1][["n_es", "n_nq"]].tolist() == [0, 0]
    assert (result.trades["reason"] == "eod_final").sum() == 2


def test_unchanged_signal_does_not_rebalance() -> None:
    result = run_simulation(
        make_bars(),
        config(),
        signal_override=[0, 0, 0, 1] + [1] * 14,
    )
    assert len(result.trades[result.trades["reason"] == "entry"]) == 2
    assert not (result.trades["reason"] == "flip").any()


def test_simulation_is_deterministic() -> None:
    bars = make_bars()
    one = run_simulation(bars, config(), signal_override=[0, 0, 0, 1] + [1] * 14)
    two = run_simulation(bars, config(), signal_override=[0, 0, 0, 1] + [1] * 14)
    pd.testing.assert_frame_equal(one.trades, two.trades)


def test_lookahead_utility_on_small_bars() -> None:
    bars = make_bars(3, 6)
    assert_no_lookahead(
        bars,
        config(min_hedge_sessions=1, hedge_lookback_sessions=2),
        cut_index=10,
        rng=None,
    )


def _spread_bars(spread: np.ndarray, bars_per_session: int = 10) -> pd.DataFrame:
    rows = []
    for index, value in enumerate(spread):
        nq = 18_000.0 + np.sin(index / 4.0) * 100.0
        es = float(np.exp(np.log(nq) * 0.75 + value))
        session = pd.Timestamp("2026-01-05").date() + pd.Timedelta(days=index // bars_per_session)
        rows.append(
            {
                "ts_event": pd.Timestamp("2026-01-05", tz="UTC") + pd.Timedelta(minutes=5 * index),
                "session_date": session,
                "es_open": es,
                "es_high": es + 1,
                "es_low": es - 1,
                "es_close": es,
                "es_volume": 1,
                "nq_open": nq,
                "nq_high": nq + 1,
                "nq_low": nq - 1,
                "nq_close": nq,
                "nq_volume": 1,
                "es_contract": "ESH6",
                "nq_contract": "NQH6",
                "es_roll": False,
                "nq_roll": False,
                "es_close_raw": es,
                "nq_close_raw": nq,
            }
        )
    return pd.DataFrame(rows)


def test_ou_threshold_blocks_random_walk_and_accepts_ou() -> None:
    rng = np.random.default_rng(4)
    random_walk = np.arange(100, dtype=float) * 0.01
    ou_spread = np.zeros(100)
    for index in range(1, len(ou_spread)):
        ou_spread[index] = 0.85 * ou_spread[index - 1] + rng.normal(0, 0.03)
    ou_spread[60:] += 0.15
    cfg = config(
        min_hedge_sessions=2,
        hedge_lookback_sessions=5,
        z_window=30,
        z_reset_each_session=False,
        threshold_mode="ou",
        ou_min_obs=20,
        entry=1.0,
        exit=0.2,
    )
    random_bars = _spread_bars(random_walk)
    random_bars[["nq_open", "nq_high", "nq_low", "nq_close"]] = [
        18_000.0,
        18_001.0,
        17_999.0,
        18_000.0,
    ]
    random_result = run_simulation(random_bars, cfg)
    ou_result = run_simulation(_spread_bars(ou_spread), cfg)
    assert len(random_result.trades) == 0
    assert ou_result.signals["half_life"].notna().any()
    assert len(ou_result.trades) > 0


def test_time_stop_has_distinct_reason() -> None:
    rng = np.random.default_rng(4)
    spread = np.zeros(100)
    for index in range(1, len(spread)):
        spread[index] = 0.85 * spread[index - 1] + rng.normal(0, 0.01)
    spread[60:] += 0.15
    cfg = config(
        min_hedge_sessions=2,
        hedge_lookback_sessions=5,
        z_window=30,
        z_reset_each_session=False,
        threshold_mode="ou",
        ou_min_obs=20,
        entry=1.0,
        exit=0.2,
        max_holding_half_lives=0.5,
    )
    result = run_simulation(_spread_bars(spread), cfg)
    assert "time_stop" in set(result.trades["reason"])


def test_time_stop_is_unset_by_default() -> None:
    rng = np.random.default_rng(4)
    spread = np.zeros(100)
    for index in range(1, len(spread)):
        spread[index] = 0.85 * spread[index - 1] + rng.normal(0, 0.01)
    spread[60:] += 0.15
    result = run_simulation(
        _spread_bars(spread),
        config(
            min_hedge_sessions=2,
            hedge_lookback_sessions=5,
            z_window=30,
            z_reset_each_session=False,
            threshold_mode="ou",
            ou_min_obs=20,
            entry=1.0,
            exit=0.2,
        ),
    )
    assert "time_stop" not in set(result.trades["reason"])


def test_ols_residual_history_recomputed_after_refit() -> None:
    bars = _spread_bars(np.sin(np.arange(80) / 3.0) * 0.01)
    result = run_simulation(
        bars,
        config(
            min_hedge_sessions=2,
            hedge_lookback_sessions=3,
            z_window=12,
            z_reset_each_session=False,
        ),
    )
    assert result.signals["z"].notna().any()
    assert result.hedge_history["beta"].notna().all()
