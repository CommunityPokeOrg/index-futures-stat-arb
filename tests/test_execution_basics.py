import pandas as pd

from index_futures_stat_arb.contracts import PRODUCTS
from index_futures_stat_arb.execution.costs import CostModel
from index_futures_stat_arb.execution.engine import SimulationConfig, run_simulation
from index_futures_stat_arb.execution.sizing import (
    DollarNeutralSizer,
    FixedContracts,
    SizerState,
    VolTargetSizer,
)


def _bars() -> pd.DataFrame:
    rows = []
    for index in range(10):
        session = (
            pd.Timestamp("2026-01-05").date() if index < 5 else pd.Timestamp("2026-01-06").date()
        )
        rows.append(
            {
                "ts_event": pd.Timestamp("2026-01-05 14:30", tz="UTC")
                + pd.Timedelta(minutes=5 * index),
                "session_date": session,
                "es_open": 5000 + index,
                "es_high": 5001 + index,
                "es_low": 4999 + index,
                "es_close": 5000.5 + index,
                "es_volume": 1,
                "nq_open": 18000 + index,
                "nq_high": 18001 + index,
                "nq_low": 17999 + index,
                "nq_close": 18000.5 + index,
                "nq_volume": 1,
                "es_contract": "ESH6",
                "nq_contract": "NQH6",
                "es_roll": False,
                "nq_roll": False,
                "es_close_raw": 5000.5 + index,
                "nq_close_raw": 18000.5 + index,
            }
        )
    return pd.DataFrame(rows)


def test_costs_and_sizing() -> None:
    costs = CostModel()
    assert costs.fill_price(PRODUCTS["ES"], 5000.0, 1) == 5000.5
    assert costs.fill_price(PRODUCTS["ES"], 5000.0, -1) == 4999.5
    assert costs.fees_usd(PRODUCTS["ES"], -2) == 5.26
    state = SizerState()
    assert DollarNeutralSizer(es_contracts=2).size(1, 5000, 18000, 0.75, state) == (2, -1)
    assert FixedContracts().size(-1, 5000, 18000, 1, state) == (-1, 1)
    target = VolTargetSizer(1000, lookback_sessions=2, inner=FixedContracts())
    assert target.size(1, 5000, 18000, 1, state) == (1, -1)
    state.unit_pnl_daily = pd.Series([100.0, 200.0])
    assert target.size(1, 5000, 18000, 1, state) == (14, -14)


def test_signal_override_fills_on_next_bar_and_final_flat() -> None:
    result = run_simulation(
        _bars(),
        SimulationConfig(start="2026-01-05", end="2026-01-07", min_hedge_sessions=5),
        signal_override=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    )
    entries = result.trades[result.trades["reason"] == "entry"]
    assert set(entries["product"]) == {"ES", "NQ"}
    assert entries["ts_event"].nunique() == 1
    assert result.positions.iloc[-1][["n_es", "n_nq"]].tolist() == [0, 0]
