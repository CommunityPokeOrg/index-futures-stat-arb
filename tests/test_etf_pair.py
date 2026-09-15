import numpy as np
import pandas as pd

from index_futures_stat_arb.contracts import PRODUCTS
from index_futures_stat_arb.execution.costs import CostModel
from index_futures_stat_arb.execution.engine import SimulationConfig, run_simulation
from index_futures_stat_arb.execution.sizing import (
    DollarNeutralSizer,
    SizerSpec,
    SizerState,
    VolTargetSizer,
)


def test_es_spy_notional_sizing_and_equity_costs() -> None:
    sizer = DollarNeutralSizer(
        es_contracts=1,
        max_units_a=50,
        spec_a=PRODUCTS["ES"],
        spec_b=PRODUCTS["SPY"],
    )
    assert sizer.unit(1, 4800.0, 480.0, 1.0) == (1.0, -500.0)
    costs = CostModel()
    assert costs.fees_usd(PRODUCTS["SPY"], 500) == 2.5
    assert costs.fill_price(PRODUCTS["SPY"], 480.0, 1) == 480.02
    assert costs.fill_price(PRODUCTS["SPY"], 480.0, -1) == 479.98


def test_vol_target_scales_fractional_etf_hedge() -> None:
    inner = DollarNeutralSizer(es_contracts=1, spec_a=PRODUCTS["ES"], spec_b=PRODUCTS["NQ"])
    sizer = VolTargetSizer(
        target_daily_vol_usd=18.384776310850235,
        lookback_sessions=2,
        max_units_a=50,
        inner=inner,
    )
    result = sizer.size(1, 2100.0, 4450.0, 0.75, SizerState(pd.Series([0.0, 1.0])))
    assert result[0] == 26
    assert result[1] == -23


def test_es_spy_engine_accepts_generic_pair_columns() -> None:
    rng = np.random.default_rng(7)
    n = 100
    b = 480.0 + np.cumsum(rng.normal(0, 0.4, n))
    ou = np.zeros(n)
    for i in range(1, n):
        ou[i] = 0.85 * ou[i - 1] + rng.normal(0, 0.01)
    a = 10.0 * b * np.exp(ou)
    ts = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    bars = pd.DataFrame(
        {
            "ts_event": ts,
            "session_date": ts.date,
            "a_open": a,
            "a_high": a,
            "a_low": a,
            "a_close": a,
            "a_volume": 1,
            "b_open": b,
            "b_high": b,
            "b_low": b,
            "b_close": b,
            "b_volume": 1,
            "a_contract": "ES=F",
            "b_contract": "SPY",
            "a_roll": False,
            "b_roll": False,
            "a_close_raw": a,
            "b_close_raw": b,
            "carry": 0.0,
        }
    )
    config = SimulationConfig(
        start="2025-01-01",
        end="2025-05-01",
        products=("ES", "SPY"),
        z_window=20,
        hedge_lookback_sessions=30,
        min_hedge_sessions=10,
        signal_lag_bars=0,
        mask_roll_sessions=False,
        sizer=SizerSpec("dollar_neutral", {"es_contracts": 1, "max_units_a": 5}),
    )
    result = run_simulation(bars, config)
    assert {"n_a", "n_b"} <= set(result.positions.columns)
