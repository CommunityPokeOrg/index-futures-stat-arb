import numpy as np
import pandas as pd
from test_engine import make_bars

from index_futures_stat_arb.execution.engine import SimulationConfig, run_simulation
from index_futures_stat_arb.execution.hedge import (
    KalmanHedge,
    rolling_engle_granger,
)


def test_kalman_tracks_drifting_beta() -> None:
    rng = np.random.default_rng(7)
    log_nq = rng.normal(9.7, 0.15, 1000)
    true_beta = np.linspace(0.6, 0.9, len(log_nq))
    log_es = 0.4 + true_beta * log_nq + rng.normal(0, 0.001, len(log_nq))
    hedge = KalmanHedge(delta=1e-5, obs_var=1e-4)
    estimates = []
    for x, y in zip(log_nq, log_es, strict=True):
        hedge.predict(float(x))
        hedge.update(float(y), float(x))
        estimates.append(hedge.beta)
    assert abs(estimates[-1] - true_beta[-1]) < 0.05
    assert np.mean(np.abs(np.asarray(estimates)[200:] - true_beta[200:])) < 0.05


def test_rolling_engle_granger_distinguishes_random_walks() -> None:
    rng = np.random.default_rng(8)
    x = np.cumsum(rng.normal(size=500))
    cointegrated = x + rng.normal(0, 0.2, len(x))
    independent = np.cumsum(rng.normal(size=500))
    assert rolling_engle_granger(cointegrated, x).pvalue < 0.05
    assert rolling_engle_granger(independent, x).pvalue > 0.05


def test_kalman_simulation_has_no_future_dependency() -> None:
    bars = make_bars(sessions=8, bars_per_session=6)
    cfg = SimulationConfig(
        start="2026-01-05",
        end="2026-01-20",
        min_hedge_sessions=2,
        hedge_lookback_sessions=3,
        hedge_method="kalman",
    )
    cut = 30
    perturbed = bars.copy()
    perturbed.loc[cut:, "es_close"] *= 1.2
    perturbed.loc[cut:, "nq_close"] *= 1.2
    one = run_simulation(bars, cfg)
    two = run_simulation(perturbed, cfg)
    pd.testing.assert_frame_equal(one.signals.iloc[:cut], two.signals.iloc[:cut])
    pd.testing.assert_frame_equal(one.positions.iloc[:cut], two.positions.iloc[:cut])
    first_trades = one.trades
    second_trades = two.trades
    if "ts_event" in first_trades and "ts_event" in second_trades:
        cutoff = bars.iloc[cut - 1]["ts_event"]
        first_trades = first_trades[first_trades["ts_event"] <= cutoff]
        second_trades = second_trades[second_trades["ts_event"] <= cutoff]
    pd.testing.assert_frame_equal(
        first_trades.reset_index(drop=True), second_trades.reset_index(drop=True)
    )
