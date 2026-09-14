import numpy as np
import pandas as pd
import pytest

from index_futures_stat_arb import backtest as bt
from index_futures_stat_arb import data


def test_backtest_pnl_with_lag_and_costs():
    spread = pd.Series([0.0, 1.0, 3.0, 2.0, 2.0])
    positions = pd.Series([1.0, 1.0, 1.0, -1.0, -1.0])
    res = bt.backtest_spread(spread, positions, cost_per_unit=0.1, lag=1)
    # held = pos shifted 1: [0,1,1,1,-1]; dspread: [nan,1,2,-1,0]
    # gross: [0,1,2,-1,0]; cost: |dpos|*0.1: [0.1,0,0,0.2,0]
    expected = pd.Series([-0.1, 1.0, 2.0, -1.2, 0.0])
    assert np.allclose(res.pnl, expected)
    assert np.allclose(res.equity, expected.cumsum())
    assert res.trades == 2  # initial long + flip to short


def test_backtest_zero_cost_no_lag():
    spread = pd.Series([0.0, 2.0, 1.0])
    positions = pd.Series([0.0, 1.0, 1.0])
    res = bt.backtest_spread(spread, positions, lag=0)
    assert np.allclose(res.pnl, [0.0, 2.0, -1.0])


def test_performance_metrics():
    pnl = pd.Series([1.0, -0.5, 2.0, -1.0, 0.5])
    m = bt.performance_metrics(pnl)
    assert m["total_pnl"] == pytest.approx(2.0)
    assert m["n_periods"] == 5
    # equity: 1.0, 0.5, 2.5, 1.5, 2.0 -> max drawdown 1.0
    assert m["max_drawdown"] == pytest.approx(2.5 - 1.5)
    assert m["hit_rate"] == pytest.approx(3 / 5)
    expected_sharpe = pnl.mean() / pnl.std() * np.sqrt(252)
    assert m["sharpe"] == pytest.approx(expected_sharpe)


def test_performance_metrics_zero_std():
    m = bt.performance_metrics(pd.Series([1.0, 1.0, 1.0]))
    assert m["sharpe"] == 0.0


def test_train_test_split_by_fraction_and_date():
    df = data.generate_synthetic_pair(n=100, seed=0)
    train, test = bt.train_test_split_by_date(df, 0.7)
    assert len(train) == 70
    assert len(test) == 30
    split_date = str(test.index[0].date())
    train2, test2 = bt.train_test_split_by_date(df, split_date)
    pd.testing.assert_frame_equal(train, train2)
    pd.testing.assert_frame_equal(test, test2)


def test_walk_forward_backtest_keys_and_lengths():
    prices = data.generate_synthetic_pair(
        n=2000, seed=0, sigma_spread=0.05, sigma_common=0.02
    )
    out = bt.walk_forward_backtest(
        prices, split=0.7, z_window=60, entry=2.0, exit=0.5, stop=4.0
    )
    for key in ("hedge", "ou", "train", "test", "spread", "zscore"):
        assert key in out
    train, test = bt.train_test_split_by_date(prices, 0.7)
    assert len(out["test"].equity) == len(test)
    assert len(out["train"].equity) == len(train)
    assert out["hedge"].beta == pytest.approx(1.3, abs=0.15)
    assert "total_pnl" in out["test"].metrics
