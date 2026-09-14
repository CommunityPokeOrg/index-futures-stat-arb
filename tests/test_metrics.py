import pandas as pd

from index_futures_stat_arb.execution.metrics import compute_metrics


def _inputs() -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    pnl = pd.Series([10.0, -5.0, 0.0, 20.0, -2.0])
    daily = pd.Series([5.0, 18.0])
    positions = pd.DataFrame({"n_es": [1, 1, 0, -1, 0], "n_nq": [-1, -1, 0, 1, 0]})
    trades = pd.DataFrame(
        {
            "qty": [1, -1, -1, 1],
            "fees_usd": [2.0, 2.0, 2.0, 2.0],
            "slippage_usd": [1.0, 1.0, 1.0, 1.0],
        }
    )
    return pnl, daily, trades, positions


def test_total_and_return_metrics() -> None:
    pnl, daily, trades, positions = _inputs()
    metrics = compute_metrics(pnl, daily, trades, positions, 1000.0)
    assert metrics["total_pnl_usd"] == 23.0
    assert metrics["total_return_pct"] == 2.3
    assert metrics["n_sessions"] == 2


def test_drawdown_metrics() -> None:
    pnl, daily, trades, positions = _inputs()
    metrics = compute_metrics(pnl, daily, trades, positions, 1000.0)
    assert metrics["max_drawdown_usd"] == 5.0
    assert metrics["max_drawdown_pct"] == 0.5


def test_round_trip_and_trade_metrics() -> None:
    pnl, daily, trades, positions = _inputs()
    metrics = compute_metrics(pnl, daily, trades, positions, 1000.0)
    assert metrics["n_round_trips"] == 2
    assert metrics["win_rate"] == 1.0
    assert metrics["profit_factor"] == 0.0
    assert metrics["total_fees_usd"] == 8.0


def test_metrics_exposure_and_turnover() -> None:
    pnl, daily, trades, positions = _inputs()
    metrics = compute_metrics(pnl, daily, trades, positions, 1000.0)
    assert metrics["turnover_contracts"] == 4.0
    assert metrics["exposure_frac"] == 3 / 5


def test_empty_inputs_are_safe() -> None:
    metrics = compute_metrics(
        pd.Series(dtype=float),
        pd.Series(dtype=float),
        pd.DataFrame(),
        pd.DataFrame(columns=["n_es", "n_nq"]),
        0.0,
    )
    assert metrics["total_pnl_usd"] == 0.0
    assert metrics["sharpe"] == 0.0
    assert metrics["n_round_trips"] == 0
