from datetime import date, timedelta

import pandas as pd

from index_futures_stat_arb.execution.engine import (
    SimulationConfig,
    SimulationResult,
    bar_derived_ofi_proxy,
)
from index_futures_stat_arb.walkforward import (
    Fold,
    SearchSpace,
    WalkForwardConfig,
    _fold_metric,
    make_folds,
    sample_trials,
)


def test_sample_trials_is_deterministic_and_ordered() -> None:
    first = sample_trials(SearchSpace(), 12, 7)
    second = sample_trials(SearchSpace(), 12, 7)
    assert first == second
    assert first[0]["entry"] == 2.0
    for trial in first:
        assert float(trial["exit_ratio"]) < 1.0 < float(trial["stop_ratio"])
        assert 1e-7 <= float(trial["kalman_delta"]) <= 1e-3
        assert 1e-8 <= float(trial["kalman_obs_var"]) <= 1e-4


def test_make_folds_are_anchored_and_cover_the_test_period() -> None:
    sessions = [date(2020, 1, 1) + timedelta(days=item) for item in range(23)]
    folds = make_folds(sessions, WalkForwardConfig(n_folds=4, min_train_sessions=7))
    assert folds[0].test_start == sessions[7]
    assert folds[0].train_sessions[-1] < folds[0].test_start
    assert [item for fold in folds for item in fold.test_sessions] == sessions[7:]
    for previous, current in zip(folds, folds[1:], strict=False):
        assert previous.test_end < current.test_start
        assert current.train_sessions[-1] < current.test_start


def test_bar_derived_ofi_proxy_is_causal_and_zero_volume_is_neutral() -> None:
    opens = [10.0, 10.0, 10.0]
    closes = [11.0, 9.0, 11.0]
    volumes = [2.0, 1.0, 3.0]
    before = bar_derived_ofi_proxy(opens[:2], closes[:2], volumes[:2], 2)
    after = bar_derived_ofi_proxy(opens, closes, volumes, 2)
    assert before == 1 / 3
    assert after == 1 / 2
    assert bar_derived_ofi_proxy([1.0], [0.0], [0.0], 20) == 0.0


def test_fold_win_rate_uses_closed_round_trip_pnl() -> None:
    dates = pd.date_range("2020-01-01", periods=5, tz="UTC")
    positions = pd.DataFrame(
        {"n_a": [1, 1, 0, 1, 0], "n_b": [1, 1, 0, 1, 0]},
        index=dates,
    )
    pnl = pd.Series([0.0, 10.0, 0.0, -5.0, 0.0], index=dates)
    daily_pnl = pd.Series([10.0, -5.0], index=[date(2020, 1, 3), date(2020, 1, 5)])
    config = SimulationConfig(
        products=("ES", "NQ"),
        start="2020-01-01",
        end="2020-01-05",
        initial_capital_usd=1000.0,
    )
    result = SimulationResult(
        positions=positions,
        pnl=pnl,
        equity=1000.0 + pnl.cumsum(),
        trades=pd.DataFrame(),
        daily_pnl=daily_pnl,
        metrics={},
        config=config,
        hedge_history=pd.DataFrame(),
        signals=pd.DataFrame(),
    )
    fold = Fold(0, (date(2020, 1, 1),), (date(2020, 1, 3), date(2020, 1, 5)))
    metrics = _fold_metric(result, config, fold)
    assert metrics["round_trips"] == 2
    assert metrics["win_rate"] == 0.5
