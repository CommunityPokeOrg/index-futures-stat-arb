from datetime import date, timedelta

from index_futures_stat_arb.execution.engine import bar_derived_ofi_proxy
from index_futures_stat_arb.walkforward import (
    SearchSpace,
    WalkForwardConfig,
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
