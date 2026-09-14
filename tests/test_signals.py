import numpy as np
import pandas as pd
import pytest

from index_futures_stat_arb import signals


def test_compute_spread_arithmetic():
    y = pd.Series([10.0, 20.0])
    x = pd.Series([3.0, 5.0])
    s = signals.compute_spread(y, x, beta=2.0, alpha=1.0)
    assert s.tolist() == [3.0, 9.0]


def test_zscore_full_sample():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(5.0, 2.0, 5000))
    z = signals.zscore(s)
    assert z.mean() == pytest.approx(0.0, abs=1e-10)
    assert z.std() == pytest.approx(1.0, abs=1e-10)


def test_zscore_rolling():
    s = pd.Series(np.arange(100, dtype=float))
    z = signals.zscore(s, window=10)
    assert z.iloc[:9].isna().all()
    assert not z.iloc[9:].isna().any()


def test_generate_signals_transitions():
    # long entry at -2.1, exit at 0.4, short entry at 2.1, exit at -0.3,
    # stop at 4.1 -> flat, stay flat until |z|<exit (cleared at -0.2),
    # then re-enter long at -2.5.
    z = pd.Series(
        [0.0, -2.1, -1.0, 0.4, 0.0, 2.1, 1.0, -0.3, 4.1, 3.0, -0.2, -2.5]
    )
    pos = signals.generate_signals(z, entry=2.0, exit=0.5, stop=4.0)
    expected = [0, 1, 1, 0, 0, -1, -1, 0, 0, 0, 0, 1]
    assert pos.tolist() == expected


def test_generate_signals_nan_holds_position():
    z = pd.Series([-2.5, np.nan, np.nan, 0.1])
    pos = signals.generate_signals(z, entry=2.0, exit=0.5)
    assert pos.tolist() == [1, 1, 1, 0]


def test_generate_signals_no_stop():
    z = pd.Series([2.5, 5.0, 0.0])
    pos = signals.generate_signals(z, entry=2.0, exit=0.5, stop=None)
    assert pos.tolist() == [-1, -1, 0]
