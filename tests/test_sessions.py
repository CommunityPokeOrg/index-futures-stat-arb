import numpy as np
import pandas as pd
import pytest

from index_futures_stat_arb.sessions import (
    add_business_days,
    cme_holidays,
    is_rth,
    session_date,
)


def test_session_date_boundaries_and_dst():
    before = pd.Timestamp("2026-01-05 22:59:00", tz="UTC")
    after = pd.Timestamp("2026-01-05 23:00:00", tz="UTC")
    dst = pd.Timestamp("2026-03-08 22:00:00", tz="UTC")
    assert session_date(before).isoformat() == "2026-01-05"
    assert session_date(after).isoformat() == "2026-01-06"
    assert session_date(dst).isoformat() == "2026-03-09"
    result = session_date(pd.DatetimeIndex([before, after]))
    assert np.array_equal(result, np.array([date for date in result]))


def test_is_rth_boundaries():
    assert not is_rth(pd.Timestamp("2026-01-05 14:29:00", tz="UTC"))
    assert is_rth(pd.Timestamp("2026-01-05 14:30:00", tz="UTC"))
    assert not is_rth(pd.Timestamp("2026-01-05 21:00:00", tz="UTC"))


def test_naive_timestamps_raise():
    with pytest.raises(ValueError):
        session_date(pd.Timestamp("2026-01-05"))
    with pytest.raises(ValueError):
        is_rth(pd.DatetimeIndex(["2026-01-05"]))


def test_business_days_skip_good_friday():
    assert pd.Timestamp("2026-04-03").date() in cme_holidays(2026)
    assert add_business_days(pd.Timestamp("2026-04-02").date(), 1).isoformat() == ("2026-04-06")
