from datetime import date

import pandas as pd

from index_futures_stat_arb.continuous import build_continuous


def test_panama_continuous_series_adjusts_ohlc_and_marks_roll():
    bars = pd.DataFrame(
        {
            "ts_event": pd.to_datetime(
                ["2026-01-05 15:00", "2026-01-06 15:00", "2026-01-07 15:00"],
                utc=True,
            ),
            "session_date": [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)],
            "contract": ["ESH6", "ESH6", "ESM6"],
            "open": [99.0, 100.0, 111.0],
            "high": [101.0, 102.0, 113.0],
            "low": [98.0, 99.0, 110.0],
            "close": [100.0, 101.0, 112.0],
            "volume": [1, 2, 3],
            "is_rth": [True, True, True],
        }
    )
    calendar = pd.DataFrame(
        {
            "roll_session_date": [date(2026, 1, 7)],
            "from_contract": ["ESH6"],
            "to_contract": ["ESM6"],
            "panama_offset": [10.0],
            "ratio_factor": [112 / 102],
        }
    )
    result = build_continuous(bars, calendar, [], "panama")
    assert result["roll_flag"].tolist() == [True, False, True]
    assert result["close"].tolist() == [110.0, 111.0, 112.0]
    assert result["open"].tolist() == [109.0, 110.0, 111.0]
