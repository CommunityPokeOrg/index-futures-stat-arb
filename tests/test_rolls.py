from datetime import date

import pandas as pd

from index_futures_stat_arb.contracts import parse_contract
from index_futures_stat_arb.rolls import (
    RollConfig,
    build_roll_calendar,
    cme_roll_date,
    daily_from_bars,
)


def test_cme_roll_date_and_prior_day_volume():
    esh = parse_contract("ESH6", pivot_year=2026)
    esm = parse_contract("ESM6", pivot_year=2026)
    assert cme_roll_date(esh) == date(2026, 3, 16)
    sessions = pd.date_range("2026-03-05", "2026-03-17", freq="B").date
    rows = []
    for session in sessions:
        front_volume = 100 if session < date(2026, 3, 10) else 50
        next_volume = 50 if session < date(2026, 3, 10) else 200
        for contract, close, volume in (
            ("ESH6", 5000.0, front_volume),
            ("ESM6", 5020.0, next_volume),
        ):
            rows.append(
                {
                    "session_date": session,
                    "contract": contract,
                    "close": close,
                    "volume": volume,
                }
            )
    daily = pd.DataFrame(rows)
    calendar = build_roll_calendar(
        "ES",
        daily,
        [esh, esm],
        RollConfig(earliest_bdays_before_expiry=10),
        date(2026, 3, 1),
        date(2026, 3, 20),
    )
    assert calendar.iloc[0]["roll_session_date"] == date(2026, 3, 11)
    assert "volume_prev_day" in calendar.iloc[0]["decision_basis"]
    assert calendar.iloc[0]["panama_offset"] == 20.0


def test_daily_from_bars():
    bars = pd.DataFrame(
        {
            "session_date": [date(2026, 1, 5), date(2026, 1, 5)],
            "contract": ["ESH6", "ESH6"],
            "ts_event": pd.to_datetime(["2026-01-05 00:00", "2026-01-05 00:01"], utc=True),
            "close": [1.0, 2.0],
            "volume": [3, 4],
        }
    )
    result = daily_from_bars(bars)
    assert result.iloc[0]["close"] == 2.0
    assert result.iloc[0]["volume"] == 7
