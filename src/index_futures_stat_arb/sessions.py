"""CME equity-index session labeling."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")


def _observed_fixed_holiday(day: date) -> date:
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def _easter_sunday(year: int) -> date:
    """Anonymous Gregorian computus."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    easter_weekday = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * easter_weekday) // 451
    month, day = divmod(h + easter_weekday - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def cme_holidays(year: int) -> set[date]:
    """Approximate CME equity-index full-closure holidays.

    Early closes and product-specific holiday schedules are intentionally not
    modeled; this helper is for session/contract date calculations.
    """
    fixed = {
        _observed_fixed_holiday(date(year, 1, 1)),
        _observed_fixed_holiday(date(year, 6, 19)),
        _observed_fixed_holiday(date(year, 7, 4)),
        _observed_fixed_holiday(date(year, 12, 25)),
    }
    easter = _easter_sunday(year)
    return fixed | {
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        easter - timedelta(days=2),
        _nth_weekday(year, 5, 0, 5),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4),
    }


def is_business_day(day: date) -> bool:
    return day.weekday() < 5 and day not in cme_holidays(day.year)


def add_business_days(day: date, n: int) -> date:
    step = 1 if n >= 0 else -1
    remaining = abs(n)
    current = day
    while remaining:
        current += timedelta(days=step)
        if is_business_day(current):
            remaining -= 1
    return current


def business_days_between(start: date, end: date) -> int:
    if start == end:
        return 0
    step = 1 if end > start else -1
    current = start
    count = 0
    while current != end:
        current += timedelta(days=step)
        if is_business_day(current):
            count += step
    return count


def session_bounds_utc(session: date) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the half-open UTC interval for a CME trade session."""
    start = pd.Timestamp(
        datetime.combine(session - timedelta(days=1), datetime.min.time().replace(hour=18)),
        tz=ET,
    ).tz_convert("UTC")
    end = pd.Timestamp(
        datetime.combine(session, datetime.min.time().replace(hour=18)),
        tz=ET,
    ).tz_convert("UTC")
    return start, end


def _require_aware(value: pd.Timestamp | pd.DatetimeIndex) -> None:
    if isinstance(value, pd.DatetimeIndex):
        if value.tz is None:
            raise ValueError("timestamps must be timezone-aware")
    elif pd.Timestamp(value).tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")


def session_date(
    ts: pd.Timestamp | pd.DatetimeIndex,
) -> date | np.ndarray:
    """Return the CME trade date using the 18:00 ET session boundary."""
    _require_aware(ts)
    if isinstance(ts, pd.DatetimeIndex):
        local_index = ts.tz_convert(ET)
        return (local_index + pd.Timedelta(hours=6)).date
    local_timestamp = pd.Timestamp(ts).tz_convert(ET)
    return (local_timestamp + pd.Timedelta(hours=6)).date()


def is_rth(ts: pd.Timestamp | pd.DatetimeIndex) -> bool | np.ndarray:
    """Return whether timestamps fall in the 09:30 inclusive to 16:00 exclusive RTH."""
    _require_aware(ts)
    if isinstance(ts, pd.DatetimeIndex):
        local_index = ts.tz_convert(ET)
        minutes_index = local_index.hour * 60 + local_index.minute
        return (minutes_index >= 570) & (minutes_index < 960)
    local_timestamp = pd.Timestamp(ts).tz_convert(ET)
    minutes_scalar = local_timestamp.hour * 60 + local_timestamp.minute
    return 570 <= minutes_scalar < 960
