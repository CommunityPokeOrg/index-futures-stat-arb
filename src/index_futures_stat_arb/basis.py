"""Cash-and-carry basis and index ETF dividend adjustments."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from .contracts import adjust_expiry_date, third_friday


def next_quarterly_expiry(day: date) -> date:
    """Return the next quarterly futures expiry on or after ``day``."""
    for year in range(day.year, day.year + 2):
        for month in (3, 6, 9, 12):
            expiry = adjust_expiry_date(third_friday(year, month), holidays=set())
            if expiry >= day:
                return expiry
    raise ValueError("unable to find quarterly expiry")


def time_to_expiry_years(day: date) -> float:
    return max((next_quarterly_expiry(day) - day).days, 0) / 365.0


def trailing_dividend_yield(dividends: pd.Series, price: pd.Series) -> pd.Series:
    """Return trailing 365-day cash dividends divided by aligned price."""
    div = dividends.copy()
    div.index = pd.to_datetime(div.index).tz_localize(None).normalize()
    px = price.copy()
    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
    values = pd.Series(0.0, index=px.index)
    for timestamp in px.index:
        values.at[timestamp] = float(
            div.loc[
                (div.index > timestamp - pd.Timedelta(days=365)) & (div.index <= timestamp)
            ].sum()
        )
    return (values / px).fillna(0.0)


def carry(rate: pd.Series, div_yield: pd.Series, tau_years: pd.Series) -> pd.Series:
    return (rate - div_yield) * tau_years


def fair_value(spot: pd.Series, carry: pd.Series) -> pd.Series:
    return spot * np.exp(carry)


def log_basis(future: pd.Series, spot: pd.Series, carry: pd.Series | None = None) -> pd.Series:
    result = np.log(future) - np.log(spot)
    return result if carry is None else result - carry
