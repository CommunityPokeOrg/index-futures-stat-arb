from datetime import date

import numpy as np
import pandas as pd

from index_futures_stat_arb.basis import (
    log_basis,
    next_quarterly_expiry,
    time_to_expiry_years,
    trailing_dividend_yield,
)


def test_next_quarterly_expiry() -> None:
    assert next_quarterly_expiry(date(2026, 4, 1)) == date(2026, 6, 19)
    assert next_quarterly_expiry(date(2026, 6, 19)) == date(2026, 6, 19)
    assert next_quarterly_expiry(date(2026, 6, 20)) == date(2026, 9, 18)


def test_trailing_dividend_yield() -> None:
    dates = pd.date_range("2024-01-01", periods=5, freq="QS")
    dividends = pd.Series(1.0, index=dates)
    prices = pd.Series(100.0, index=pd.date_range("2023-12-01", "2025-02-01", freq="D"))
    result = trailing_dividend_yield(dividends, prices)
    assert result.loc["2023-12-15"] == 0.0
    assert result.loc["2025-01-01"] == 0.04


def test_log_basis_fair_value_is_zero() -> None:
    spot = pd.Series([100.0, 101.0])
    carry = pd.Series([0.02, -0.01])
    future = spot * np.exp(carry)
    assert np.allclose(log_basis(future, spot, carry), 0.0)
    assert time_to_expiry_years(date(2026, 4, 1)) > 0
