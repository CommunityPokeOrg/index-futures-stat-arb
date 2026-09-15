"""CME equity-index contract metadata and persistence."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .schema import CONTRACTS_SCHEMA
from .sessions import ET, cme_holidays

MONTH_CODES = {"H": 3, "M": 6, "U": 9, "Z": 12}
INVERSE_MONTH_CODES = {month: code for code, month in MONTH_CODES.items()}


@dataclass(frozen=True)
class ProductSpec:
    product: str
    multiplier_usd: float
    tick_size: float
    tick_value_usd: float
    exchange: str = "XCME"
    currency: str = "USD"
    asset_class: Literal["future", "equity"] = "future"


PRODUCTS = {
    "ES": ProductSpec("ES", 50.0, 0.25, 12.5),
    "NQ": ProductSpec("NQ", 20.0, 0.25, 5.0),
    "SPY": ProductSpec("SPY", 1.0, 0.01, 0.01, exchange="ARCX", asset_class="equity"),
    "QQQ": ProductSpec("QQQ", 1.0, 0.01, 0.01, exchange="ARCX", asset_class="equity"),
}


def third_friday(year: int, month: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(4 - first.weekday()) % 7 + 14)


def adjust_expiry_date(day: date, holidays: set[date] | None = None) -> date:
    holiday_set = holidays if holidays is not None else cme_holidays(day.year)
    while day.weekday() >= 5 or day in holiday_set:
        day -= timedelta(days=1)
    return day


def contract_symbol(product: str, year: int, month: int | str) -> str:
    code = month if isinstance(month, str) else INVERSE_MONTH_CODES[month]
    return f"{product}{code}{year % 100:02d}".replace(f"{year % 100:02d}", str(year % 10))


@dataclass(frozen=True)
class Contract:
    product: str
    month_code: str
    year: int
    symbol: str
    expiry_date: date
    expiry_ts: pd.Timestamp
    first_trade_date: date
    last_trade_date: date


def parse_contract(symbol: str, pivot_year: int | None = None) -> Contract:
    for product in PRODUCTS:
        if symbol.startswith(product):
            break
    else:
        raise ValueError(f"unknown futures product in {symbol!r}")
    suffix = symbol[len(product) :]
    if len(suffix) < 2 or suffix[0] not in MONTH_CODES:
        raise ValueError(f"invalid futures contract symbol: {symbol!r}")
    month_code = suffix[0]
    year_text = suffix[1:]
    if len(year_text) == 1:
        pivot = pivot_year or date.today().year
        year = pivot // 10 * 10 + int(year_text)
        while year < pivot - 2:
            year += 10
    elif len(year_text) == 2:
        year = 2000 + int(year_text)
    else:
        raise ValueError(f"invalid futures contract symbol: {symbol!r}")
    expiry_date = adjust_expiry_date(third_friday(year, MONTH_CODES[month_code]))
    expiry_ts = pd.Timestamp(
        datetime.combine(expiry_date, datetime.min.time().replace(hour=9, minute=30)),
        tz=ET,
    )
    prior_month = MONTH_CODES[month_code] - 6
    prior_year = year
    if prior_month <= 0:
        prior_month += 12
        prior_year -= 1
    first_trade = adjust_expiry_date(third_friday(prior_year, prior_month))
    return Contract(
        product,
        month_code,
        year,
        symbol,
        expiry_date,
        expiry_ts,
        first_trade,
        expiry_date,
    )


def list_contracts(product: str, start: date, end: date) -> list[Contract]:
    if product not in PRODUCTS:
        raise ValueError(f"unknown product: {product}")
    contracts: list[Contract] = []
    for year in range(start.year - 1, end.year + 2):
        for month in MONTH_CODES.values():
            contract = parse_contract(contract_symbol(product, year, month), pivot_year=year)
            if contract.expiry_date >= start and contract.first_trade_date <= end:
                contracts.append(contract)
    return sorted(contracts, key=lambda item: item.expiry_date)


def contracts_table(contracts: Iterable[Contract]) -> pd.DataFrame:
    rows = []
    for contract in contracts:
        spec = PRODUCTS[contract.product]
        rows.append(
            {
                "product": contract.product,
                "contract": contract.symbol,
                "month_code": contract.month_code,
                "year": contract.year,
                "first_trade_date": contract.first_trade_date,
                "last_trade_date": contract.last_trade_date,
                "expiration_ts": contract.expiry_ts.tz_convert("UTC"),
                "multiplier_usd": spec.multiplier_usd,
                "tick_size": spec.tick_size,
                "tick_value_usd": spec.tick_value_usd,
                "exchange": spec.exchange,
                "currency": spec.currency,
                "source": "cme-calendar-approximation",
                "asof": date.today(),
            }
        )
    return pd.DataFrame(rows, columns=CONTRACTS_SCHEMA.names)


def write_contracts(df: pd.DataFrame, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, schema=CONTRACTS_SCHEMA, preserve_index=False)
    pq.write_table(table, destination, compression="zstd", compression_level=3)
