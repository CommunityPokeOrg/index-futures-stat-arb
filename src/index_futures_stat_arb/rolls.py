"""Contract roll-calendar construction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

import numpy as np
import pandas as pd

from .contracts import Contract, parse_contract
from .schema import ROLL_CALENDAR_SCHEMA
from .sessions import add_business_days, is_business_day

RollRule = Literal["calendar", "volume", "open_interest", "fixed_k"]


@dataclass(frozen=True)
class RollConfig:
    rule: RollRule = "volume"
    fixed_k_days: int = 8
    earliest_bdays_before_expiry: int = 10
    latest_rule: Literal["cme_monday"] = "cme_monday"
    one_way: bool = True
    hysteresis_ratio: float = 1.0


def cme_roll_date(contract: Contract) -> date:
    monday = contract.expiry_date - timedelta(days=contract.expiry_date.weekday())
    while not is_business_day(monday):
        monday += timedelta(days=1)
    return monday


def calendar_roll_date(
    contract: Contract,
    rule: Literal["calendar", "fixed_k"],
    fixed_k_days: int = 8,
) -> date:
    if rule == "calendar":
        return cme_roll_date(contract)
    return add_business_days(contract.expiry_date, -fixed_k_days)


def daily_from_bars(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame(
            columns=["session_date", "contract", "close", "volume", "open_interest"]
        )
    grouped = bars.sort_values("ts_event").groupby(["session_date", "contract"], sort=True)
    result = grouped.agg(close=("close", "last"), volume=("volume", "sum")).reset_index()
    if "open_interest" in bars:
        result["open_interest"] = grouped["open_interest"].last().to_numpy()
    else:
        result["open_interest"] = np.nan
    return result[["session_date", "contract", "close", "volume", "open_interest"]]


def _previous_session(day: date) -> date:
    return add_business_days(day, -1)


def _prior_value(
    daily: pd.DataFrame,
    contract: str,
    day: date,
    column: str,
) -> float:
    cutoff = day
    for _ in range(5):
        rows = daily[(daily["contract"] == contract) & (daily["session_date"] <= cutoff)]
        if not rows.empty:
            value = rows.sort_values("session_date").iloc[-1][column]
            if pd.notna(value):
                return float(value)
        cutoff = _previous_session(cutoff)
    return float("nan")


def build_roll_calendar(
    product: str,
    daily: pd.DataFrame | None,
    contracts: list[Contract],
    cfg: RollConfig,
    start: date,
    end: date,
) -> pd.DataFrame:
    selected = [contract for contract in contracts if contract.product == product]
    selected.sort(key=lambda item: item.expiry_date)
    rows: list[dict[str, object]] = []
    for front, next_contract in zip(selected[:-1], selected[1:], strict=False):
        earliest = add_business_days(front.expiry_date, -cfg.earliest_bdays_before_expiry)
        guard = cme_roll_date(front)
        if guard < start or earliest > end:
            continue
        if cfg.rule in ("volume", "open_interest"):
            metric = "volume" if cfg.rule == "volume" else "open_interest"
            if daily is None:
                raise ValueError(f"{cfg.rule} rolls require daily data")
            roll = None
            lower, upper = max(start, earliest), min(end, guard)
            current = lower
            while current <= upper:
                prior = _previous_session(current)
                from_value = _prior_value(daily, front.symbol, prior, metric)
                to_value = _prior_value(daily, next_contract.symbol, prior, metric)
                if (
                    pd.notna(from_value)
                    and pd.notna(to_value)
                    and to_value > cfg.hysteresis_ratio * from_value
                ):
                    roll = (
                        current,
                        f"{metric}_prev_day from={from_value:g} to={to_value:g}",
                    )
                    break
                current = add_business_days(current, 1)
            if roll is None:
                roll = (upper, "calendar_guard")
        elif cfg.rule == "calendar":
            roll = (max(start, max(earliest, cme_roll_date(front))), "cme_monday")
        else:
            fixed = add_business_days(front.expiry_date, -cfg.fixed_k_days)
            roll = (max(start, max(earliest, min(guard, fixed))), "fixed_k")
        roll_date, basis = roll
        if roll_date > end:
            continue
        decision_day = _previous_session(roll_date)
        from_close = (
            _prior_value(daily, front.symbol, decision_day, "close")
            if daily is not None
            else np.nan
        )
        to_close = (
            _prior_value(daily, next_contract.symbol, decision_day, "close")
            if daily is not None
            else np.nan
        )
        offset = to_close - from_close if pd.notna(from_close) and pd.notna(to_close) else np.nan
        ratio = (
            to_close / from_close
            if pd.notna(from_close) and pd.notna(to_close) and from_close
            else np.nan
        )
        rows.append(
            {
                "product": product,
                "rule": cfg.rule,
                "from_contract": front.symbol,
                "to_contract": next_contract.symbol,
                "roll_session_date": roll_date,
                "decision_basis": basis,
                "panama_offset": offset,
                "ratio_factor": ratio,
                "joint_roll_session_date": pd.NaT,
                "asof": end,
            }
        )
    return pd.DataFrame(rows, columns=ROLL_CALENDAR_SCHEMA.names)


def front_contract_series(
    calendar: pd.DataFrame,
    contracts: list[Contract],
    sessions: Sequence[date],
) -> pd.Series:
    ordered = sorted(contracts, key=lambda item: item.expiry_date)
    rows = calendar.sort_values("roll_session_date")
    values: list[str] = []
    for session in sessions:
        current = next(
            (contract.symbol for contract in ordered if contract.first_trade_date <= session),
            ordered[0].symbol,
        )
        for row in rows.itertuples(index=False):
            roll_day = pd.Timestamp(str(row.roll_session_date)).date()
            if session >= roll_day:
                current = str(row.to_contract)
        values.append(current)
    return pd.Series(values, index=pd.Index(sessions, name="session_date"), name="contract")


def joint_roll_calendar(
    cal_es: pd.DataFrame,
    cal_nq: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    es = cal_es.copy()
    nq = cal_nq.copy()
    for frame in (es, nq):
        frame["joint_roll_session_date"] = frame["roll_session_date"]
    for index, row in es.iterrows():
        matching = nq[nq["from_contract"].str[2:] == row["from_contract"][2:]]
        if matching.empty:
            continue
        nq_row = matching.iloc[0]
        es_cme = cme_roll_date(parse_contract(str(row["from_contract"])))
        nq_cme = cme_roll_date(parse_contract(str(nq_row["from_contract"])))
        joint = min(
            max(row["roll_session_date"], nq_row["roll_session_date"]),
            es_cme,
            nq_cme,
        )
        es.loc[index, "joint_roll_session_date"] = joint
        nq.loc[nq_row.name, "joint_roll_session_date"] = joint
    return {"ES": es, "NQ": nq}
