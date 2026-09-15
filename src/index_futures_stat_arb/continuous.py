"""Continuous futures series with explicit roll adjustments."""

from __future__ import annotations

from datetime import date
from typing import Literal

import numpy as np
import pandas as pd

Adjust = Literal["none", "panama", "ratio"]


def build_continuous(
    bars: pd.DataFrame,
    calendar: pd.DataFrame,
    contracts: object,
    adjust: Adjust,
    asof: date | None = None,
    direction: Literal["back", "forward"] = "back",
) -> pd.DataFrame:
    del contracts
    if adjust not in ("none", "panama", "ratio"):
        raise ValueError(f"unknown adjustment: {adjust}")
    if direction not in ("back", "forward"):
        raise ValueError(f"unknown direction: {direction}")
    if bars.empty:
        return pd.DataFrame(
            columns=[
                "ts_event",
                "session_date",
                "contract",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "is_rth",
                "roll_flag",
                "adj_offset",
            ]
        )
    rolls = calendar.sort_values("roll_session_date").copy()
    if asof is not None:
        rolls = rolls[rolls["roll_session_date"] <= asof]
    bars = bars.sort_values("ts_event").copy()
    default_front = str(bars["contract"].iloc[0])
    front_map = {
        str(session): _front_for_session(str(session), rolls, default_front)
        for session in bars["session_date"].drop_duplicates()
    }
    front = bars["session_date"].astype(str).map(front_map)
    selected = bars.loc[front.notna() & (bars["contract"] == front)].copy()
    if selected.empty:
        return selected
    selected["_front"] = front.loc[selected.index]
    roll_dates = list(rolls["roll_session_date"])
    offsets = pd.to_numeric(rolls["panama_offset"], errors="coerce").to_numpy()
    ratios = pd.to_numeric(rolls["ratio_factor"], errors="coerce").to_numpy()
    additive: list[float] = []
    factors: list[float] = []
    for session in selected["session_date"]:
        if direction == "back":
            applicable = [i for i, day in enumerate(roll_dates) if day > session]
            additive.append(float(np.nansum(offsets[applicable])) if applicable else 0.0)
            factors.append(float(np.nanprod(ratios[applicable])) if applicable else 1.0)
        else:
            applicable = [i for i, day in enumerate(roll_dates) if day <= session]
            additive.append(-float(np.nansum(offsets[applicable])) if applicable else 0.0)
            factors.append(float(1.0 / np.nanprod(ratios[applicable])) if applicable else 1.0)
    selected["adj_offset"] = additive if adjust == "panama" else factors
    if adjust == "panama":
        adjustment = np.asarray(additive)
    elif adjust == "ratio":
        adjustment = np.asarray(factors)
    else:
        adjustment = np.ones(len(selected))
        selected["adj_offset"] = 0.0
    for column in ("open", "high", "low", "close"):
        values = selected[column].to_numpy(dtype=float)
        selected[column] = (
            values + adjustment
            if adjust == "panama"
            else (values * adjustment if adjust == "ratio" else values)
        )
    selected["roll_flag"] = selected["_front"].ne(selected["_front"].shift())
    selected["contract"] = selected["_front"]
    selected = selected.drop(columns="_front")
    return selected[
        [
            "ts_event",
            "session_date",
            "contract",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "is_rth",
            "roll_flag",
            "adj_offset",
        ]
    ].reset_index(drop=True)


def _front_for_session(
    session: str, calendar: pd.DataFrame, default_front: str | None = None
) -> str | None:
    if calendar.empty:
        return default_front
    current = str(calendar.iloc[0]["from_contract"])
    for row in calendar.itertuples(index=False):
        if session >= str(row.roll_session_date):
            current = str(row.to_contract)
    return current
