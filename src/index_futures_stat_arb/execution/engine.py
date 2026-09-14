"""Incremental, no-lookahead ES/NQ simulation engine."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from ..continuous import Adjust, build_continuous
from ..contracts import PRODUCTS, list_contracts
from ..ingest.databento import read_partitioned
from ..rolls import RollConfig, build_roll_calendar, daily_from_bars, joint_roll_calendar
from .costs import CostModel
from .metrics import compute_metrics
from .sizing import Sizer, SizerSpec, SizerState, build_sizer


class LookaheadError(RuntimeError):
    pass


@dataclass(frozen=True)
class SimulationConfig:
    start: str
    end: str
    products: tuple[str, str] = ("ES", "NQ")
    bar_minutes: int = 5
    rth_only: bool = True
    signal_lag_bars: int = 1
    z_window: int = 78
    entry: float = 2.0
    exit: float = 0.5
    stop: float | None = 4.0
    hedge_lookback_sessions: int = 10
    min_hedge_sessions: int = 5
    mask_roll_sessions: bool = True
    z_reset_each_session: bool = True
    adjust: Adjust = "panama"
    roll: RollConfig = field(default_factory=RollConfig)
    initial_capital_usd: float = 250_000.0
    seed: int = 0
    costs: CostModel = field(default_factory=CostModel)
    sizer: SizerSpec = field(default_factory=SizerSpec)


@dataclass
class SimulationResult:
    positions: pd.DataFrame
    pnl: pd.Series
    equity: pd.Series
    trades: pd.DataFrame
    daily_pnl: pd.Series
    metrics: dict[str, float | int]
    config: SimulationConfig
    hedge_history: pd.DataFrame
    signals: pd.DataFrame


@dataclass
class SignalState:
    entry: float
    exit: float
    stop: float | None
    position: int = 0
    stopped: bool = False
    last_event: str | None = None

    def update(self, value: float, allow_entry: bool = True) -> int:
        previous = self.position
        self.last_event = None
        if np.isnan(value):
            return self.position
        absolute = abs(value)
        if self.stopped:
            if absolute < self.exit:
                self.stopped = False
            else:
                self.position = 0
                return 0
        if self.stop is not None and absolute > self.stop:
            self.position = 0
            self.stopped = True
            if previous:
                self.last_event = "stop"
        elif self.position == 0 and allow_entry:
            if value > self.entry:
                self.position = -1
            elif value < -self.entry:
                self.position = 1
        elif self.position and absolute < self.exit:
            self.position = 0
        if self.last_event is None and self.position != previous:
            self.last_event = _signal_event(previous, self.position)
        return self.position


def _signal_event(old_signal: int, new_signal: int) -> str | None:
    if old_signal == new_signal:
        return None
    if old_signal == 0:
        return "entry"
    if new_signal == 0:
        return "exit"
    return "flip"


class _BarCursor:
    def __init__(self, bars: pd.DataFrame, index: int = 0) -> None:
        self._bars = bars
        self.index = index

    def at(self, index: int) -> pd.Series:
        if index > self.index:
            raise LookaheadError(f"requested bar {index} beyond cursor {self.index}")
        return self._bars.iloc[index]


def load_pair_bars(
    cfg: SimulationConfig,
    data_root: str | Path,
    source: str,
    dataset: str,
    schema: str,
) -> pd.DataFrame:
    raw = read_partitioned(
        data_root,
        source,
        dataset,
        schema,
        products=list(cfg.products),
        start=date.fromisoformat(cfg.start[:10]),
        end=date.fromisoformat(cfg.end[:10]),
    )
    start = date.fromisoformat(cfg.start[:10])
    end = date.fromisoformat(cfg.end[:10])
    calendars: dict[str, pd.DataFrame] = {}
    contracts_by_product = {}
    daily = daily_from_bars(raw)
    for product in cfg.products:
        contracts = list_contracts(product, start, end)
        contracts_by_product[product] = contracts
        calendars[product] = build_roll_calendar(product, daily, contracts, cfg.roll, start, end)
    joint = joint_roll_calendar(calendars["ES"], calendars["NQ"])
    frames = []
    for product in cfg.products:
        calendars[product] = joint[product]
        adjusted = build_continuous(
            raw[raw["product"] == product],
            calendars[product],
            contracts_by_product[product],
            cfg.adjust,
            direction="forward",
        )
        unadjusted = build_continuous(
            raw[raw["product"] == product],
            calendars[product],
            contracts_by_product[product],
            "none",
            direction="forward",
        )
        roll_sessions = set(calendars[product].loc[:, "roll_session_date"].dropna().tolist())
        adjusted["roll_flag"] = adjusted["session_date"].isin(roll_sessions)
        adjusted = _resample(adjusted, cfg.bar_minutes, cfg.rth_only, product)
        unadjusted = _resample(unadjusted, cfg.bar_minutes, cfg.rth_only, product)
        adjusted[f"{product.lower()}_close_raw"] = unadjusted["close"].to_numpy()
        frames.append(adjusted)
    es, nq = frames
    result = es.merge(nq, on="ts_event", suffixes=("_es", "_nq"))
    result["session_date"] = result["session_date_es"]
    result = result.rename(
        columns={
            "open_es": "es_open",
            "high_es": "es_high",
            "low_es": "es_low",
            "close_es": "es_close",
            "volume_es": "es_volume",
            "contract_es": "es_contract",
            "roll_flag_es": "es_roll",
            "open_nq": "nq_open",
            "high_nq": "nq_high",
            "low_nq": "nq_low",
            "close_nq": "nq_close",
            "volume_nq": "nq_volume",
            "contract_nq": "nq_contract",
            "roll_flag_nq": "nq_roll",
        }
    )
    return (
        result[
            [
                "ts_event",
                "session_date",
                "es_open",
                "es_high",
                "es_low",
                "es_close",
                "es_volume",
                "nq_open",
                "nq_high",
                "nq_low",
                "nq_close",
                "nq_volume",
                "es_contract",
                "nq_contract",
                "es_roll",
                "nq_roll",
                "es_close_raw",
                "nq_close_raw",
            ]
        ]
        .sort_values("ts_event")
        .reset_index(drop=True)
    )


def _resample(frame: pd.DataFrame, minutes: int, rth_only: bool, product: str = "") -> pd.DataFrame:
    frame = frame.copy()
    if "roll_flag" not in frame:
        frame["roll_flag"] = False
    if "roll_flag" not in frame:
        frame["roll_flag"] = False
    if rth_only:
        frame = frame[frame["is_rth"]]
    frame["_bar"] = frame["ts_event"].dt.floor(f"{minutes}min")
    grouped = frame.sort_values("ts_event").groupby("_bar", sort=True)
    result = grouped.agg(
        session_date=("session_date", "first"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        contract=("contract", "last"),
        roll_flag=("roll_flag", "any"),
        is_rth=("is_rth", "any"),
    ).reset_index(names="_bar")
    result["ts_event"] = result["_bar"]
    result = result.drop(columns="_bar")
    return result


def run_simulation(
    bars: pd.DataFrame,
    cfg: SimulationConfig,
    signal_override: Sequence[int] | None = None,
) -> SimulationResult:
    """Run a bar-by-bar simulation with delayed execution.

    A signal decided at bar ``i`` close is filled at bar ``i + lag`` open and
    marked to that bar's close. Forward-Panama-adjusted prices keep USD PnL
    continuous across rolls; roll close/reopen trades occur at the first bar
    of the roll session at that bar's open.
    """
    bars = bars.sort_values("ts_event").reset_index(drop=True)
    if bars.empty:
        empty = pd.Series(dtype=float)
        return SimulationResult(
            pd.DataFrame(),
            empty,
            empty,
            pd.DataFrame(),
            empty,
            {},
            cfg,
            pd.DataFrame(),
            pd.DataFrame(),
        )
    cursor = _BarCursor(bars)
    sizer: Sizer = build_sizer(cfg.sizer)
    state = SizerState()
    signal_state = SignalState(cfg.entry, cfg.exit, cfg.stop)
    positions = []
    pnl_values: list[float] = []
    equity_values: list[float] = []
    signals = []
    trades: list[dict[str, object]] = []
    hedge_rows = []
    pending: dict[int, tuple[int, int, str]] = {}
    held_es = held_nq = 0
    previous_es = previous_nq = 0.0
    alpha = beta = float("nan")
    session_spreads: list[float] = []
    sessions = bars["session_date"].drop_duplicates().tolist()
    unit_session_pnl = 0.0
    previous_session: date | None = None
    for i in range(len(bars)):
        cursor.index = i
        row = cursor.at(i)
        session = row["session_date"]
        new_session = i == 0 or session != bars.iloc[i - 1]["session_date"]
        if new_session:
            if previous_session is not None:
                state.unit_pnl_daily.loc[str(previous_session)] = unit_session_pnl
            unit_session_pnl = 0.0
            prior_sessions = [item for item in sessions if item < session][
                -cfg.hedge_lookback_sessions :
            ]
            history = bars[bars["session_date"].isin(prior_sessions)]
            if len(prior_sessions) >= cfg.min_hedge_sessions and len(history) >= 2:
                x = np.log(history["nq_close"].to_numpy())
                y = np.log(history["es_close"].to_numpy())
                beta, alpha = np.polyfit(x, y, 1)
                hedge_rows.append({"session_date": session, "alpha": alpha, "beta": beta})
            else:
                alpha = beta = float("nan")
            if cfg.z_reset_each_session:
                session_spreads = []
        trade_start = len(trades)
        if i in pending:
            target_es, target_nq, reason = pending.pop(i)
            held_es, held_nq = _execute_target(
                row, held_es, held_nq, target_es, target_nq, reason, cfg, trades
            )
        if bool(row.get("es_roll", False)) or bool(row.get("nq_roll", False)):
            if i == 0 or session != bars.iloc[i - 1]["session_date"]:
                previous_row = None if i == 0 else bars.iloc[i - 1]
                if held_es and (
                    previous_row is None or row["es_contract"] != previous_row["es_contract"]
                ):
                    _record_roll(row, previous_row, "ES", held_es, cfg, trades)
                if held_nq and (
                    previous_row is None or row["nq_contract"] != previous_row["nq_contract"]
                ):
                    _record_roll(row, previous_row, "NQ", held_nq, cfg, trades)
        current_pnl = _mark_pnl(
            row,
            i,
            held_es,
            held_nq,
            previous_es,
            previous_nq,
            trades[trade_start:],
        )
        current_pnl -= sum(
            float(cast(Any, trade.get("fees_usd", 0.0))) for trade in trades[trade_start:]
        )
        pnl_values.append(current_pnl)
        if i > 0:
            unit_session_pnl += (float(row["es_close"]) - previous_es) * PRODUCTS[
                "ES"
            ].multiplier_usd - (float(row["nq_close"]) - previous_nq) * PRODUCTS[
                "NQ"
            ].multiplier_usd
        previous_session = session
        previous_es, previous_nq = float(row["es_close"]), float(row["nq_close"])
        equity_values.append(cfg.initial_capital_usd + sum(pnl_values))
        spread = (
            float(np.log(row["es_close"]) - alpha - beta * np.log(row["nq_close"]))
            if np.isfinite(alpha + beta)
            else float("nan")
        )
        session_spreads.append(spread)
        window = session_spreads[-cfg.z_window :]
        z = (
            (spread - np.nanmean(window)) / np.nanstd(window, ddof=1)
            if len(window) > 1 and np.nanstd(window, ddof=1) > 0
            else float("nan")
        )
        allow_entry = not (
            cfg.mask_roll_sessions and (bool(row.get("es_roll")) or bool(row.get("nq_roll")))
        )
        old_signal = signal_state.position
        signal = (
            signal_override[i]
            if signal_override is not None and i < len(signal_override)
            else signal_state.update(z, allow_entry)
        )
        if signal_override is not None:
            signal_state.last_event = _signal_event(old_signal, signal)
            signal_state.position = signal
        if (np.isfinite(beta) or signal_override is not None) and signal != old_signal:
            sizing_beta = beta if np.isfinite(beta) else 1.0
            target = sizer.size(
                signal,
                float(row["es_close"]),
                float(row["nq_close"]),
                float(sizing_beta),
                state,
            )
            reason = signal_state.last_event or _signal_event(old_signal, signal) or "exit"
            pending[i + cfg.signal_lag_bars] = (*target, reason)
        signals.append({"ts_event": row["ts_event"], "z": z, "spread": spread, "signal": signal})
        positions.append({"ts_event": row["ts_event"], "n_es": held_es, "n_nq": held_nq})
    if held_es or held_nq:
        row = bars.iloc[-1].copy()
        row["es_open"] = row["es_close"]
        row["nq_open"] = row["nq_close"]
        trade_start = len(trades)
        held_es, held_nq = _execute_target(row, held_es, held_nq, 0, 0, "eod_final", cfg, trades)
        pnl_values[-1] -= sum(
            float(cast(Any, trade.get("fees_usd", 0.0))) for trade in trades[trade_start:]
        )
        positions[-1]["n_es"] = held_es
        positions[-1]["n_nq"] = held_nq
    pnl = pd.Series(pnl_values, index=bars["ts_event"], name="pnl")
    position_frame = pd.DataFrame(positions).set_index("ts_event")
    equity = cfg.initial_capital_usd + pnl.cumsum()
    daily = pnl.groupby(bars["session_date"].to_numpy()).sum()
    trade_frame = pd.DataFrame(trades)
    metrics = compute_metrics(pnl, daily, trade_frame, position_frame, cfg.initial_capital_usd)
    return SimulationResult(
        position_frame,
        pnl,
        equity,
        trade_frame,
        daily,
        metrics,
        cfg,
        pd.DataFrame(hedge_rows),
        pd.DataFrame(signals).set_index("ts_event"),
    )


def _execute_target(
    row: pd.Series,
    old_es: int,
    old_nq: int,
    target_es: int,
    target_nq: int,
    reason: str,
    cfg: SimulationConfig,
    trades: list[dict[str, object]],
    contract_overrides: dict[str, str] | None = None,
) -> tuple[int, int]:
    for product, old, target, reference in (
        ("ES", old_es, target_es, float(row["es_open"])),
        ("NQ", old_nq, target_nq, float(row["nq_open"])),
    ):
        delta = target - old
        if not delta:
            continue
        spec = PRODUCTS[product]
        side: Literal[1, -1] = 1 if delta > 0 else -1
        fill = cfg.costs.fill_price(spec, reference, side)
        fees = cfg.costs.fees_usd(spec, delta)
        trades.append(
            {
                "ts_event": row["ts_event"],
                "product": product,
                "contract": (contract_overrides or {}).get(
                    product, row[f"{product.lower()}_contract"]
                ),
                "side": side,
                "qty": delta,
                "fill_px": fill,
                "reference_px": reference,
                "fees_usd": fees,
                "slippage_usd": abs(fill - reference) * abs(delta) * spec.multiplier_usd,
                "reason": reason,
            }
        )
    return target_es, target_nq


def _record_roll(
    row: pd.Series,
    previous_row: pd.Series | None,
    product: str,
    quantity: int,
    cfg: SimulationConfig,
    trades: list[dict[str, object]],
) -> None:
    key = product.lower()
    old_contract = (
        row[f"{key}_contract"] if previous_row is None else previous_row[f"{key}_contract"]
    )
    new_contract = row[f"{key}_contract"]
    if product == "ES":
        _execute_target(row, quantity, 0, 0, 0, "roll", cfg, trades, {"ES": str(old_contract)})
        _execute_target(row, 0, 0, quantity, 0, "roll", cfg, trades, {"ES": str(new_contract)})
    else:
        _execute_target(row, 0, quantity, 0, 0, "roll", cfg, trades, {"NQ": str(old_contract)})
        _execute_target(row, 0, 0, 0, quantity, "roll", cfg, trades, {"NQ": str(new_contract)})


def _mark_pnl(
    row: pd.Series,
    i: int,
    es_qty: int,
    nq_qty: int,
    previous_es: float,
    previous_nq: float,
    fills: list[dict[str, object]],
) -> float:
    if i == 0:
        return 0.0
    total = 0.0
    for product, quantity, previous, close in (
        ("ES", es_qty, previous_es, float(row["es_close"])),
        ("NQ", nq_qty, previous_nq, float(row["nq_close"])),
    ):
        product_fills = [fill for fill in fills if fill["product"] == product]
        running_qty = quantity
        running_previous = previous
        for fill in reversed(product_fills):
            running_qty -= int(cast(Any, fill["qty"]))
        for fill in product_fills:
            fill_price = float(cast(Any, fill["fill_px"]))
            delta = int(cast(Any, fill["qty"]))
            old_qty = running_qty
            total += old_qty * PRODUCTS[product].multiplier_usd * (fill_price - running_previous)
            running_qty += delta
            running_previous = fill_price
        total += running_qty * PRODUCTS[product].multiplier_usd * (close - running_previous)
    return total
