"""Incremental, no-lookahead ES/NQ simulation engine."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from ..cointegration import adf_test
from ..continuous import Adjust, build_continuous
from ..contracts import PRODUCTS, list_contracts
from ..ingest.databento import read_partitioned
from ..ou import fit_ou
from ..rolls import RollConfig, build_roll_calendar, daily_from_bars, joint_roll_calendar
from .costs import CostModel
from .hedge import HedgeMethod, KalmanHedge, rolling_engle_granger
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
    hedge_method: HedgeMethod = "ols"
    kalman_delta: float = 1e-5
    kalman_obs_var: float = 1e-4
    coint_pvalue_gate: float | None = None
    threshold_mode: Literal["fixed", "ou"] = "fixed"
    half_life_min_bars: float = 1.0
    half_life_max_bars: float | None = None
    max_holding_half_lives: float | None = None
    ou_min_obs: int = 30
    adjust: Adjust = "panama"
    roll: RollConfig = field(default_factory=RollConfig)
    initial_capital_usd: float = 1_000_000.0
    seed: int = 0
    costs: CostModel = field(default_factory=CostModel)
    sizer: SizerSpec = field(default_factory=SizerSpec)

    def __post_init__(self) -> None:
        if self.hedge_method not in {"ols", "kalman", "rolling_eg"}:
            raise ValueError(f"unsupported hedge method: {self.hedge_method!r}")
        if self.threshold_mode not in {"fixed", "ou"}:
            raise ValueError(f"unsupported threshold mode: {self.threshold_mode!r}")


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
    max_holding_bars: float | None = None
    bars_held: int = 0

    def update(self, value: float, allow_entry: bool = True) -> int:
        previous = self.position
        self.last_event = None
        if self.position:
            self.bars_held += 1
            if self.max_holding_bars is not None and self.bars_held > self.max_holding_bars:
                self.position = 0
                self.last_event = "time_stop"
                self.max_holding_bars = None
                self.bars_held = 0
                return 0
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
                self.bars_held = 0
            elif value < -self.entry:
                self.position = 1
                self.bars_held = 0
        elif self.position and absolute < self.exit:
            self.position = 0
            self.max_holding_bars = None
            self.bars_held = 0
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
    del product
    frame = frame.copy()
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
    positions: list[dict[str, object]] = []
    pnl_values: list[float] = []
    signals: list[dict[str, object]] = []
    trades: list[dict[str, object]] = []
    hedge_rows: list[dict[str, object]] = []
    pending: dict[int, tuple[int, int, str]] = {}
    held_es = held_nq = 0
    previous_es = previous_nq = 0.0
    previous_beta = float("nan")
    alpha = beta = float("nan")
    kalman: KalmanHedge | None = None
    session_spreads: list[float] = []
    kalman_innovations: list[float] = []
    sessions = bars["session_date"].drop_duplicates().tolist()
    unit_session_pnl = 0.0
    previous_session: date | None = None
    session_entry_gate = True
    session_gate_reason: str | None = None
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
            session_gate_reason = None
            if len(prior_sessions) >= cfg.min_hedge_sessions and len(history) >= 2:
                x = np.log(history["nq_close"].to_numpy())
                y = np.log(history["es_close"].to_numpy())
                if cfg.hedge_method == "rolling_eg":
                    fit = rolling_engle_granger(y, x)
                    alpha, beta, pvalue = fit.alpha, fit.beta, fit.pvalue
                else:
                    beta, alpha = np.polyfit(x, y, 1)
                    pvalue = float("nan")
                    if cfg.coint_pvalue_gate is not None:
                        pvalue = rolling_engle_granger(y, x).pvalue
                if cfg.hedge_method == "kalman":
                    if kalman is None and np.isfinite(alpha + beta):
                        kalman = KalmanHedge(
                            delta=cfg.kalman_delta,
                            obs_var=cfg.kalman_obs_var,
                            beta=float(beta),
                            alpha=float(alpha),
                        )
                    if kalman is not None:
                        alpha, beta = kalman.alpha, kalman.beta
                    if cfg.coint_pvalue_gate is not None:
                        try:
                            pvalue = adf_test(pd.Series(kalman_innovations[-cfg.z_window :]))[
                                "pvalue"
                            ]
                        except Exception:
                            pvalue = float("nan")
                if cfg.coint_pvalue_gate is not None and (
                    not np.isfinite(pvalue) or pvalue >= cfg.coint_pvalue_gate
                ):
                    session_entry_gate = False
                    session_gate_reason = "cointegration"
                else:
                    session_entry_gate = True
                if cfg.hedge_method != "kalman":
                    hedge_rows.append(
                        {
                            "session_date": session,
                            "alpha": alpha,
                            "beta": beta,
                            "pvalue": pvalue,
                        }
                    )
                if cfg.hedge_method != "kalman" and not cfg.z_reset_each_session:
                    prior_spreads = y - alpha - beta * x
                    session_spreads = list(prior_spreads[-cfg.z_window :])
                elif cfg.z_reset_each_session:
                    session_spreads = []
            else:
                alpha = beta = float("nan")
                session_entry_gate = False
                session_gate_reason = "hedge_history"
                if cfg.z_reset_each_session:
                    session_spreads = []
            if cfg.hedge_method == "kalman" and kalman is not None:
                alpha, beta = kalman.alpha, kalman.beta
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
            hedge_unit = (
                previous_beta
                * previous_es
                * PRODUCTS["ES"].multiplier_usd
                / (previous_nq * PRODUCTS["NQ"].multiplier_usd)
                if np.isfinite(previous_beta) and previous_nq
                else 1.0
            )
            unit_session_pnl += (float(row["es_close"]) - previous_es) * PRODUCTS[
                "ES"
            ].multiplier_usd - hedge_unit * (float(row["nq_close"]) - previous_nq) * PRODUCTS[
                "NQ"
            ].multiplier_usd
        previous_session = session
        previous_es, previous_nq = float(row["es_close"]), float(row["nq_close"])
        log_es = float(np.log(row["es_close"]))
        log_nq = float(np.log(row["nq_close"]))
        if kalman is not None:
            prediction, _innovation_var = kalman.predict(log_nq)
            spread = log_es - prediction
            kalman.update(log_es, log_nq)
            alpha, beta = kalman.alpha, kalman.beta
            kalman_innovations.append(spread)
            hedge_rows.append(
                {
                    "session_date": session,
                    "ts_event": row["ts_event"],
                    "alpha": alpha,
                    "beta": beta,
                }
            )
        else:
            spread = (
                float(log_es - alpha - beta * log_nq) if np.isfinite(alpha + beta) else float("nan")
            )
        prior_window = np.asarray(session_spreads[-cfg.z_window :], dtype=float)
        half_life = float("nan")
        ou_mu = float("nan")
        ou_sigma = float("nan")
        threshold_gate = True
        if cfg.threshold_mode == "ou":
            if len(prior_window) >= cfg.ou_min_obs:
                try:
                    params = fit_ou(prior_window, dt=1.0)
                    half_life = params.half_life
                    ou_mu = params.mu
                    ou_sigma = params.stationary_std
                    threshold_gate = (
                        np.isfinite(half_life)
                        and half_life >= cfg.half_life_min_bars
                        and (cfg.half_life_max_bars is None or half_life <= cfg.half_life_max_bars)
                        and np.isfinite(ou_mu)
                        and np.isfinite(ou_sigma)
                        and ou_sigma > 0
                    )
                    z = (spread - ou_mu) / ou_sigma if threshold_gate else float("nan")
                except (ValueError, FloatingPointError):
                    threshold_gate = False
                    z = float("nan")
            else:
                threshold_gate = False
                z = float("nan")
        else:
            current_window = np.asarray([*prior_window, spread], dtype=float)
            finite_window = current_window[np.isfinite(current_window)]
            if len(finite_window) > 1:
                mean = float(np.mean(finite_window))
                std = float(np.std(finite_window, ddof=1))
                z = (spread - mean) / std if std > 0 else float("nan")
            else:
                z = float("nan")
        session_spreads.append(spread)
        gate_reasons: list[str] = []
        if not session_entry_gate:
            gate_reasons.append(session_gate_reason or "cointegration")
        if not threshold_gate:
            gate_reasons.append("half_life")
        allow_entry = (
            session_entry_gate
            and threshold_gate
            and not (
                cfg.mask_roll_sessions and (bool(row.get("es_roll")) or bool(row.get("nq_roll")))
            )
        )
        gated = not allow_entry
        gate_reason = ",".join(gate_reasons) if gate_reasons else None
        old_signal = signal_state.position
        signal = (
            signal_override[i]
            if signal_override is not None and i < len(signal_override)
            else signal_state.update(z, allow_entry)
        )
        if signal_override is not None:
            signal_state.last_event = _signal_event(old_signal, signal)
            signal_state.position = signal
        if (
            signal_override is None
            and old_signal == 0
            and signal
            and cfg.max_holding_half_lives is not None
            and np.isfinite(half_life)
        ):
            signal_state.max_holding_bars = cfg.max_holding_half_lives * half_life
            signal_state.bars_held = 0
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
        signals.append(
            {
                "ts_event": row["ts_event"],
                "session_date": session,
                "z": z,
                "spread": spread,
                "signal": signal,
                "half_life": half_life,
                "ou_mu": ou_mu,
                "ou_sigma": ou_sigma,
                "gated_bars": gated,
                "gate_reason": gate_reason,
            }
        )
        positions.append({"ts_event": row["ts_event"], "n_es": held_es, "n_nq": held_nq})
        previous_beta = beta
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
    signal_frame = pd.DataFrame(signals).set_index("ts_event")
    metrics = compute_metrics(pnl, daily, trade_frame, position_frame, cfg.initial_capital_usd)
    gated_sessions = signal_frame.groupby("session_date")["gated_bars"].any()
    metrics["entry_gated_fraction"] = float(gated_sessions.mean()) if len(gated_sessions) else 0.0
    reasons = signal_frame["gate_reason"].fillna("")
    metrics["cointegration_gated_fraction"] = (
        float(reasons.str.contains("cointegration").mean()) if len(reasons) else 0.0
    )
    metrics["hedge_history_gated_fraction"] = (
        float(reasons.str.contains("hedge_history").mean()) if len(reasons) else 0.0
    )
    metrics["half_life_gated_fraction"] = (
        float(reasons.str.contains("half_life").mean()) if len(reasons) else 0.0
    )
    return SimulationResult(
        position_frame,
        pnl,
        equity,
        trade_frame,
        daily,
        metrics,
        cfg,
        pd.DataFrame(hedge_rows),
        signal_frame,
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
