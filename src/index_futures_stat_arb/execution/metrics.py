"""Performance metrics for execution simulations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(
    pnl_bar: pd.Series,
    daily_pnl: pd.Series,
    trades: pd.DataFrame,
    positions: pd.DataFrame,
    initial_capital: float,
) -> dict[str, float | int]:
    if "n_a" not in positions:
        positions = positions.rename(columns={"n_es": "n_a", "n_nq": "n_b"})
    pnl = pnl_bar.fillna(0.0)
    daily = daily_pnl.fillna(0.0)
    total = float(pnl.sum())
    equity = initial_capital + pnl.cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    max_dd = float(-drawdown.min()) if len(drawdown) else 0.0
    max_dd_pct = max_dd / initial_capital * 100 if initial_capital else 0.0
    std = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
    downside = daily[daily < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sharpe = float(daily.mean() / std * np.sqrt(252)) if std > 0 else 0.0
    sortino = float(daily.mean() / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0
    ann_return = (
        ((equity.iloc[-1] / initial_capital) ** (252 / len(daily)) - 1) * 100
        if initial_capital and len(daily)
        else 0.0
    )
    round_trip_pnl = _round_trip_pnl(pnl, positions)
    wins = [value for value in round_trip_pnl if value > 0]
    losses = [value for value in round_trip_pnl if value < 0]
    gross_loss = abs(sum(losses))
    return {
        "total_pnl_usd": total,
        "total_return_pct": total / initial_capital * 100 if initial_capital else 0.0,
        "n_sessions": int(len(daily)),
        "ann_return_pct": float(ann_return),
        "ann_vol_pct": float(daily.std(ddof=1) / initial_capital * np.sqrt(252) * 100)
        if initial_capital and len(daily) > 1
        else 0.0,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown_usd": max_dd,
        "max_drawdown_pct": max_dd_pct,
        "calmar": ann_return / max_dd_pct if max_dd_pct else 0.0,
        "n_trades": int(len(trades)),
        "n_round_trips": len(round_trip_pnl),
        "win_rate": len(wins) / len(round_trip_pnl) if round_trip_pnl else 0.0,
        "avg_round_trip_pnl_usd": float(np.mean(round_trip_pnl)) if round_trip_pnl else 0.0,
        "profit_factor": sum(wins) / gross_loss if gross_loss else 0.0,
        "total_fees_usd": float(trades["fees_usd"].sum()) if "fees_usd" in trades else 0.0,
        "total_slippage_usd": float(trades["slippage_usd"].sum())
        if "slippage_usd" in trades
        else 0.0,
        "turnover_contracts": float(trades["qty"].abs().sum()) if "qty" in trades else 0.0,
        "exposure_frac": float((positions[["n_a", "n_b"]].fillna(0).abs().sum(axis=1) > 0).mean())
        if len(positions)
        else 0.0,
        "avg_holding_bars": float(
            (positions[["n_a", "n_b"]].fillna(0).abs().sum(axis=1) > 0).sum() / len(round_trip_pnl)
        )
        if round_trip_pnl
        else 0.0,
        "max_gross_contracts": float(positions[["n_a", "n_b"]].fillna(0).abs().sum(axis=1).max())
        if len(positions)
        else 0.0,
        "max_gross_units_a": float(positions["n_a"].abs().max()) if len(positions) else 0.0,
        "max_gross_units_b": float(positions["n_b"].abs().max()) if len(positions) else 0.0,
        "max_gross_notional_usd": float(
            positions[["notional_a", "notional_b"]].fillna(0).sum(axis=1).max()
        )
        if {"notional_a", "notional_b"} <= set(positions.columns) and len(positions)
        else 0.0,
    }


def _round_trip_pnl(pnl: pd.Series, positions: pd.DataFrame) -> list[float]:
    active = positions.fillna(0).abs().sum(axis=1) > 0
    values: list[float] = []
    start: int | None = None
    for index, is_active in enumerate(active.to_numpy()):
        if is_active and start is None:
            start = index
        elif not is_active and start is not None:
            values.append(float(pnl.iloc[start:index].sum()))
            start = None
    if start is not None:
        values.append(float(pnl.iloc[start:].sum()))
    return values
