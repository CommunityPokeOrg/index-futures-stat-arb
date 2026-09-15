"""Deterministic anchored walk-forward evaluation for basis simulations."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .contracts import PRODUCTS
from .execution.engine import SimulationConfig, SimulationResult, run_simulation


@dataclass(frozen=True)
class SearchSpace:
    kalman_delta: tuple[float, float] = (1e-7, 1e-3)
    kalman_obs_var: tuple[float, float] = (1e-8, 1e-4)
    entry: tuple[float, float] = (1.0, 3.0)
    exit_ratio: tuple[float, float] = (0.0, 0.6)
    stop_ratio: tuple[float, float] = (1.5, 3.0)
    half_life_min_bars: tuple[float, float] = (0.5, 3.0)
    half_life_max_bars: tuple[float, float] = (5.0, 60.0)
    max_holding_half_lives: tuple[float, float] = (1.0, 4.0)
    min_edge_cost_multiple: tuple[float, float] = (0.0, 3.0)
    ofi_threshold: tuple[float, float] = (0.0, 0.8)
    coint_pvalue_gate: tuple[float, float] = (0.05, 0.2)


@dataclass(frozen=True)
class WalkForwardConfig:
    n_trials: int = 48
    n_folds: int = 5
    min_train_sessions: int = 500
    min_trades_per_fold: int = 3
    seed: int = 0
    space: SearchSpace = field(default_factory=SearchSpace)


@dataclass(frozen=True)
class Fold:
    index: int
    train_sessions: tuple[date, ...]
    test_sessions: tuple[date, ...]

    @property
    def test_start(self) -> date:
        return self.test_sessions[0]

    @property
    def test_end(self) -> date:
        return self.test_sessions[-1]


@dataclass
class WalkForwardResult:
    base: SimulationConfig
    config: WalkForwardConfig
    trials: list[dict[str, Any]]
    folds: list[Fold]
    fold_metrics: dict[int, list[dict[str, Any]]]
    simulations: dict[int, SimulationResult] = field(default_factory=dict)


_PARAMETER_NAMES = (
    "kalman_delta",
    "kalman_obs_var",
    "entry",
    "exit_ratio",
    "stop_ratio",
    "half_life_min_bars",
    "half_life_max_bars",
    "max_holding_half_lives",
    "min_edge_cost_multiple",
    "ofi_threshold",
    "coint_pvalue_gate",
)


def _model_params(params: dict[str, Any]) -> dict[str, float | None]:
    return {name: params.get(name) for name in _PARAMETER_NAMES}


def _in_sessions(value: Any, sessions: tuple[date, ...]) -> bool:
    return value in sessions


def _session_mask(index: Any, sessions: tuple[date, ...]) -> list[bool]:
    return [_in_sessions(value, sessions) for value in index]


def _base_params(base: SimulationConfig) -> dict[str, float | None]:
    return {
        "kalman_delta": base.kalman_delta,
        "kalman_obs_var": base.kalman_obs_var,
        "entry": base.entry,
        "exit_ratio": base.exit / base.entry if base.entry else 0.0,
        "stop_ratio": base.stop / base.entry if base.stop is not None and base.entry else None,
        "half_life_min_bars": base.half_life_min_bars,
        "half_life_max_bars": base.half_life_max_bars,
        "max_holding_half_lives": base.max_holding_half_lives,
        "min_edge_cost_multiple": base.min_edge_cost_multiple,
        "ofi_threshold": base.ofi_threshold,
        "coint_pvalue_gate": base.coint_pvalue_gate,
    }


def sample_trials(
    space: SearchSpace,
    n_trials: int,
    seed: int,
    base: SimulationConfig | None = None,
) -> list[dict[str, float | None]]:
    """Sample deterministic parameter dictionaries; trial zero is the base."""
    if n_trials < 1:
        raise ValueError("n_trials must be positive")
    rng = np.random.default_rng(seed)
    trials: list[dict[str, float | None]] = [
        _base_params(base)
        if base is not None
        else {
            "kalman_delta": 1e-5,
            "kalman_obs_var": 1e-6,
            "entry": 2.0,
            "exit_ratio": 0.25,
            "stop_ratio": 2.0,
            "half_life_min_bars": 1.0,
            "half_life_max_bars": 30.0,
            "max_holding_half_lives": 2.0,
            "min_edge_cost_multiple": 0.0,
            "ofi_threshold": None,
            "coint_pvalue_gate": 0.1,
        }
    ]

    def uniform(bounds: tuple[float, float]) -> float:
        return float(rng.uniform(*bounds))

    def log_uniform(bounds: tuple[float, float]) -> float:
        return float(np.exp(rng.uniform(np.log(bounds[0]), np.log(bounds[1]))))

    while len(trials) < n_trials:
        entry = uniform(space.entry)
        stop = entry * uniform(space.stop_ratio)
        exit_value = entry * uniform(space.exit_ratio)
        if not exit_value < entry < stop:
            continue
        trial = {
            "kalman_delta": log_uniform(space.kalman_delta),
            "kalman_obs_var": log_uniform(space.kalman_obs_var),
            "entry": entry,
            "exit_ratio": exit_value / entry,
            "stop_ratio": stop / entry,
            "half_life_min_bars": uniform(space.half_life_min_bars),
            "half_life_max_bars": uniform(space.half_life_max_bars),
            "max_holding_half_lives": uniform(space.max_holding_half_lives),
            "min_edge_cost_multiple": uniform(space.min_edge_cost_multiple),
            "ofi_threshold": uniform(space.ofi_threshold) if rng.random() >= 0.3 else None,
            "coint_pvalue_gate": uniform(space.coint_pvalue_gate),
        }
        min_half_life = trial["half_life_min_bars"]
        max_half_life = trial["half_life_max_bars"]
        if min_half_life is None or max_half_life is None or min_half_life >= max_half_life:
            continue
        trials.append(trial)
    return trials


def make_folds(session_dates: Sequence[date], cfg: WalkForwardConfig) -> list[Fold]:
    """Create anchored, contiguous test folds after the training warm-up."""
    sessions = list(dict.fromkeys(session_dates))
    if len(sessions) <= cfg.min_train_sessions:
        raise ValueError("not enough sessions after min_train_sessions")
    test_sessions = sessions[cfg.min_train_sessions :]
    partitions = np.array_split(np.asarray(test_sessions, dtype=object), cfg.n_folds)
    folds: list[Fold] = []
    for index, partition in enumerate(partitions):
        values = tuple(partition.tolist())
        if not values:
            continue
        test_start = cfg.min_train_sessions + sum(len(item) for item in partitions[:index])
        folds.append(Fold(index, tuple(sessions[:test_start]), values))
    return folds


def _trial_config(base: SimulationConfig, params: dict[str, float | None]) -> SimulationConfig:
    def number(name: str) -> float:
        value = params[name]
        if value is None:
            raise ValueError(f"missing trial parameter {name}")
        return float(value)

    entry = number("entry")
    stop_ratio = params["stop_ratio"]
    return replace(
        base,
        kalman_delta=number("kalman_delta"),
        kalman_obs_var=number("kalman_obs_var"),
        entry=entry,
        exit=entry * number("exit_ratio"),
        stop=None if stop_ratio is None else entry * float(stop_ratio),
        half_life_min_bars=number("half_life_min_bars"),
        half_life_max_bars=(
            None if params["half_life_max_bars"] is None else number("half_life_max_bars")
        ),
        max_holding_half_lives=(
            None if params["max_holding_half_lives"] is None else number("max_holding_half_lives")
        ),
        min_edge_cost_multiple=(
            None if params["min_edge_cost_multiple"] is None else number("min_edge_cost_multiple")
        ),
        ofi_threshold=None if params["ofi_threshold"] is None else number("ofi_threshold"),
        coint_pvalue_gate=(
            None if params["coint_pvalue_gate"] is None else number("coint_pvalue_gate")
        ),
    )


def _fold_metric(
    result: SimulationResult,
    cfg: SimulationConfig,
    fold: Fold,
) -> dict[str, Any]:
    daily = result.daily_pnl
    mask = daily.index.map(lambda value: value in fold.test_sessions)
    selected_daily = daily.loc[mask]
    pnl = selected_daily.fillna(0.0)
    sharpe = (
        float(pnl.mean() / pnl.std(ddof=1) * np.sqrt(252))
        if len(pnl) > 1 and pnl.std() > 0
        else 0.0
    )
    equity = cfg.initial_capital_usd + pnl.cumsum()
    drawdown = equity - equity.cummax()
    trades = result.trades
    if len(trades) and "ts_event" in trades:
        timestamps = pd.to_datetime(trades["ts_event"], utc=True)
        trade_mask = timestamps.dt.date.isin(fold.test_sessions)
        fold_trades = trades.loc[trade_mask]
    else:
        fold_trades = trades.iloc[0:0]
    positions = result.positions
    position_dates = pd.Index(list(pd.to_datetime(positions.index, utc=True).date))
    position_mask = position_dates.isin(list(fold.test_sessions))
    active = positions.loc[position_mask].fillna(0)[["n_a", "n_b"]].abs().sum(axis=1) > 0
    round_trips = 0
    if len(active):
        round_trips = int((active & ~active.shift(1, fill_value=False)).sum())
    years = max(len(fold.test_sessions) / 252.0, 1.0 / 252.0)
    turnover_notional = 0.0
    if len(fold_trades) and {"qty", "fill_px", "product"} <= set(fold_trades.columns):
        turnover_notional = sum(
            abs(
                float(cast(Any, values[0]))
                * float(cast(Any, values[1]))
                * PRODUCTS[str(cast(Any, values[2]))].multiplier_usd
            )
            for values in fold_trades[["qty", "fill_px", "product"]].to_numpy()
        )
    return {
        "sharpe": sharpe,
        "max_dd": float(-drawdown.min()) if len(drawdown) else 0.0,
        "win_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
        "round_trips": round_trips,
        "trades": int(len(fold_trades)),
        "turnover": turnover_notional / cfg.initial_capital_usd / years,
        "fees_slippage": float(
            fold_trades.get("fees_usd", pd.Series(dtype=float)).sum()
            + fold_trades.get("slippage_usd", pd.Series(dtype=float)).sum()
        ),
    }


def evaluate_trials(
    base: SimulationConfig,
    bars: pd.DataFrame,
    cfg: WalkForwardConfig,
    run_fn: Callable[[pd.DataFrame, SimulationConfig], SimulationResult] = run_simulation,
) -> WalkForwardResult:
    sessions = list(bars["session_date"].drop_duplicates())
    folds = make_folds(sessions, cfg)
    trials = sample_trials(cfg.space, cfg.n_trials, cfg.seed, base)
    fold_metrics: dict[int, list[dict[str, Any]]] = {}
    simulations: dict[int, SimulationResult] = {}
    for trial_id, params in enumerate(trials):
        trial_config = _trial_config(base, params)
        simulation = run_fn(bars, trial_config)
        simulations[trial_id] = simulation
        fold_metrics[trial_id] = [_fold_metric(simulation, trial_config, fold) for fold in folds]
        params["trial_id"] = trial_id
        params.update(
            {
                "in_sample_sharpe": float(simulation.metrics.get("sharpe", 0.0)),
                "in_sample_pnl": float(simulation.metrics.get("total_pnl_usd", 0.0)),
            }
        )
        for fold, metrics in zip(folds, fold_metrics[trial_id], strict=True):
            for key, value in metrics.items():
                params[f"fold_{fold.index}_{key}"] = value
        values = [item["sharpe"] for item in fold_metrics[trial_id]]
        params["median_oos_sharpe"] = float(np.median(values)) if values else 0.0
        params["min_oos_sharpe"] = float(min(values)) if values else 0.0
    return WalkForwardResult(base, cfg, trials, folds, fold_metrics, simulations)


def select(result: WalkForwardResult, fold_index: int) -> int:
    """Select by median minus half IQR of prior train-fold Sharpes."""
    if fold_index == 0:
        return 0
    candidates: list[tuple[float, int]] = []
    for trial_id, metrics in result.fold_metrics.items():
        train = metrics[:fold_index]
        if not all(item["trades"] >= result.config.min_trades_per_fold for item in train):
            continue
        values = np.asarray([item["sharpe"] for item in train], dtype=float)
        score = float(
            np.median(values) - 0.5 * (np.quantile(values, 0.75) - np.quantile(values, 0.25))
        )
        candidates.append((score, trial_id))
    return min(candidates, key=lambda value: (-value[0], value[1]))[1] if candidates else 0


def _stitched_metrics(result: WalkForwardResult) -> dict[str, float]:
    pieces: list[pd.Series] = []
    turnover = 0.0
    fees_slippage = 0.0
    for fold in result.folds:
        trial_id = select(result, fold.index)
        daily = result.simulations[trial_id].daily_pnl
        sessions = fold.test_sessions
        pieces.append(daily.loc[_session_mask(daily.index, sessions)])
        turnover += float(result.fold_metrics[trial_id][fold.index]["turnover"])
        fees_slippage += float(result.fold_metrics[trial_id][fold.index]["fees_slippage"])
    pnl = pd.concat(pieces) if pieces else pd.Series(dtype=float)
    sharpe = (
        float(pnl.mean() / pnl.std(ddof=1) * np.sqrt(252)) if len(pnl) > 1 and pnl.std() else 0.0
    )
    equity = result.base.initial_capital_usd + pnl.cumsum()
    return {
        "sharpe": sharpe,
        "max_dd": float(-(equity - equity.cummax()).min()) if len(equity) else 0.0,
        "win_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
        "bars": float(len(pnl)),
        "turnover": turnover,
        "fees_slippage": fees_slippage,
    }


def _base_oos_metrics(result: WalkForwardResult) -> dict[str, float]:
    pieces = [
        result.simulations[0].daily_pnl.loc[
            _session_mask(result.simulations[0].daily_pnl.index, fold.test_sessions)
        ]
        for fold in result.folds
    ]
    pnl = pd.concat(pieces) if pieces else pd.Series(dtype=float)
    sharpe = (
        float(pnl.mean() / pnl.std(ddof=1) * np.sqrt(252)) if len(pnl) > 1 and pnl.std() else 0.0
    )
    equity = result.base.initial_capital_usd + pnl.cumsum()
    return {
        "sharpe": sharpe,
        "max_dd": float(-(equity - equity.cummax()).min()) if len(equity) else 0.0,
        "win_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
        "bars": float(len(pnl)),
    }


def summarize(result: WalkForwardResult) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    trial_frame = pd.DataFrame(result.trials).sort_values("trial_id").reset_index(drop=True)
    fold_rows: list[dict[str, Any]] = []
    for fold in result.folds:
        trial_id = select(result, fold.index)
        fold_rows.append(
            {
                "fold": fold.index,
                "train_start": fold.train_sessions[0],
                "train_end": fold.train_sessions[-1],
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "selected_trial": trial_id,
                **result.fold_metrics[trial_id][fold.index],
            }
        )
    fold_frame = pd.DataFrame(fold_rows)
    oos_values = trial_frame["median_oos_sharpe"].to_numpy()
    trial_ids = [int(cast(Any, value)) for value in trial_frame["trial_id"].tolist()]
    eligible = [
        trial_id
        for trial_id in trial_ids
        if sum(result.fold_metrics[trial_id][i]["trades"] > 0 for i in range(len(result.folds)))
        >= int(np.ceil(len(result.folds) / 2))
    ]
    recommended = (
        max(eligible, key=lambda trial_id: (oos_values[trial_id], -trial_id)) if eligible else 0
    )
    ranks = []
    for fold in result.folds:
        ordered = sorted(
            result.fold_metrics,
            key=lambda trial_id: result.fold_metrics[trial_id][fold.index]["sharpe"],
            reverse=True,
        )
        ranks.append(ordered.index(recommended) < max(1, int(np.ceil(len(ordered) / 4))))
    best_index = int(cast(Any, trial_frame["in_sample_sharpe"].astype(float).idxmax()))
    best_is = int(cast(Any, trial_frame.iloc[best_index]["trial_id"]))
    summary = {
        "stitched_oos": _stitched_metrics(result),
        "recommended_trial": recommended,
        "recommended_rank_stability": int(sum(ranks)),
        "best_in_sample_trial": best_is,
        "best_in_sample_oos_sharpe": float(
            cast(Any, trial_frame.loc[best_is, "median_oos_sharpe"])
        ),
        "base_trial": 0,
    }
    return trial_frame, fold_frame, summary


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{value:.10f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_artifacts(result: WalkForwardResult, out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    trial_frame, fold_frame, summary = summarize(result)
    trial_frame = trial_frame.reindex(sorted(trial_frame.columns), axis=1)
    fold_frame = fold_frame.reindex(sorted(fold_frame.columns), axis=1)
    trial_frame.to_csv(out / "trials.csv", index=False, float_format="%.10f")
    fold_frame.to_csv(out / "folds.csv", index=False, float_format="%.10f")
    selected_trials = [
        {"fold": fold.index, "trial_id": select(result, fold.index)} for fold in result.folds
    ]
    recommended_trial = summary["recommended_trial"]
    selection = {
        "rule": (
            "median(train Sharpe) - 0.5*IQR(train Sharpe), "
            "min trades per train fold; ties lowest trial id"
        ),
        **summary,
        "selected_trials": selected_trials,
        "recommended_model": {
            "trial_id": recommended_trial,
            "params": _model_params(result.trials[recommended_trial]),
            "fold_sharpes": [
                result.fold_metrics[recommended_trial][i]["sharpe"]
                for i in range(len(result.folds))
            ],
            "rank_stability": summary["recommended_rank_stability"],
        },
        "recommended_params": _model_params(result.trials[recommended_trial]),
        "recommended_fold_sharpes": [
            result.fold_metrics[summary["recommended_trial"]][i]["sharpe"]
            for i in range(len(result.folds))
        ],
    }
    (out / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True, default=str) + "\n"
    )
    top = trial_frame.nlargest(10, "median_oos_sharpe")
    summary_text = "# Walk-forward evaluation\n\n"
    summary_text += "## Top 10 by median OOS Sharpe\n\n" + _markdown_table(top) + "\n\n"
    summary_text += "## Stitched OOS vs base vs best in-sample\n\n"
    benchmark = pd.DataFrame(
        [
            {"model": "stitched_oos", **summary["stitched_oos"]},
            {"model": "base_config", **_base_oos_metrics(result)},
            {"model": "best_in_sample", "sharpe": summary["best_in_sample_oos_sharpe"]},
        ]
    )
    summary_text += _markdown_table(benchmark) + "\n"
    (out / "summary.md").write_text(summary_text)
    _plots(result, trial_frame, out)
    return selection


def _plots(result: WalkForwardResult, trial_frame: pd.DataFrame, out: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(trial_frame["median_oos_sharpe"], bins=min(20, max(5, len(trial_frame) // 3)))
    plt.xlabel("Median OOS Sharpe")
    plt.tight_layout()
    plt.savefig(out / "oos_sharpe_hist.png", dpi=100)
    plt.close()
    plt.figure(figsize=(8, 4))
    base = result.simulations[0].daily_pnl.cumsum()
    plt.plot(base.index, base, label="base trial")
    stitched_parts = []
    for fold in result.folds:
        trial = result.simulations[select(result, fold.index)].daily_pnl
        sessions = fold.test_sessions
        stitched_parts.append(trial.loc[_session_mask(trial.index, sessions)])
        plt.axvline(fold.index, color="black", alpha=0.2)
    if stitched_parts:
        stitched = pd.concat(stitched_parts)
        plt.plot(stitched.index, stitched.cumsum(), label="stitched OOS")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "stitched_equity.png", dpi=100)
    plt.close()
    params = [
        "kalman_delta",
        "kalman_obs_var",
        "entry",
        "exit_ratio",
        "stop_ratio",
        "half_life_min_bars",
        "half_life_max_bars",
        "max_holding_half_lives",
        "min_edge_cost_multiple",
        "ofi_threshold",
        "coint_pvalue_gate",
    ]
    fig, axes = plt.subplots(3, 4, figsize=(10, 7))
    for axis, param in zip(axes.flat, params, strict=False):
        axis.scatter(trial_frame[param], trial_frame["median_oos_sharpe"], s=8)
        axis.set_title(param)
    for axis in axes.flat[len(params) :]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(out / "param_vs_oos.png", dpi=100)
    plt.close(fig)
