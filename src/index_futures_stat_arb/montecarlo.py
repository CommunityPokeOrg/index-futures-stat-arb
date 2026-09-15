"""Monte Carlo harness for power and stress tests, never evidence of real edge.

The bootstrap mode resamples observed PnL, synthetic mode exercises the existing
simulation engine on generated paths, and deflate mode estimates multiple-testing
adjustments. These outputs are harness diagnostics only and must not be presented
as evidence of profitability or a real trading edge.

The deflated Sharpe calculations follow Bailey and López de Prado, *The
Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and
Non-Normality* (2014). The probabilistic Sharpe ratio uses the standard
skewness/kurtosis correction:

``PSR = Phi((SR - SR0) * sqrt(T - 1) /
            sqrt(1 - skew*SR + (kurtosis - 1)*SR**2/4))``.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

from .basis import time_to_expiry_years, trailing_dividend_yield
from .config import load_simulation_config, load_yahoo_config
from .execution.engine import SimulationConfig, run_simulation
from .ingest.yahoo import (
    YahooError,
    build_pair_bars,
    fetch_yahoo_bars,
    fetch_yahoo_dividends,
    fetch_yahoo_rate,
)

DISCLAIMER = "SYNTHETIC / RESAMPLED — harness test, not evidence of edge"


def circular_block_bootstrap(
    values: Sequence[float] | np.ndarray,
    n_samples: int,
    block_len: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a circular block bootstrap sample with contiguous source blocks."""
    source = np.asarray(values)
    if source.ndim != 1 or len(source) == 0:
        raise ValueError("values must be a non-empty one-dimensional sequence")
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    length = block_len or int(math.ceil(len(source) ** (1.0 / 3.0)))
    if length < 1:
        raise ValueError("block_len must be positive")
    output: list[Any] = []
    while len(output) < n_samples:
        start = int(rng.integers(0, len(source)))
        take = min(length, n_samples - len(output))
        output.extend(source[(start + np.arange(take)) % len(source)].tolist())
    return np.asarray(output[:n_samples])


def _annualized_sharpe(pnl: Sequence[float] | np.ndarray) -> float:
    values = np.asarray(pnl, dtype=float)
    if len(values) < 2 or float(values.std(ddof=1)) == 0:
        return 0.0
    return float(values.mean() / values.std(ddof=1) * np.sqrt(252.0))


def _max_drawdown_pct(pnl: Sequence[float] | np.ndarray, capital: float = 1_000_000.0) -> float:
    values = np.asarray(pnl, dtype=float)
    if len(values) == 0 or capital == 0:
        return 0.0
    equity = capital + np.cumsum(values)
    return float(-np.min(equity - np.maximum.accumulate(equity)) / capital * 100.0)


def _path_summary(pnl: Sequence[float] | np.ndarray) -> dict[str, float]:
    values = np.asarray(pnl, dtype=float)
    return {
        "sharpe": _annualized_sharpe(values),
        "pnl": float(values.sum()),
        "max_dd_pct": _max_drawdown_pct(values),
        "loss_probability": float(values.sum() < 0),
    }


def _spawn_seeds(seed: int, n_paths: int) -> list[np.random.SeedSequence]:
    if n_paths < 1:
        raise ValueError("n_paths must be positive")
    return list(np.random.SeedSequence(seed).spawn(n_paths))


def _circular_block_sample(
    source: np.ndarray,
    n_samples: int,
    block_len: int | None,
    rng: np.random.Generator,
    regime_labels: np.ndarray | None = None,
) -> np.ndarray:
    length = block_len or int(math.ceil(len(source) ** (1.0 / 3.0)))
    if regime_labels is None:
        return circular_block_bootstrap(source, n_samples, length, rng)
    if len(regime_labels) != len(source):
        raise ValueError("regime_labels must have the same length as values")
    starts_by_regime = {
        int(label): np.flatnonzero(regime_labels == label) for label in np.unique(regime_labels)
    }
    starts = [values for values in starts_by_regime.values() if len(values)]
    if not starts:
        return circular_block_bootstrap(source, n_samples, length, rng)
    output: list[Any] = []
    while len(output) < n_samples:
        candidates = starts[int(rng.integers(0, len(starts)))]
        start = int(candidates[int(rng.integers(0, len(candidates)))])
        take = min(length, n_samples - len(output))
        output.extend(source[(start + np.arange(take)) % len(source)].tolist())
    return np.asarray(output[:n_samples])


def _bootstrap_worker(
    payload: tuple[np.ndarray, int | None, np.random.SeedSequence, np.ndarray | None],
) -> dict[str, float]:
    source, block_len, seed, regime_labels = payload
    rng = np.random.default_rng(seed)
    return _path_summary(_circular_block_sample(source, len(source), block_len, rng, regime_labels))


def _run_workers(
    worker: Any,
    payloads: list[Any],
    workers: int,
    chunk_size: int,
) -> list[Any]:
    if workers <= 1:
        return [worker(payload) for payload in payloads]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(worker, payloads, chunksize=chunk_size))


def _read_pnl_file(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    if "pnl" not in frame:
        raise ValueError(f"{path} does not contain a pnl column")
    return pd.Series(frame["pnl"].astype(float).to_numpy(), name="pnl")


def load_daily_pnl(run_dir: str | Path, walkforward: bool = False) -> pd.Series:
    """Load daily PnL from a simulation run or persisted walk-forward output."""
    root = Path(run_dir)
    path = root / ("stitched_oos_daily_pnl.csv" if walkforward else "daily.csv")
    if not path.exists():
        if walkforward:
            raise FileNotFoundError(
                f"{path} is missing; rerun walk-forward with the current writer first"
            )
        raise FileNotFoundError(path)
    return _read_pnl_file(path)


def _read_daily_frame(run_dir: str | Path, walkforward: bool = False) -> pd.DataFrame:
    root = Path(run_dir)
    path = root / ("stitched_oos_daily_pnl.csv" if walkforward else "daily.csv")
    frame = pd.read_csv(path)
    if "pnl" not in frame:
        raise ValueError(f"{path} does not contain a pnl column")
    return frame


def _regime_labels_from_bars(bars: pd.DataFrame) -> pd.Series:
    daily = (
        bars.sort_values("ts_event")
        .groupby("session_date", sort=True)["a_close"]
        .last()
        .pct_change()
    )
    volatility = daily.rolling(21, min_periods=5).std()
    labels = pd.Series(1, index=volatility.index, dtype=int)
    valid = volatility.dropna()
    if len(valid) >= 3:
        labels.loc[valid.index] = pd.qcut(valid.rank(method="first"), 3, labels=False).astype(int)
    return labels


def _load_yahoo_pair_bars(config_path: str | Path) -> tuple[SimulationConfig, pd.DataFrame]:
    sim_config = load_simulation_config(config_path)
    yahoo_config = load_yahoo_config(config_path)
    start = pd.Timestamp(sim_config.start).date()
    end = pd.Timestamp(sim_config.end).date()
    frames: dict[str, pd.DataFrame] = {}
    for product in sim_config.products:
        frames[product], _ = fetch_yahoo_bars(
            product,
            start,
            end,
            yahoo_config.interval,
            cache_dir=yahoo_config.cache_dir,
            use_cache=True,
        )
    carry = None
    if yahoo_config.carry_adjust:
        try:
            rates, _ = fetch_yahoo_rate(
                yahoo_config.rate_symbol,
                start,
                end,
                cache_dir=yahoo_config.cache_dir,
                use_cache=True,
            )
            dividends, _ = fetch_yahoo_dividends(
                sim_config.products[1],
                cache_dir=yahoo_config.cache_dir,
                use_cache=True,
            )
            spot = (
                frames[sim_config.products[1]]
                .set_index("session_date")["close"]
                .groupby(level=0)
                .last()
            )
            dividend_yield = trailing_dividend_yield(dividends, spot)
            tau = pd.Series(
                [time_to_expiry_years(pd.Timestamp(item).date()) for item in spot.index],
                index=spot.index,
            )
            carry = (
                rates.groupby(level=0).last().shift(1).reindex(spot.index).ffill().fillna(0.0)
                - dividend_yield
            ) * tau
        except (YahooError, OSError, ValueError):
            if (
                yahoo_config.fallback_risk_free_rate is None
                or yahoo_config.fallback_dividend_yield is None
            ):
                raise
            sessions = frames[sim_config.products[1]]["session_date"].drop_duplicates()
            tau = pd.Series(
                [time_to_expiry_years(pd.Timestamp(item).date()) for item in sessions],
                index=sessions,
            )
            carry = (
                yahoo_config.fallback_risk_free_rate - yahoo_config.fallback_dividend_yield
            ) * tau
    return sim_config, build_pair_bars(
        frames[sim_config.products[0]],
        frames[sim_config.products[1]],
        bar_minutes=yahoo_config.bar_minutes,
        rth_only=yahoo_config.rth_only,
        carry=carry,
    )


def bootstrap(
    pnl: Sequence[float] | np.ndarray,
    n_paths: int = 10_000,
    block_len: int | None = None,
    seed: int = 0,
    workers: int = 1,
    chunk_size: int = 1,
    regime_labels: Sequence[int] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Run a deterministic circular block bootstrap and a demeaned null test."""
    source = np.asarray(pnl, dtype=float)
    seeds = _spawn_seeds(seed, n_paths)
    labels = None if regime_labels is None else np.asarray(regime_labels)
    payloads = [(source, block_len, child, labels) for child in seeds]
    paths = _run_workers(_bootstrap_worker, payloads, workers, chunk_size)
    observed = _path_summary(source)
    null_payloads = [
        (source - source.mean(), block_len, child, labels)
        for child in _spawn_seeds(seed + 1, n_paths)
    ]
    null_paths = _run_workers(_bootstrap_worker, null_payloads, workers, chunk_size)
    sharpes = np.asarray([item["sharpe"] for item in paths], dtype=float)
    null_sharpes = np.asarray([item["sharpe"] for item in null_paths], dtype=float)
    return {
        "mode": "bootstrap",
        "disclaimer": DISCLAIMER,
        "n_observations": int(len(source)),
        "n_paths": int(n_paths),
        "block_len": int(block_len or math.ceil(len(source) ** (1.0 / 3.0))),
        "observed": observed,
        "distribution": {
            "sharpe_percentiles": {
                "5": float(np.percentile(sharpes, 5)),
                "50": float(np.percentile(sharpes, 50)),
                "95": float(np.percentile(sharpes, 95)),
            },
            "sharpe_samples": [float(value) for value in sharpes],
            "pnl_percentiles": {
                str(q): float(np.percentile([item["pnl"] for item in paths], q))
                for q in (5, 50, 95)
            },
            "max_dd_pct_percentiles": {
                str(q): float(np.percentile([item["max_dd_pct"] for item in paths], q))
                for q in (5, 50, 95)
            },
            "probability_of_loss": float(np.mean([item["pnl"] < 0 for item in paths])),
        },
        "null": {
            "null_sharpe_mean": float(null_sharpes.mean()),
            "null_sharpe_percentiles": {
                str(q): float(np.percentile(null_sharpes, q)) for q in (5, 50, 95)
            },
            "p_value_observed_sharpe": float(np.mean(null_sharpes >= observed["sharpe"])),
        },
        "paths": paths,
        "null_sharpe_samples": [float(value) for value in null_sharpes],
    }


def _synthetic_path(
    bars: pd.DataFrame,
    seed: np.random.SeedSequence,
    kappa: float,
    sigma: float,
    n_rows: int | None = None,
) -> pd.DataFrame:
    block_seed, innovation_seed = seed.spawn(2)
    rng = np.random.default_rng(block_seed)
    source = bars.sort_values("ts_event").reset_index(drop=True)
    output_rows = len(source) if n_rows is None else n_rows
    if output_rows < 1:
        raise ValueError("n_rows must be positive")
    returns = pd.Series(np.log(source["b_close"].astype(float))).diff().fillna(0.0).to_numpy()
    volumes = source["b_volume"].fillna(0.0).astype(float).to_numpy()
    starts: list[int] = []
    remaining = output_rows
    block_len = 3
    while remaining:
        start = int(rng.integers(0, len(source)))
        take = min(block_len, remaining)
        starts.extend(int(value) for value in (start + np.arange(take)) % len(source))
        remaining -= take
    starts_array = np.asarray(starts, dtype=int)
    boot_returns = returns[starts_array]
    boot_volumes = volumes[starts_array]
    b_log = np.log(float(cast(Any, source.loc[0, "b_close"]))) + np.cumsum(boot_returns)
    spread = np.zeros(output_rows, dtype=float)
    innovations = np.random.default_rng(innovation_seed).normal(size=output_rows)
    for index in range(1, output_rows):
        spread[index] = (
            spread[index - 1] + kappa * (-spread[index - 1]) + sigma * innovations[index]
        )
    template = source.iloc[np.arange(output_rows) % len(source)].reset_index(drop=True)
    carry = template.get("carry", pd.Series(0.0, index=template.index)).astype(float).to_numpy()
    a_log = b_log + carry + spread
    b_close = np.exp(b_log)
    a_close = np.exp(a_log)
    result = template.copy()
    result["b_close"] = b_close
    result["a_close"] = a_close
    result["b_open"] = np.r_[b_close[0], b_close[:-1]]
    result["a_open"] = np.r_[a_close[0], a_close[:-1]]
    result["b_high"] = np.maximum(result["b_open"], b_close)
    result["b_low"] = np.minimum(result["b_open"], b_close)
    result["a_high"] = np.maximum(result["a_open"], a_close)
    result["a_low"] = np.minimum(result["a_open"], a_close)
    result["b_volume"] = boot_volumes
    result["a_volume"] = boot_volumes
    result["a_roll"] = False
    result["b_roll"] = False
    return result


def _synthetic_worker(
    payload: tuple[pd.DataFrame, SimulationConfig, np.random.SeedSequence, float, float],
) -> dict[str, float]:
    bars, config, seed, kappa, sigma = payload
    started = time.perf_counter()
    path = _synthetic_path(bars, seed, kappa, sigma)
    result = run_simulation(path, config)
    output = _path_summary(result.daily_pnl.to_numpy())
    output["trades"] = float(result.metrics.get("n_round_trips", 0))
    output["costs"] = float(
        result.metrics.get("total_fees_usd", 0.0) + result.metrics.get("total_slippage_usd", 0.0)
    )
    output["seconds"] = time.perf_counter() - started
    return output


def synthetic(
    bars: pd.DataFrame,
    config: SimulationConfig,
    kappa: float,
    sigma: float,
    n_paths: int = 20,
    seed: int = 0,
    workers: int = 1,
    chunk_size: int = 1,
    threshold: float = 0.0,
) -> dict[str, Any]:
    """Exercise the existing engine on causal OU/random-walk synthetic paths."""
    if kappa < 0 or sigma < 0:
        raise ValueError("kappa and sigma must be non-negative")
    payloads = [(bars, config, child, kappa, sigma) for child in _spawn_seeds(seed, n_paths)]
    paths = _run_workers(_synthetic_worker, payloads, workers, chunk_size)
    seconds = np.asarray([item["seconds"] for item in paths], dtype=float)
    sharpes = np.asarray([item["sharpe"] for item in paths], dtype=float)
    return {
        "mode": "synthetic",
        "disclaimer": DISCLAIMER,
        "kappa": float(kappa),
        "sigma": float(sigma),
        "n_paths": int(n_paths),
        "threshold": float(threshold),
        "classification": "power" if kappa > 0 else "false_positive_rate",
        "fraction_sharpe_above_threshold": float(np.mean(sharpes > threshold)),
        "mean_trades": float(np.mean([item["trades"] for item in paths])),
        "mean_costs": float(np.mean([item["costs"] for item in paths])),
        "per_path_seconds": {
            "mean": float(seconds.mean()),
            "min": float(seconds.min()),
            "max": float(seconds.max()),
        },
        "projected_seconds_for_1000_paths": float(seconds.mean() * 1000.0),
        "paths": paths,
    }


def probabilistic_sharpe_ratio(
    sharpe: float,
    benchmark: float,
    skewness: float,
    kurtosis_value: float,
    observations: int,
) -> float:
    if observations <= 1:
        return 0.5
    denominator = math.sqrt(
        max(1e-15, 1.0 - skewness * sharpe + (kurtosis_value - 1.0) * sharpe**2 / 4.0)
    )
    return float(norm.cdf((sharpe - benchmark) * math.sqrt(observations - 1.0) / denominator))


def deflate(
    trials: pd.DataFrame,
    stitched_pnl: Sequence[float] | np.ndarray,
    observations: int | None = None,
) -> dict[str, Any]:
    """Calculate PSR/DSR and minimum track record length."""
    column = "median_oos_sharpe" if "median_oos_sharpe" in trials else "in_sample_sharpe"
    trial_sharpes = trials[column].astype(float).to_numpy()
    best_annualized = float(np.max(trial_sharpes))
    n_trials = len(trial_sharpes)
    variance_annualized = float(np.var(trial_sharpes, ddof=1)) if n_trials > 1 else 0.0
    pnl = np.asarray(stitched_pnl, dtype=float)
    t = int(observations or len(pnl))
    stitched_annualized = _annualized_sharpe(pnl)
    best_daily = best_annualized / math.sqrt(252.0)
    variance_daily = variance_annualized / 252.0
    stitched_daily = stitched_annualized / math.sqrt(252.0)
    skewness = float(skew(pnl, bias=False)) if len(pnl) > 2 else 0.0
    kurtosis_value = float(kurtosis(pnl, fisher=False, bias=False)) if len(pnl) > 3 else 3.0
    if n_trials <= 1:
        benchmark = 0.0
    else:
        q1 = norm.ppf(1.0 - 1.0 / n_trials)
        q2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
        benchmark = math.sqrt(max(variance_daily, 0.0)) * (
            (1.0 - 0.5772156649) * q1 + 0.5772156649 * q2
        )
    stitched_psr = probabilistic_sharpe_ratio(stitched_daily, 0.0, skewness, kurtosis_value, t)
    best_dsr = probabilistic_sharpe_ratio(best_daily, benchmark, skewness, kurtosis_value, t)
    z = norm.ppf(0.95)
    correction = math.sqrt(
        max(
            1e-15,
            1.0 - skewness * stitched_daily + (kurtosis_value - 1.0) * stitched_daily**2 / 4.0,
        )
    )
    min_track = float(1.0 + (z * correction / stitched_daily) ** 2) if stitched_daily > 0 else None
    return {
        "mode": "deflate",
        "disclaimer": DISCLAIMER,
        "n_trials": int(n_trials),
        "trial_sharpe_column": column,
        "best_trial_sharpe_annualized": best_annualized,
        "best_trial_sharpe_daily": best_daily,
        "stitched_sharpe_annualized": stitched_annualized,
        "stitched_sharpe_daily": stitched_daily,
        "trial_sharpe_variance_annualized": variance_annualized,
        "trial_sharpe_variance_daily": variance_daily,
        "stitched_sessions": t,
        "stitched_pnl_skew": skewness,
        "stitched_pnl_kurtosis": kurtosis_value,
        "expected_max_sharpe_daily": float(benchmark),
        "expected_max_sharpe_annualized": float(benchmark * math.sqrt(252.0)),
        "stitched_psr_against_zero": stitched_psr,
        "best_trial_dsr": best_dsr,
        "minimum_track_record_sessions_95pct": min_track,
    }


def _artifact_summary(result: dict[str, Any]) -> dict[str, Any]:
    summary = dict(result)
    summary.pop("paths", None)
    summary.pop("null_sharpe_samples", None)
    if "distribution" in summary:
        distribution = dict(summary["distribution"])
        distribution.pop("sharpe_samples", None)
        summary["distribution"] = distribution
    return summary


def write_artifacts(result: dict[str, Any], out: str | Path) -> dict[str, Any]:
    """Write deterministic JSON, Markdown, and a compact Sharpe histogram."""
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    if "paths" in result:
        rows = []
        null_sharpes = result.get("null_sharpe_samples", [])
        for index, path in enumerate(result["paths"]):
            row = {
                "path": index,
                "sharpe": float(path["sharpe"]),
                "pnl": float(path["pnl"]),
                "max_dd_pct": float(path["max_dd_pct"]),
            }
            if null_sharpes:
                row["null_sharpe"] = float(null_sharpes[index])
            rows.append(row)
        pd.DataFrame(rows).to_csv(root / "paths.csv", index=False)
    summary = _artifact_summary(result)
    (root / "montecarlo.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    lines = [f"# Monte Carlo harness — {DISCLAIMER}", "", "```json"]
    lines.append(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    lines.extend(["```", ""])
    (root / "montecarlo.md").write_text("\n".join(lines))
    if "sharpe_samples" in result.get("distribution", {}):
        values = np.asarray(result["distribution"]["sharpe_samples"], dtype=float)
    elif "paths" in result:
        values = np.asarray([item["sharpe"] for item in result["paths"]], dtype=float)
    else:
        values = np.asarray([result.get("best_trial_sharpe_annualized", 0.0)], dtype=float)
    plt.figure(figsize=(5, 3))
    if len(values):
        plt.hist(values, bins=min(20, max(5, len(values))))
    plt.xlabel("Sharpe")
    plt.title(DISCLAIMER)
    plt.tight_layout()
    plt.savefig(root / "sharpe_distribution.png", dpi=90)
    plt.close()
    return summary
