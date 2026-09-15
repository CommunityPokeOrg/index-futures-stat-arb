"""Command-line entry point for index-futures-stat-arb."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from .config import (
    load_config,
    load_roll_config,
    load_simulation_config,
    load_walkforward_config,
    load_yahoo_config,
)
from .ingest import BarClient


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ifsa")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest = subparsers.add_parser("ingest")
    ingest.add_argument("--config", required=True, type=Path)
    ingest.add_argument("--offline", action="store_true")
    ingest.add_argument("--no-resume", action="store_true")
    rolls = subparsers.add_parser("rolls")
    rolls.add_argument("--config", required=True, type=Path)
    rolls.add_argument("--rule", choices=["calendar", "volume", "open_interest", "fixed_k"])
    rolls.add_argument("--out", required=True, type=Path)
    continuous = subparsers.add_parser("continuous")
    continuous.add_argument("--config", type=Path)
    continuous.add_argument("--product", required=True, choices=["ES", "NQ"])
    continuous.add_argument("--adjust", required=True, choices=["none", "panama", "ratio"])
    continuous.add_argument("--calendar", type=Path)
    continuous.add_argument("--out", required=True, type=Path)
    simulate = subparsers.add_parser("simulate")
    simulate.add_argument("--config", required=True, type=Path)
    simulate.add_argument("--offline", action="store_true")
    simulate.add_argument("--out", required=True, type=Path)
    simulate_yahoo = subparsers.add_parser("simulate-yahoo")
    simulate_yahoo.add_argument("--config", required=True, type=Path)
    simulate_yahoo.add_argument("--out", required=True, type=Path)
    simulate_yahoo.add_argument("--no-cache", action="store_true")
    simulate_yahoo.add_argument("--offline-fixture", type=Path)
    diagnose_yahoo = subparsers.add_parser("diagnose-yahoo")
    diagnose_yahoo.add_argument("--config", required=True, type=Path)
    diagnose_yahoo.add_argument("--out", required=True, type=Path)
    diagnose_yahoo.add_argument("--no-cache", action="store_true")
    diagnose_yahoo.add_argument("--offline-fixture", type=Path)
    walkforward = subparsers.add_parser("walkforward")
    walkforward.add_argument("--config", required=True, type=Path)
    walkforward.add_argument("--out", required=True, type=Path)
    walkforward.add_argument("--no-cache", action="store_true")
    walkforward.add_argument("--offline-fixture", type=Path)
    walkforward.add_argument("--n-trials", type=int)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--runs", nargs="+", type=Path, required=True)
    compare.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.command == "ingest":
        from .ingest.databento import DatabentoClient, run_ingest

        config = load_config(args.config)
        if args.offline:
            from .fixtures import SyntheticBarClient

            config = config.__class__(**{**config.__dict__, "source": "synthetic"})
            client: BarClient = SyntheticBarClient()
        else:
            client = DatabentoClient()
        manifest = run_ingest(config, client, resume=not args.no_resume)
        manifest_path = config.data_root / "manifests" / f"{manifest.dataset_id}.json"
        print(f"{manifest_path} row_count={manifest.row_count}")
    if args.command == "rolls":
        from datetime import date

        from .contracts import list_contracts
        from .ingest.databento import read_partitioned
        from .rolls import build_roll_calendar, daily_from_bars, joint_roll_calendar

        config = load_config(args.config)
        roll_config = load_roll_config(args.config, args.rule)
        start_date = date.fromisoformat(config.start[:10])
        end_date = date.fromisoformat(config.end[:10])
        source = config.source if config.source != "databento" else "synthetic"
        bars = read_partitioned(
            config.data_root,
            source,
            config.dataset,
            config.schema,
            start=start_date,
            end=end_date,
        )
        daily = daily_from_bars(bars)
        calendars = {}
        for product in ("ES", "NQ"):
            contracts = list_contracts(product, start_date, end_date)
            calendars[product] = build_roll_calendar(
                product, daily, contracts, roll_config, start_date, end_date
            )
        joint = joint_roll_calendar(calendars["ES"], calendars["NQ"])
        args.out.mkdir(parents=True, exist_ok=True)
        joint_frames = []
        for product in ("ES", "NQ"):
            calendars[product] = joint[product]
            output = args.out / f"{product.lower()}.parquet"
            calendars[product].to_parquet(output, index=False)
            print(product)
            print(calendars[product].to_string(index=False))
            joint_frames.append(calendars[product])
        pd.concat(joint_frames, ignore_index=True).to_parquet(
            args.out / "joint.parquet", index=False
        )
        return 0
    if args.command == "continuous":
        from datetime import date

        from .continuous import build_continuous
        from .contracts import list_contracts
        from .ingest.databento import read_partitioned

        config = load_config(args.config or "configs/ingest_example.toml")
        calendar_path = args.calendar or Path("data/reference/roll_calendar") / (
            f"{args.product.lower()}.parquet"
        )
        calendar = pd.read_parquet(calendar_path)
        start_date = date.fromisoformat(config.start[:10])
        end_date = date.fromisoformat(config.end[:10])
        contracts = list_contracts(args.product, start_date, end_date)
        source = config.source if config.source != "databento" else "synthetic"
        bars = read_partitioned(
            config.data_root,
            source,
            config.dataset,
            config.schema,
            products=[args.product],
        )
        continuous_result = build_continuous(bars, calendar, contracts, args.adjust)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        continuous_result.to_parquet(args.out, index=False)
        print(f"{args.out} row_count={len(continuous_result)}")
        return 0
    if args.command == "simulate":
        from .contracts import list_contracts
        from .execution.engine import load_pair_bars, run_simulation
        from .fixtures import write_fixture_dataset
        from .ingest.databento import read_partitioned
        from .rolls import build_roll_calendar, daily_from_bars, joint_roll_calendar

        sim_config = load_simulation_config(args.config)
        ingest_config = load_config(args.config)
        source = ingest_config.source
        manifest_path = ingest_config.data_root / "manifests"
        if args.offline:
            source = "synthetic"
            expected = list(manifest_path.glob("synthetic-*.json"))
            if not expected:
                write_fixture_dataset(
                    ingest_config.data_root, sim_config.start, sim_config.end, sim_config.seed
                )
        pair_bars = load_pair_bars(
            sim_config,
            ingest_config.data_root,
            source,
            ingest_config.dataset,
            ingest_config.schema,
        )
        simulation_result = run_simulation(pair_bars, sim_config)
        raw = read_partitioned(
            ingest_config.data_root,
            source,
            ingest_config.dataset,
            ingest_config.schema,
            products=list(sim_config.products),
            start=pd.Timestamp(sim_config.start).date(),
            end=pd.Timestamp(sim_config.end).date(),
        )
        daily = daily_from_bars(raw)
        calendars = {}
        for product in sim_config.products:
            contracts = list_contracts(
                product,
                pd.Timestamp(sim_config.start).date(),
                pd.Timestamp(sim_config.end).date(),
            )
            calendars[product] = build_roll_calendar(
                product,
                daily,
                contracts,
                sim_config.roll,
                pd.Timestamp(sim_config.start).date(),
                pd.Timestamp(sim_config.end).date(),
            )
        joint = joint_roll_calendar(calendars["ES"], calendars["NQ"])
        config_json = json.dumps(asdict(sim_config), default=str, sort_keys=True)
        run_id = _run_id(config_json)
        run_dir = args.out / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        report_path = _write_run(
            run_dir,
            sim_config,
            simulation_result,
            {
                "data_source": source,
                "data_manifest_id": expected[0].stem if args.offline and expected else "unknown",
                "roll_calendars": {
                    product: joint[product].to_dict(orient="records")
                    for product in sim_config.products
                },
            },
        )
        print(f"report={report_path}")
        for key, value in simulation_result.metrics.items():
            print(f"{key}: {value}")
        return 0
    if args.command == "simulate-yahoo" or args.command == "diagnose-yahoo":
        from datetime import date

        from .basis import time_to_expiry_years, trailing_dividend_yield
        from .execution.engine import run_simulation
        from .ingest.yahoo import (
            YahooError,
            build_pair_bars,
            fetch_yahoo_bars,
            fetch_yahoo_dividends,
            fetch_yahoo_rate,
        )

        sim_config = load_simulation_config(args.config)
        yahoo_config = load_yahoo_config(args.config)
        start = date.fromisoformat(sim_config.start[:10])
        end = date.fromisoformat(sim_config.end[:10])
        metadata: dict[str, dict[str, object]] = {}
        frames: dict[str, pd.DataFrame] = {}
        if args.offline_fixture:
            for product in sim_config.products:
                candidate = args.offline_fixture / f"{product.lower()}.parquet"
                if not candidate.exists():
                    candidate = args.offline_fixture / f"{product}.parquet"
                frames[product] = pd.read_parquet(candidate)
                metadata[product] = {
                    "product": product,
                    "interval": yahoo_config.interval,
                    "rows": len(frames[product]),
                    "cache_hit": True,
                    "cache_path": str(candidate),
                }
        else:
            for product in sim_config.products:
                frames[product], metadata[product] = fetch_yahoo_bars(
                    product,
                    start,
                    end,
                    yahoo_config.interval,
                    cache_dir=yahoo_config.cache_dir,
                    use_cache=not args.no_cache,
                )
        carry_source = "none"
        carry_values = None
        carry_meta: dict[str, object] = {}
        if yahoo_config.carry_adjust:
            try:
                rate_values, rate_meta = fetch_yahoo_rate(
                    yahoo_config.rate_symbol,
                    start,
                    end,
                    cache_dir=yahoo_config.cache_dir,
                    use_cache=not args.no_cache,
                )
                dividends, dividends_meta = fetch_yahoo_dividends(
                    sim_config.products[1],
                    cache_dir=yahoo_config.cache_dir,
                    use_cache=not args.no_cache,
                )
                rate_daily = rate_values.groupby(level=0).last().shift(1)
                spot_daily = (
                    frames[sim_config.products[1]]
                    .set_index("session_date")["close"]
                    .groupby(level=0)
                    .last()
                )
                div_yield = trailing_dividend_yield(dividends, spot_daily)
                index = spot_daily.index
                tau = pd.Series(
                    [time_to_expiry_years(pd.Timestamp(item).date()) for item in index],
                    index=index,
                )
                carry_values = (rate_daily.reindex(index).ffill().fillna(0.0) - div_yield) * tau
                carry_source = "yahoo_irx_dividends"
                carry_meta = {"rate": rate_meta, "dividends": dividends_meta}
            except (YahooError, OSError, ValueError) as exc:
                if (
                    yahoo_config.fallback_risk_free_rate is None
                    or yahoo_config.fallback_dividend_yield is None
                ):
                    raise
                session_index = frames[sim_config.products[1]]["session_date"].drop_duplicates()
                tau_values = pd.Series(
                    [time_to_expiry_years(pd.Timestamp(item).date()) for item in session_index],
                    index=session_index,
                    dtype=float,
                )
                carry_values = (
                    yahoo_config.fallback_risk_free_rate - yahoo_config.fallback_dividend_yield
                ) * tau_values
                carry_source = "fallback_constant"
                carry_meta = {"fallback_reason": repr(exc)}
        pair_bars = build_pair_bars(
            frames[sim_config.products[0]],
            frames[sim_config.products[1]],
            bar_minutes=yahoo_config.bar_minutes,
            rth_only=yahoo_config.rth_only,
            carry=carry_values,
        )
        if args.command == "diagnose-yahoo":
            diagnostics = _yahoo_diagnostics(pair_bars, sim_config)
            diagnostics["data_meta"] = metadata | carry_meta
            diagnostics["carry_source"] = carry_source
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / "diagnostics.json").write_text(
                json.dumps(diagnostics, default=str, indent=2)
            )
            (args.out / "diagnostics.md").write_text(_diagnostics_markdown(diagnostics))
            print((args.out / "diagnostics.md").read_text(), end="")
            return 0
        simulation_result = run_simulation(pair_bars, sim_config)
        effective_start = metadata[sim_config.products[0]].get("effective_start", start.isoformat())
        effective_end = metadata[sim_config.products[0]].get("effective_end", end.isoformat())
        config_json = json.dumps(asdict(sim_config), default=str, sort_keys=True)
        run_id = _run_id(config_json)
        report_path = _write_run(
            args.out / run_id,
            sim_config,
            simulation_result,
            {
                "data_source": "yahoo",
                "data_meta": metadata | carry_meta | {"carry_source": carry_source},
                "roll_calendars": {},
                "data_manifest_id": f"yahoo:{yahoo_config.interval}:"
                f"{effective_start}:{effective_end}",
            },
        )
        print(f"report={report_path}")
        print(
            f"data_source=yahoo interval={yahoo_config.interval} "
            + " ".join(
                f"{product} rows={metadata[product].get('rows', len(frames[product]))}"
                for product in sim_config.products
            )
            + f" range={effective_start}..{effective_end}"
        )
        for key, value in simulation_result.metrics.items():
            print(f"{key}: {value}")
        return 0
    if args.command == "walkforward":
        from datetime import date

        from .basis import time_to_expiry_years, trailing_dividend_yield
        from .ingest.yahoo import (
            YahooError,
            build_pair_bars,
            fetch_yahoo_bars,
            fetch_yahoo_dividends,
            fetch_yahoo_rate,
        )
        from .walkforward import WalkForwardConfig, evaluate_trials, write_artifacts

        sim_config, wf_config = load_walkforward_config(args.config)
        if args.n_trials is not None:
            wf_config = WalkForwardConfig(
                n_trials=args.n_trials,
                n_folds=wf_config.n_folds,
                min_train_sessions=wf_config.min_train_sessions,
                min_trades_per_fold=wf_config.min_trades_per_fold,
                seed=wf_config.seed,
                space=wf_config.space,
            )
        yahoo_config = load_yahoo_config(args.config)
        start = date.fromisoformat(sim_config.start[:10])
        end = date.fromisoformat(sim_config.end[:10])
        wf_frames: dict[str, pd.DataFrame] = {}
        if args.offline_fixture:
            for product in sim_config.products:
                candidate = args.offline_fixture / f"{product.lower()}.parquet"
                if not candidate.exists():
                    candidate = args.offline_fixture / f"{product}.parquet"
                wf_frames[product] = pd.read_parquet(candidate)
        else:
            for product in sim_config.products:
                wf_frames[product], _ = fetch_yahoo_bars(
                    product,
                    start,
                    end,
                    yahoo_config.interval,
                    cache_dir=yahoo_config.cache_dir,
                    use_cache=not args.no_cache,
                )
        carry_values = None
        if yahoo_config.carry_adjust:
            try:
                rate_values, _ = fetch_yahoo_rate(
                    yahoo_config.rate_symbol,
                    start,
                    end,
                    cache_dir=yahoo_config.cache_dir,
                    use_cache=not args.no_cache,
                )
                dividends, _ = fetch_yahoo_dividends(
                    sim_config.products[1],
                    cache_dir=yahoo_config.cache_dir,
                    use_cache=not args.no_cache,
                )
                rate_daily = rate_values.groupby(level=0).last().shift(1)
                spot_daily = (
                    wf_frames[sim_config.products[1]]
                    .set_index("session_date")["close"]
                    .groupby(level=0)
                    .last()
                )
                div_yield = trailing_dividend_yield(dividends, spot_daily)
                tau = pd.Series(
                    [time_to_expiry_years(pd.Timestamp(item).date()) for item in spot_daily.index],
                    index=spot_daily.index,
                )
                carry_values = (
                    rate_daily.reindex(spot_daily.index).ffill().fillna(0.0) - div_yield
                ) * tau
            except (YahooError, OSError, ValueError):
                if (
                    yahoo_config.fallback_risk_free_rate is None
                    or yahoo_config.fallback_dividend_yield is None
                ):
                    raise
                session_index = wf_frames[sim_config.products[1]]["session_date"].drop_duplicates()
                tau = pd.Series(
                    [time_to_expiry_years(pd.Timestamp(item).date()) for item in session_index],
                    index=session_index,
                )
                carry_values = (
                    yahoo_config.fallback_risk_free_rate - yahoo_config.fallback_dividend_yield
                ) * tau
        pair_bars = build_pair_bars(
            wf_frames[sim_config.products[0]],
            wf_frames[sim_config.products[1]],
            bar_minutes=yahoo_config.bar_minutes,
            rth_only=yahoo_config.rth_only,
            carry=carry_values,
        )
        result = evaluate_trials(sim_config, pair_bars, wf_config)
        selection = write_artifacts(result, args.out)
        print(json.dumps(selection, indent=2, sort_keys=True, default=str))
        return 0
    if args.command == "compare":
        table = _comparison_table(args.runs)
        print(table)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table + "\n")
        return 0
    return 0


def _write_run(
    run_dir: Path,
    sim_config: Any,
    simulation_result: Any,
    extra: dict[str, object],
) -> Path:
    """Write the common simulation artifacts and report."""
    run_dir.mkdir(parents=True, exist_ok=True)
    result = simulation_result
    result.trades.to_csv(run_dir / "trades.csv", index=False)
    result.signals.reset_index().to_csv(run_dir / "signals.csv", index=False)
    pd.DataFrame(
        {
            "ts_event": result.pnl.index,
            "pnl": result.pnl.to_numpy(),
            "equity": result.equity.to_numpy(),
            "n_a": result.positions["n_a"].to_numpy(),
            "n_b": result.positions["n_b"].to_numpy(),
            "z": result.signals["z"].to_numpy(),
        }
    ).to_csv(run_dir / "equity.csv", index=False)
    result.daily_pnl.rename("pnl").to_csv(run_dir / "daily.csv")
    report = {
        "config": asdict(sim_config),
        "metrics": result.metrics,
        "hedge_history": result.hedge_history.to_dict(orient="records"),
        "git_sha": _git_sha(),
        "package_version": "0.1.0",
        **extra,
    }
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, default=str, indent=2))
    return report_path


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _run_id(config_json: str) -> str:
    return (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + "-"
        + hashlib.sha256(config_json.encode()).hexdigest()[:8]
    )


def _yahoo_diagnostics(bars: pd.DataFrame, config: Any) -> dict[str, object]:
    from .cointegration import adf_test
    from .execution.hedge import KalmanHedge, KalmanLevel
    from .ou import fit_ou

    a = np.log(bars["a_close"].astype(float))
    b = np.log(bars["b_close"].astype(float))
    carry = bars.get("carry", pd.Series(0.0, index=bars.index)).astype(float)
    raw = a - b
    adjusted = a - carry - b
    beta, alpha = np.polyfit(b, a - carry if config.carry_adjust else a, 1)
    residual = (a - carry if config.carry_adjust else a) - alpha - beta * b

    def series_summary(values: pd.Series) -> dict[str, object]:
        values = values.replace([np.inf, -np.inf], np.nan).dropna()
        try:
            adf = adf_test(values)
            adf_pvalue = adf["pvalue"]
        except Exception:
            adf_pvalue = float("nan")
        try:
            params = fit_ou(values.to_numpy(), dt=1.0)
            half_life = params.half_life
            stationary_sigma = params.stationary_std
        except Exception:
            half_life = float("nan")
            stationary_sigma = float("nan")
        return {
            "adf_pvalue": adf_pvalue,
            "ou_half_life": half_life,
            "ou_stationary_sigma": stationary_sigma,
        }

    full: dict[str, object] = {
        "ols_alpha": float(alpha),
        "ols_beta": float(beta),
        "engle_granger_pvalue": _coint_pvalue(
            pd.Series(a - carry if config.carry_adjust else a), pd.Series(b)
        ),
        "raw_log_basis": series_summary(pd.Series(raw)),
        "carry_adjusted_log_basis": (
            series_summary(pd.Series(adjusted)) if config.carry_adjust else None
        ),
        "ols_residual": series_summary(pd.Series(residual)),
    }
    sessions = list(bars["session_date"].drop_duplicates())
    windows: list[dict[str, object]] = []
    width = config.hedge_lookback_sessions
    step = max(1, width // 2)
    for start in range(0, max(0, len(sessions) - width + 1), step):
        selected = bars[bars["session_date"].isin(sessions[start : start + width])]
        sx = np.log(selected["b_close"].to_numpy())
        sy = np.log(selected["a_close"].to_numpy())
        if config.carry_adjust:
            sy = sy - selected["carry"].to_numpy()
        if len(sx) < 3:
            continue
        w_beta, _w_alpha = np.polyfit(sx, sy, 1)
        w_residual = sy - _w_alpha - w_beta * sx
        windows.append(
            {
                "eg_pvalue": _coint_pvalue(pd.Series(sy), pd.Series(sx)),
                "beta": float(w_beta),
                "half_life": series_summary(pd.Series(w_residual))["ou_half_life"],
            }
        )
    pvalues = [
        float(cast(Any, item["eg_pvalue"]))
        for item in windows
        if np.isfinite(float(cast(Any, item["eg_pvalue"])))
    ]
    betas = [float(cast(Any, item["beta"])) for item in windows]
    half_lives = [
        float(cast(Any, item["half_life"]))
        for item in windows
        if np.isfinite(float(cast(Any, item["half_life"])))
    ]
    kalman = KalmanHedge(delta=config.kalman_delta, obs_var=config.kalman_obs_var)
    beta_path: list[float] = []
    for x_value, y_value in zip(np.asarray(b), np.asarray(a - carry), strict=True):
        kalman.predict(float(x_value))
        kalman.update(float(y_value), float(x_value))
        beta_path.append(kalman.beta)
    unit_level = KalmanLevel(
        delta=config.kalman_delta,
        obs_var=config.kalman_obs_var,
        alpha=float(np.mean(np.asarray(a - carry) - np.asarray(b))),
    )
    unit_alpha_path: list[float] = []
    for x_value, y_value in zip(np.asarray(b), np.asarray(a - carry), strict=True):
        unit_level.predict()
        unit_level.update(float(y_value - x_value))
        unit_alpha_path.append(unit_level.alpha)
    return {
        "rows": {"a": int(len(bars)), "b": int(len(bars))},
        "effective_range": {
            "start": str(bars["session_date"].min()),
            "end": str(bars["session_date"].max()),
        },
        "interval": str(bars["ts_event"].diff().dropna().median()),
        "full_sample": full,
        "rolling": {
            "window_sessions": width,
            "step_sessions": step,
            "fraction_eg_p_lt_005": float(sum(p < 0.05 for p in pvalues) / len(pvalues))
            if pvalues
            else 0.0,
            "fraction_eg_p_lt_010": float(sum(p < 0.10 for p in pvalues) / len(pvalues))
            if pvalues
            else 0.0,
            "beta_min": min(betas) if betas else float("nan"),
            "beta_median": float(np.median(betas)) if betas else float("nan"),
            "beta_max": max(betas) if betas else float("nan"),
            "half_life_median": float(np.median(half_lives)) if half_lives else float("nan"),
        },
        "kalman_beta": {
            "min": min(beta_path) if beta_path else float("nan"),
            "median": float(np.median(beta_path)) if beta_path else float("nan"),
            "max": max(beta_path) if beta_path else float("nan"),
            "delta": config.kalman_delta,
        },
        "unit_alpha": {
            "min": min(unit_alpha_path) if unit_alpha_path else float("nan"),
            "median": float(np.median(unit_alpha_path)) if unit_alpha_path else float("nan"),
            "max": max(unit_alpha_path) if unit_alpha_path else float("nan"),
            "delta": config.kalman_delta,
        },
    }


def _coint_pvalue(y: pd.Series, x: pd.Series) -> float:
    from .cointegration import engle_granger

    try:
        return float(engle_granger(y, x).pvalue)
    except Exception:
        return float("nan")


def _diagnostics_markdown(diagnostics: dict[str, object]) -> str:
    lines = ["# Yahoo pair diagnostics", ""]
    lines.append("```json")
    lines.append(json.dumps(diagnostics, default=str, indent=2))
    lines.extend(["```", ""])
    return "\n".join(lines)


def _comparison_table(run_dirs: list[Path]) -> str:
    headers = [
        "label",
        "data_source",
        "data range",
        "bars",
        "hedge_method",
        "threshold_mode",
        "initial capital",
        "net PnL",
        "return %",
        "annualized volatility",
        "Sharpe",
        "max drawdown %",
        "round trips",
        "win rate",
        "average holding",
        "fees",
        "slippage",
        "entry_gated_fraction",
    ]
    rows: list[list[str]] = []
    for run_dir in run_dirs:
        report = json.loads((run_dir / "report.json").read_text())
        config = report.get("config", {})
        metrics = report.get("metrics", {})
        metadata = report.get("data_meta", {})
        leg_meta = (
            [
                value
                for key, value in metadata.items()
                if key not in {"carry_source", "rate", "dividends"} and isinstance(value, dict)
            ]
            if isinstance(metadata, dict)
            else []
        )
        first_meta = leg_meta[0] if leg_meta else {}
        start = first_meta.get("effective_start", config.get("start", ""))
        end = first_meta.get("effective_end", config.get("end", ""))
        bars = first_meta.get("rows", "")
        rows.append(
            [
                run_dir.name,
                str(report.get("data_source", "")),
                f"{start}..{end}",
                str(bars),
                str(config.get("hedge_method", "ols")),
                str(config.get("threshold_mode", "fixed")),
                _fmt(config.get("initial_capital_usd", ""), money=True),
                _fmt(metrics.get("total_pnl_usd", ""), money=True),
                _fmt(metrics.get("total_return_pct", ""), percent=True, percent_points=True),
                _fmt(metrics.get("ann_vol_pct", ""), percent=True, percent_points=True),
                _fmt(metrics.get("sharpe", ""), fixed_precision=2),
                _fmt(metrics.get("max_drawdown_pct", ""), percent=True, percent_points=True),
                _fmt(metrics.get("n_round_trips", "")),
                _fmt(metrics.get("win_rate", ""), percent=True),
                _fmt(metrics.get("avg_holding_bars", "")),
                _fmt(metrics.get("total_fees_usd", ""), money=True),
                _fmt(metrics.get("total_slippage_usd", ""), money=True),
                _fmt(metrics.get("entry_gated_fraction", ""), percent=True),
            ]
        )
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _fmt(
    value: object,
    percent: bool = False,
    money: bool = False,
    precision: int = 6,
    percent_points: bool = False,
    fixed_precision: int | None = None,
) -> str:
    if not isinstance(value, (float, int)):
        return str(value)
    if percent:
        if percent_points:
            return f"{float(value):.2f}%"
        return f"{float(value):.2%}"
    if money:
        return f"${float(value):,.0f}"
    if fixed_precision is not None:
        return f"{float(value):.{fixed_precision}f}"
    return f"{float(value):.{precision}g}"
