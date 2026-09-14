"""Command-line entry point for index-futures-stat-arb."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_config, load_roll_config, load_simulation_config, load_yahoo_config
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
    if args.command == "simulate-yahoo":
        from datetime import date

        from .execution.engine import run_simulation
        from .ingest.yahoo import build_pair_bars, fetch_yahoo_bars

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
        pair_bars = build_pair_bars(
            frames["ES"],
            frames["NQ"],
            bar_minutes=yahoo_config.bar_minutes,
            rth_only=yahoo_config.rth_only,
        )
        simulation_result = run_simulation(pair_bars, sim_config)
        effective_start = metadata["ES"].get("effective_start", start.isoformat())
        effective_end = metadata["ES"].get("effective_end", end.isoformat())
        config_json = json.dumps(asdict(sim_config), default=str, sort_keys=True)
        run_id = _run_id(config_json)
        report_path = _write_run(
            args.out / run_id,
            sim_config,
            simulation_result,
            {
                "data_source": "yahoo",
                "data_meta": metadata,
                "roll_calendars": {},
                "data_manifest_id": f"yahoo:{yahoo_config.interval}:"
                f"{effective_start}:{effective_end}",
            },
        )
        print(f"report={report_path}")
        print(
            f"data_source=yahoo interval={yahoo_config.interval} "
            f"ES rows={metadata['ES'].get('rows', len(frames['ES']))} "
            f"NQ rows={metadata['NQ'].get('rows', len(frames['NQ']))} "
            f"range={effective_start}..{effective_end}"
        )
        for key, value in simulation_result.metrics.items():
            print(f"{key}: {value}")
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
            "n_es": result.positions["n_es"].to_numpy(),
            "n_nq": result.positions["n_nq"].to_numpy(),
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
        es_meta = metadata.get("ES", {}) if isinstance(metadata, dict) else {}
        nq_meta = metadata.get("NQ", {}) if isinstance(metadata, dict) else {}
        start = es_meta.get("effective_start", config.get("start", ""))
        end = es_meta.get("effective_end", config.get("end", ""))
        bars = es_meta.get("rows", nq_meta.get("rows", ""))
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
