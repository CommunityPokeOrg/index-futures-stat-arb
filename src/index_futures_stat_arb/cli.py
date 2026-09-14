"""Command-line entry point for index-futures-stat-arb."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .config import load_config, load_roll_config
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
        from datetime import datetime, timezone

        from .config import load_simulation_config
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
        run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + hashlib.sha256(config_json.encode()).hexdigest()[:8]
        )
        run_dir = args.out / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        simulation_result.trades.to_csv(run_dir / "trades.csv", index=False)
        pd.DataFrame(
            {
                "ts_event": simulation_result.pnl.index,
                "pnl": simulation_result.pnl.to_numpy(),
                "equity": simulation_result.equity.to_numpy(),
                "n_es": simulation_result.positions["n_es"].to_numpy(),
                "n_nq": simulation_result.positions["n_nq"].to_numpy(),
                "z": simulation_result.signals["z"].to_numpy(),
            }
        ).to_csv(run_dir / "equity.csv", index=False)
        simulation_result.daily_pnl.rename("pnl").to_csv(run_dir / "daily.csv")
        report = {
            "config": asdict(sim_config),
            "metrics": simulation_result.metrics,
            "hedge_history": simulation_result.hedge_history.to_dict(orient="records"),
            "git_sha": _git_sha(),
            "package_version": "0.1.0",
            "data_manifest_id": expected[0].stem if args.offline and expected else "unknown",
            "roll_calendars": {
                product: joint[product].to_dict(orient="records") for product in sim_config.products
            },
        }
        (run_dir / "report.json").write_text(json.dumps(report, default=str, indent=2))
        print(f"report={run_dir / 'report.json'}")
        for key, value in simulation_result.metrics.items():
            print(f"{key}: {value}")
        return 0
    return 0


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"
