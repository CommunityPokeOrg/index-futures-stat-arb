import json
from pathlib import Path

from index_futures_stat_arb.cli import main
from index_futures_stat_arb.config import load_simulation_config


def _config(path: Path, root: Path) -> None:
    path.write_text(
        f"""
[ingest]
symbols = ["ESH6", "ESM6", "NQH6", "NQM6"]
start = "2026-01-05"
end = "2026-01-16"
data_root = "{root / "data"}"

[simulation]
start = "2026-01-05"
end = "2026-01-16"
min_hedge_sessions = 2
hedge_lookback_sessions = 3
bar_minutes = 5

[sizer]
name = "fixed"
n_es = 1
n_nq = 1
"""
    )


def test_simulate_cli_writes_report_and_artifacts(tmp_path: Path) -> None:
    config = tmp_path / "sim.toml"
    _config(config, tmp_path)
    out = tmp_path / "results"
    assert main(["simulate", "--offline", "--config", str(config), "--out", str(out)]) == 0
    report = next(out.glob("*/report.json"))
    payload = json.loads(report.read_text())
    assert {"metrics", "config", "git_sha", "data_manifest_id", "roll_calendars"} <= payload.keys()
    run_dir = report.parent
    assert (run_dir / "trades.csv").exists()
    assert (run_dir / "equity.csv").exists()
    assert (run_dir / "daily.csv").exists()


def test_simulate_cli_metrics_are_deterministic(tmp_path: Path) -> None:
    config = tmp_path / "sim.toml"
    _config(config, tmp_path)
    out = tmp_path / "results"
    main(["simulate", "--offline", "--config", str(config), "--out", str(out)])
    first = json.loads(next(out.glob("*/report.json")).read_text())["metrics"]
    main(["simulate", "--offline", "--config", str(config), "--out", str(out)])
    reports = sorted(out.glob("*/report.json"))
    assert json.loads(reports[-1].read_text())["metrics"] == first


def test_compare_cli_preserves_run_order(tmp_path: Path) -> None:
    config = tmp_path / "sim.toml"
    _config(config, tmp_path)
    out = tmp_path / "results"
    main(["simulate", "--offline", "--config", str(config), "--out", str(out)])
    main(["simulate", "--offline", "--config", str(config), "--out", str(out)])
    runs = sorted(path.parent for path in out.glob("*/report.json"))
    comparison = tmp_path / "comparison.md"
    assert main(["compare", "--runs", str(runs[1]), str(runs[0]), "--out", str(comparison)]) == 0
    text = comparison.read_text()
    assert text.startswith("| label | data_source |")
    assert text.index(runs[1].name) < text.index(runs[0].name)


def test_simulation_configs_parse_refined_fields() -> None:
    configs = Path("configs")
    for path in configs.glob("sim_*.toml"):
        config = load_simulation_config(path)
        assert config.initial_capital_usd == 1_000_000.0
        assert config.hedge_method in {"ols", "kalman", "rolling_eg"}
        assert config.threshold_mode in {"fixed", "ou"}


def test_ingest_and_rolls_smoke(tmp_path: Path) -> None:
    config = tmp_path / "ingest.toml"
    config.write_text(
        f"""
[ingest]
symbols = ["ESH6", "ESM6", "NQH6", "NQM6"]
start = "2026-01-05"
end = "2026-01-09"
data_root = "{tmp_path / "data"}"
"""
    )
    assert main(["ingest", "--offline", "--no-resume", "--config", str(config)]) == 0
    out = tmp_path / "rolls"
    assert main(["rolls", "--config", str(config), "--out", str(out)]) == 0
    assert (out / "joint.parquet").exists()
