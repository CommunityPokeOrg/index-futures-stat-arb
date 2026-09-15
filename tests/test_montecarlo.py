from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from index_futures_stat_arb.execution.engine import SimulationConfig
from index_futures_stat_arb.montecarlo import (
    _synthetic_path,
    bootstrap,
    circular_block_bootstrap,
    deflate,
    synthetic,
    write_artifacts,
)


def _bars(rows: int = 40) -> pd.DataFrame:
    sessions = [date(2020, 1, 1) + timedelta(days=i) for i in range(rows)]
    b = 100.0 * np.exp(np.cumsum(np.sin(np.arange(rows) / 3.0) * 0.002))
    a = b * np.exp(0.01 * np.sin(np.arange(rows) / 4.0))
    return pd.DataFrame(
        {
            "ts_event": pd.date_range("2020-01-01", periods=rows, tz="UTC"),
            "session_date": sessions,
            "a_open": a,
            "a_high": a * 1.001,
            "a_low": a * 0.999,
            "a_close": a,
            "a_volume": 1000,
            "b_open": b,
            "b_high": b * 1.001,
            "b_low": b * 0.999,
            "b_close": b,
            "b_volume": 1000,
            "a_contract": "A",
            "b_contract": "B",
            "a_roll": False,
            "b_roll": False,
            "a_close_raw": a,
            "b_close_raw": b,
            "carry": 0.0,
        }
    )


def test_bootstrap_is_worker_independent(tmp_path: Path) -> None:
    values = np.arange(30, dtype=float)
    one = bootstrap(values, n_paths=20, block_len=4, seed=11, workers=1)
    three = bootstrap(values, n_paths=20, block_len=4, seed=11, workers=3)
    write_artifacts(one, tmp_path / "one")
    write_artifacts(three, tmp_path / "three")
    assert (tmp_path / "one" / "montecarlo.json").read_bytes() == (
        tmp_path / "three" / "montecarlo.json"
    ).read_bytes()


def test_circular_bootstrap_preserves_contiguous_blocks() -> None:
    source = np.arange(20)
    sample = circular_block_bootstrap(source, 100, 5, np.random.default_rng(3))
    assert all(
        sample[i + 1] == (sample[i] + 1) % len(source) for i in range(len(sample) - 1) if i % 5 != 4
    )


def test_demeaned_null_is_centered() -> None:
    rng = np.random.default_rng(9)
    result = bootstrap(rng.normal(size=120), n_paths=200, seed=4)
    assert 0.01 < result["null"]["p_value_observed_sharpe"] < 0.99
    assert abs(result["null"]["null_sharpe_mean"]) < 0.4


def test_deflate_single_trial_equals_psr_against_zero() -> None:
    trials = pd.DataFrame({"median_oos_sharpe": [1.0]})
    pnl = np.arange(1.0, 21.0)
    result = deflate(trials, pnl)
    assert result["dsr"] == result["psr_against_zero"]


def test_synthetic_path_prefix_is_causal() -> None:
    bars = _bars(40)
    short = _synthetic_path(bars, np.random.SeedSequence(4), 0.2, 0.01, n_rows=20)
    long = _synthetic_path(bars, np.random.SeedSequence(4), 0.2, 0.01, n_rows=40)
    np.testing.assert_array_equal(short["a_close"], long["a_close"].iloc[:20])
    np.testing.assert_array_equal(short["b_close"], long["b_close"].iloc[:20])


def test_tiny_synthetic_run_writes_artifacts(tmp_path: Path) -> None:
    bars = _bars()
    config = SimulationConfig(
        start="2020-01-01",
        end="2020-02-15",
        products=("ES", "SPY"),
        bar_minutes=1,
        rth_only=False,
        z_window=10,
        z_reset_each_session=False,
        hedge_lookback_sessions=10,
        min_hedge_sessions=5,
        adjust="none",
    )
    result = synthetic(bars, config, kappa=0.2, sigma=0.01, n_paths=3, seed=4)
    write_artifacts(result, tmp_path)
    assert (tmp_path / "montecarlo.json").exists()
    assert (tmp_path / "montecarlo.md").exists()
    assert (tmp_path / "sharpe_distribution.png").exists()
    assert json.loads((tmp_path / "montecarlo.json").read_text())["n_paths"] == 3
