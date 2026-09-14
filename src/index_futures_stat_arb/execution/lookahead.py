"""Helpers for detecting future-data access in simulations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .engine import LookaheadError, SimulationConfig, run_simulation

__all__ = ["LookaheadError", "assert_no_lookahead"]


def assert_no_lookahead(
    bars: pd.DataFrame,
    cfg: SimulationConfig,
    cut_index: int,
    rng: np.random.Generator | None = None,
) -> None:
    rng = rng or np.random.default_rng(cfg.seed)
    full = run_simulation(bars, cfg)
    prefix = run_simulation(bars.iloc[:cut_index].copy(), cfg)
    altered = bars.copy()
    for column in (
        "es_open",
        "es_high",
        "es_low",
        "es_close",
        "nq_open",
        "nq_high",
        "nq_low",
        "nq_close",
    ):
        altered.loc[cut_index:, column] = altered.loc[cut_index:, column] * rng.uniform(0.9, 1.1)
    changed = run_simulation(altered, cfg)
    comparable = max(cut_index - 1, 0)
    expected = full.positions.iloc[:comparable]
    pd.testing.assert_frame_equal(expected, prefix.positions.iloc[:comparable])
    pd.testing.assert_frame_equal(expected, changed.positions.iloc[:comparable])
    cutoff = bars.iloc[cut_index - 1]["ts_event"]

    def prior_trades(result: Any) -> pd.DataFrame:
        frame = result.trades
        if frame.empty:
            return frame
        return frame[(frame["ts_event"] <= cutoff) & (frame["reason"] != "eod_final")].reset_index(
            drop=True
        )

    pd.testing.assert_frame_equal(prior_trades(full), prior_trades(prefix))
    pd.testing.assert_frame_equal(prior_trades(full), prior_trades(changed))
