"""Execution simulation and transaction-cost utilities."""

from .costs import CostModel
from .engine import (
    SimulationConfig,
    SimulationResult,
    load_pair_bars,
    run_simulation,
)
from .hedge import HedgeMethod, KalmanHedge, RollingEGResult, rolling_engle_granger
from .lookahead import LookaheadError, assert_no_lookahead
from .metrics import compute_metrics
from .sizing import (
    DollarNeutralSizer,
    FixedContracts,
    Sizer,
    SizerSpec,
    SizerState,
    VolTargetSizer,
    build_sizer,
)

__all__ = [
    "CostModel",
    "DollarNeutralSizer",
    "FixedContracts",
    "HedgeMethod",
    "KalmanHedge",
    "LookaheadError",
    "SimulationConfig",
    "SimulationResult",
    "Sizer",
    "SizerSpec",
    "SizerState",
    "RollingEGResult",
    "VolTargetSizer",
    "assert_no_lookahead",
    "build_sizer",
    "compute_metrics",
    "load_pair_bars",
    "rolling_engle_granger",
    "run_simulation",
]
