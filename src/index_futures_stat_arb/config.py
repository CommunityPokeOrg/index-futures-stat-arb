"""Configuration for the per-contract ingestion pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib

from .rolls import RollConfig


@dataclass(frozen=True)
class IngestConfig:
    """Validated settings for one ingestion dataset."""

    symbols: tuple[str, ...]
    start: str
    end: str
    dataset: str = "GLBX.MDP3"
    schema: str = "ohlcv-1m"
    stype_in: str = "raw_symbol"
    chunk_days: int = 1
    max_retries: int = 5
    backoff_base_s: float = 0.5
    backoff_max_s: float = 30.0
    data_root: Path = Path("data")
    source: str = "databento"
    price_scale: float = 1e-9

    def __post_init__(self) -> None:
        if not self.symbols:
            raise ValueError("symbols must be non-empty")
        if self.chunk_days < 1:
            raise ValueError("chunk_days must be at least 1")
        if self.start >= self.end:
            raise ValueError("start must be before end")
        object.__setattr__(self, "symbols", tuple(self.symbols))
        object.__setattr__(self, "data_root", Path(self.data_root))


@dataclass(frozen=True)
class YahooConfig:
    interval: str = "1d"
    cache_dir: Path = Path("data/yahoo")
    bar_minutes: int | None = None
    rth_only: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "cache_dir", Path(self.cache_dir))


def load_config(path: str | Path) -> IngestConfig:
    """Load an ``[ingest]`` TOML section."""
    with Path(path).open("rb") as stream:
        values: dict[str, Any] = tomllib.load(stream).get("ingest", {})
    if not values:
        raise ValueError("configuration is missing the [ingest] section")
    if "symbols" in values:
        values["symbols"] = tuple(values["symbols"])
    return IngestConfig(**values)


def api_key_from_env() -> str | None:
    """Return the Databento key without persisting or logging it."""
    return os.environ.get("DATABENTO_API_KEY") or None


def load_roll_config(path: str | Path, rule: str | None = None) -> RollConfig:
    with Path(path).open("rb") as stream:
        values: dict[str, Any] = tomllib.load(stream).get("rolls", {})
    if rule is not None:
        values["rule"] = rule
    return RollConfig(**values)


def load_simulation_config(path: str | Path) -> Any:
    """Load simulation, cost, sizer, and roll sections from TOML."""
    from .execution.costs import CostModel
    from .execution.engine import SimulationConfig
    from .execution.sizing import SizerSpec

    with Path(path).open("rb") as stream:
        payload = tomllib.load(stream)
    values = dict(payload.get("simulation", {}))
    values["costs"] = CostModel(**payload.get("costs", {}))
    sizer_values = payload.get("sizer", {})
    values["sizer"] = SizerSpec(
        name=sizer_values.get("name", "fixed"),
        kwargs={key: value for key, value in sizer_values.items() if key != "name"},
    )
    values["roll"] = RollConfig(**payload.get("rolls", {}))
    if "products" in values:
        values["products"] = tuple(values["products"])
    return SimulationConfig(**values)


def load_yahoo_config(path: str | Path) -> YahooConfig:
    """Load Yahoo adapter settings from the ``[yahoo]`` TOML section."""
    with Path(path).open("rb") as stream:
        values: dict[str, Any] = tomllib.load(stream).get("yahoo", {})
    return YahooConfig(**values)
