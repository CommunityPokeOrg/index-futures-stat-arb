"""Clearly-labelled, deterministic offline fixtures for ingestion tests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import IngestConfig
from .contracts import parse_contract
from .ingest import BarClient
from .ingest.databento import run_ingest
from .schema import Manifest, product_from_contract
from .sessions import ET, is_business_day


@dataclass(frozen=True)
class ContractSpecLite:
    contract: str = ""
    product: str | None = None
    first_trade: date | None = None
    expiry: date | None = None


def _contract_spec(contract: str) -> ContractSpecLite:
    parsed = parse_contract(contract)
    return ContractSpecLite(
        contract,
        parsed.product,
        parsed.first_trade_date,
        parsed.expiry_date,
    )


class SyntheticBarClient:
    """Synthetic fixture client; it never contacts Databento."""

    client_version = "synthetic-fixture"

    def __init__(
        self,
        seed: int = 42,
        contracts: dict[str, ContractSpecLite] | None = None,
    ) -> None:
        self.seed = seed
        self.contracts = contracts
        self.calls: list[tuple[str, str]] = []
        self._common_cache: dict[date, np.ndarray] = {}
        self._common_totals: dict[date, float] = {}
        self._ou_cache: dict[date, np.ndarray] = {}
        self._ou_ends: dict[date, float] = {}

    def _spec(self, contract: str) -> ContractSpecLite:
        spec = (self.contracts or {}).get(contract) or _contract_spec(contract)
        return spec if spec.contract else replace(spec, contract=contract)

    def get_range(
        self,
        dataset: str,
        schema: str,
        symbols: list[str],
        stype_in: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        del dataset, schema, stype_in
        self.calls.append((start, end))
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC")
        frames: list[pd.DataFrame] = []
        for symbol in sorted(symbols):
            spec = self._spec(symbol)
            if spec.expiry is None or spec.first_trade is None:
                continue
            frames.append(self._bars_for_contract(spec, start_ts, end_ts))
        if not frames:
            return pd.DataFrame(
                columns=[
                    "ts_event",
                    "ts_recv",
                    "symbol",
                    "instrument_id",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]
            )
        return pd.concat(frames, ignore_index=True)

    def _bars_for_contract(
        self,
        spec: ContractSpecLite,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        assert spec.first_trade is not None and spec.expiry is not None
        product = spec.product or product_from_contract(spec.contract)
        expiry_cutoff = pd.Timestamp(datetime.combine(spec.expiry, time(9, 30)), tz=ET).tz_convert(
            "UTC"
        )
        first_trade_cutoff = pd.Timestamp(spec.first_trade, tz="UTC")
        first_session = max(spec.first_trade, start.tz_convert(ET).date() - timedelta(days=1))
        last_session = min(spec.expiry, end.tz_convert(ET).date() + timedelta(days=1))
        frames: list[pd.DataFrame] = []
        current = first_session
        while current <= last_session:
            if is_business_day(current):
                opening = pd.Timestamp(
                    datetime.combine(current - timedelta(days=1), time(18)), tz=ET
                )
                closing = pd.Timestamp(datetime.combine(current, time(17)), tz=ET)
                values = pd.date_range(
                    opening.tz_convert("UTC"),
                    closing.tz_convert("UTC"),
                    freq="min",
                    inclusive="left",
                )
                mask = (
                    (values >= start)
                    & (values < end)
                    & (values >= first_trade_cutoff)
                    & (values < expiry_cutoff)
                )
                if mask.any():
                    positions = np.flatnonzero(mask)
                    common = self._common_session_path(current, len(values))
                    ou = self._ou_session_path(current, len(values))
                    level = np.log(18000.0) + common
                    if product == "ES":
                        level = np.log(5000.0) + 0.75 * common + ou
                    spot = np.exp(level)
                    contract_year = parse_contract(spec.contract).year
                    contract_quarter = (
                        contract_year * 4
                        + (parse_contract(spec.contract).expiry_date.month - 1) // 3
                    )
                    current_quarter = current.year * 4 + (current.month - 1) // 3
                    quarters_to_expiry = max(0, contract_quarter - current_quarter)
                    carry = 1.0 + 0.004 * quarters_to_expiry
                    close = spot * carry
                    open_price = np.r_[close[0], close[:-1]]
                    high = np.maximum(open_price, close) * 1.00015
                    low = np.minimum(open_price, close) * 0.99985
                    volume_seed = self._seed("volume", spec.contract, current)
                    volume_rng = np.random.default_rng(volume_seed)
                    days_to_expiry = max(0, (spec.expiry - current).days)
                    liquidity = np.clip(days_to_expiry / 8.0, 0.02, 1.0)
                    volume = volume_rng.poisson(120 * liquidity, len(values)).astype(np.int64)
                    frame = pd.DataFrame(
                        {
                            "ts_event": values[positions],
                            "ts_recv": values[positions] + pd.Timedelta(seconds=1),
                            "symbol": spec.contract,
                            "instrument_id": self._seed("instrument", spec.contract),
                            "open": np.rint(open_price[positions] * 1e9).astype(np.int64),
                            "high": np.rint(high[positions] * 1e9).astype(np.int64),
                            "low": np.rint(low[positions] * 1e9).astype(np.int64),
                            "close": np.rint(close[positions] * 1e9).astype(np.int64),
                            "volume": volume[positions],
                        }
                    )
                    frames.append(frame)
            current += timedelta(days=1)
        if not frames:
            return pd.DataFrame(
                columns=[
                    "ts_event",
                    "ts_recv",
                    "symbol",
                    "instrument_id",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]
            )
        return pd.concat(frames, ignore_index=True)

    def _seed(self, *parts: object) -> int:
        payload = ":".join(str(part) for part in (self.seed, *parts))
        return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little") % (2**32)

    def _business_dates_to(self, target: date) -> list[date]:
        current = date(2020, 1, 1)
        dates: list[date] = []
        while current <= target:
            if is_business_day(current):
                dates.append(current)
            current += timedelta(days=1)
        return dates

    def _common_session_path(self, target: date, length: int) -> np.ndarray:
        level = 0.0
        for current in self._business_dates_to(target):
            path = self._common_cache.get(current)
            if path is None:
                rng = np.random.default_rng(self._seed("common", current))
                increments = rng.normal(0.0, 0.00006, 1380)
                path = np.cumsum(increments)
                self._common_cache[current] = path
                self._common_totals[current] = float(path[-1])
            if current == target:
                return level + self._common_cache[current][:length]
            level += self._common_totals[current]
        raise ValueError(f"session date is before fixture epoch: {target}")

    def _ou_session_path(self, target: date, length: int) -> np.ndarray:
        state = 0.0
        theta = 1.0 / 240.0
        stationary_std = 0.0015
        epsilon_std = stationary_std * np.sqrt(1.0 - (1.0 - theta) ** 2)
        for current in self._business_dates_to(target):
            path = self._ou_cache.get(current)
            if path is None:
                rng = np.random.default_rng(self._seed("ou", current))
                epsilon = rng.normal(0.0, epsilon_std, 1380)
                values = np.empty(1380)
                for index, shock in enumerate(epsilon):
                    state = state * (1.0 - theta) + shock
                    values[index] = state
                self._ou_cache[current] = values
                self._ou_ends[current] = state
            else:
                state = self._ou_ends[current]
            if current == target:
                return self._ou_cache[current][:length]
        raise ValueError(f"session date is before fixture epoch: {target}")


class FlakyClient:
    """Fixture wrapper that fails a fixed number of initial calls."""

    def __init__(self, client: BarClient, n_failures: int) -> None:
        self.client = client
        self.n_failures = n_failures
        self.calls = 0
        self.client_version = client.client_version

    def get_range(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        self.calls += 1
        if self.calls <= self.n_failures:
            raise ConnectionError("synthetic transient failure")
        return self.client.get_range(*args, **kwargs)


def write_fixture_dataset(
    root: str | Path,
    start: str,
    end: str,
    seed: int = 42,
) -> Manifest:
    symbols = ("ESH6", "ESM6", "NQH6", "NQM6")
    cfg = IngestConfig(
        symbols=symbols,
        start=start,
        end=end,
        data_root=Path(root),
        source="synthetic",
    )
    return run_ingest(cfg, SyntheticBarClient(seed=seed))
