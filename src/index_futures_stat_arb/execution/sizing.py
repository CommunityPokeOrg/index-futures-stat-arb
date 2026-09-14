"""Position sizing policies for ES/NQ spreads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd

from ..contracts import PRODUCTS


@dataclass
class SizerState:
    unit_pnl_daily: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


class Sizer(Protocol):
    def size(
        self,
        signal: int,
        es_price: float,
        nq_price: float,
        beta: float,
        state: SizerState,
    ) -> tuple[int, int]: ...


@dataclass(frozen=True)
class FixedContracts:
    n_es: int = 1
    n_nq: int = 1

    def size(
        self, signal: int, es_price: float, nq_price: float, beta: float, state: SizerState
    ) -> tuple[int, int]:
        del es_price, nq_price, beta, state
        return signal * self.n_es, -signal * self.n_nq


@dataclass(frozen=True)
class DollarNeutralSizer:
    es_contracts: int = 1
    max_contracts: int = 20

    def size(
        self, signal: int, es_price: float, nq_price: float, beta: float, state: SizerState
    ) -> tuple[int, int]:
        del state
        if signal == 0:
            return 0, 0
        es = abs(self.es_contracts)
        nq = round(
            beta
            * es
            * es_price
            * PRODUCTS["ES"].multiplier_usd
            / (nq_price * PRODUCTS["NQ"].multiplier_usd)
        )
        nq = min(max(1, abs(nq)), self.max_contracts)
        return signal * es, -signal * nq


@dataclass(frozen=True)
class VolTargetSizer:
    target_daily_vol_usd: float
    lookback_sessions: int = 20
    max_contracts: int = 20
    inner: Sizer = field(default_factory=FixedContracts)

    def size(
        self, signal: int, es_price: float, nq_price: float, beta: float, state: SizerState
    ) -> tuple[int, int]:
        base_es, base_nq = self.inner.size(signal, es_price, nq_price, beta, state)
        history = state.unit_pnl_daily.dropna().tail(self.lookback_sessions)
        if len(history) < self.lookback_sessions:
            return base_es, base_nq
        realized = float(history.std(ddof=1))
        if realized <= 0 or not pd.notna(realized):
            return base_es, base_nq
        scale = max(0.0, self.target_daily_vol_usd / realized)
        scale = min(scale, self.max_contracts / max(abs(base_es), abs(base_nq), 1))
        return round(base_es * scale), round(base_nq * scale)


@dataclass(frozen=True)
class SizerSpec:
    name: str = "fixed"
    kwargs: dict[str, object] = field(default_factory=dict)


def build_sizer(spec: SizerSpec) -> Sizer:
    kwargs: dict[str, Any] = dict(spec.kwargs)
    if spec.name == "fixed":
        return FixedContracts(
            n_es=int(kwargs.get("n_es", 1)),
            n_nq=int(kwargs.get("n_nq", 1)),
        )
    if spec.name == "dollar_neutral":
        return DollarNeutralSizer(
            es_contracts=int(kwargs.get("es_contracts", 1)),
            max_contracts=int(kwargs.get("max_contracts", 20)),
        )
    if spec.name == "vol_target":
        inner = kwargs.pop("inner", FixedContracts())
        if isinstance(inner, dict):
            inner = build_sizer(SizerSpec(**inner))
        if not hasattr(inner, "size"):
            raise TypeError("inner must implement Sizer")
        return VolTargetSizer(
            target_daily_vol_usd=float(kwargs["target_daily_vol_usd"]),
            lookback_sessions=int(kwargs.get("lookback_sessions", 20)),
            max_contracts=int(kwargs.get("max_contracts", 20)),
            inner=inner,
        )
    raise ValueError(f"unknown sizer: {spec.name}")
