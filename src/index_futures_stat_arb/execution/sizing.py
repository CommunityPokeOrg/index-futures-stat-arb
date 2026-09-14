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
    def unit(
        self, signal: int, es_price: float, nq_price: float, beta: float
    ) -> tuple[float, float]: ...

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

    def unit(
        self, signal: int, es_price: float, nq_price: float, beta: float
    ) -> tuple[float, float]:
        del es_price, nq_price, beta
        return float(signal * self.n_es), float(-signal * self.n_nq)

    def size(
        self, signal: int, es_price: float, nq_price: float, beta: float, state: SizerState
    ) -> tuple[int, int]:
        del es_price, nq_price, beta, state
        return signal * self.n_es, -signal * self.n_nq


@dataclass(frozen=True)
class DollarNeutralSizer:
    es_contracts: int = 1
    max_contracts: int = 20

    def unit(
        self, signal: int, es_price: float, nq_price: float, beta: float
    ) -> tuple[float, float]:
        es = float(signal * abs(self.es_contracts))
        nq = (
            -signal
            * abs(self.es_contracts)
            * beta
            * es_price
            * PRODUCTS["ES"].multiplier_usd
            / (nq_price * PRODUCTS["NQ"].multiplier_usd)
        )
        return es, nq

    def size(
        self, signal: int, es_price: float, nq_price: float, beta: float, state: SizerState
    ) -> tuple[int, int]:
        del state
        if signal == 0:
            return 0, 0
        es, nq = self.unit(signal, es_price, nq_price, beta)
        nq_magnitude = min(max(1, round(abs(nq))), self.max_contracts)
        return round(es), -signal * nq_magnitude


@dataclass(frozen=True)
class VolTargetSizer:
    target_daily_vol_usd: float
    lookback_sessions: int = 20
    max_contracts: int = 20
    inner: Sizer = field(default_factory=FixedContracts)

    def unit(
        self, signal: int, es_price: float, nq_price: float, beta: float
    ) -> tuple[float, float]:
        return self.inner.unit(signal, es_price, nq_price, beta)

    def size(
        self, signal: int, es_price: float, nq_price: float, beta: float, state: SizerState
    ) -> tuple[int, int]:
        unit_es, unit_nq = self.unit(signal, es_price, nq_price, beta)

        def integerize(es: float, nq: float) -> tuple[int, int]:
            return (
                max(-self.max_contracts, min(self.max_contracts, round(es))),
                max(-self.max_contracts, min(self.max_contracts, round(nq))),
            )

        history = state.unit_pnl_daily.dropna().tail(self.lookback_sessions)
        if len(history) < self.lookback_sessions:
            return integerize(unit_es, unit_nq)
        realized = float(history.std(ddof=1))
        if realized <= 0 or not pd.notna(realized):
            return integerize(unit_es, unit_nq)
        scale = max(0.0, self.target_daily_vol_usd / realized)
        scale = min(scale, self.max_contracts / max(abs(unit_es), abs(unit_nq), 1))
        return integerize(unit_es * scale, unit_nq * scale)


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
            inner_name = str(inner.get("name", "fixed"))
            inner_kwargs = {key: value for key, value in inner.items() if key != "name"}
            inner = build_sizer(SizerSpec(name=inner_name, kwargs=inner_kwargs))
        if not hasattr(inner, "size"):
            raise TypeError("inner must implement Sizer")
        return VolTargetSizer(
            target_daily_vol_usd=float(kwargs["target_daily_vol_usd"]),
            lookback_sessions=int(kwargs.get("lookback_sessions", 20)),
            max_contracts=int(kwargs.get("max_contracts", 20)),
            inner=inner,
        )
    raise ValueError(f"unknown sizer: {spec.name}")
