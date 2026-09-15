"""Position sizing policies for ES/NQ spreads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd

from ..contracts import PRODUCTS, ProductSpec


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
    max_units_a: int = 20
    max_units_b: int | None = None
    max_contracts: int | None = None
    spec_a: ProductSpec = field(default_factory=lambda: PRODUCTS["ES"])
    spec_b: ProductSpec = field(default_factory=lambda: PRODUCTS["NQ"])

    def __post_init__(self) -> None:
        if self.max_contracts is not None:
            object.__setattr__(self, "max_units_a", max(self.max_units_a, abs(self.es_contracts)))
            if self.max_units_b is None:
                object.__setattr__(self, "max_units_b", self.max_contracts)

    def unit(
        self, signal: int, es_price: float, nq_price: float, beta: float
    ) -> tuple[float, float]:
        es = float(signal * abs(self.es_contracts))
        nq = (
            -signal
            * abs(self.es_contracts)
            * beta
            * es_price
            * self.spec_a.multiplier_usd
            / (nq_price * self.spec_b.multiplier_usd)
        )
        return es, nq

    def size(
        self, signal: int, es_price: float, nq_price: float, beta: float, state: SizerState
    ) -> tuple[int, int]:
        del state
        if signal == 0:
            return 0, 0
        es, nq = self.unit(signal, es_price, nq_price, beta)
        nq_magnitude = max(1, round(abs(nq)))
        if self.max_units_b is not None:
            nq_magnitude = min(nq_magnitude, self.max_units_b)
        es_magnitude = min(abs(round(es)), self.max_units_a)
        es = int(np.sign(es) * es_magnitude)
        return es, -signal * nq_magnitude


@dataclass(frozen=True)
class VolTargetSizer:
    target_daily_vol_usd: float
    lookback_sessions: int = 20
    max_units_a: int = 20
    max_units_b: int | None = None
    max_contracts: int | None = None
    inner: Sizer = field(default_factory=FixedContracts)

    def __post_init__(self) -> None:
        if self.max_contracts is not None:
            object.__setattr__(self, "max_units_a", self.max_contracts)
            if self.max_units_b is None:
                object.__setattr__(self, "max_units_b", self.max_contracts)

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
                max(-self.max_units_a, min(self.max_units_a, round(es))),
                (
                    max(-self.max_units_b, min(self.max_units_b, round(nq)))
                    if self.max_units_b is not None
                    else round(nq)
                ),
            )

        history = state.unit_pnl_daily.dropna().tail(self.lookback_sessions)
        if len(history) < self.lookback_sessions:
            return integerize(unit_es, unit_nq)
        realized = float(history.std(ddof=1))
        if realized <= 0 or not pd.notna(realized):
            return integerize(unit_es, unit_nq)
        scale = max(0.0, self.target_daily_vol_usd / realized)
        limits = [self.max_units_a / max(abs(unit_es), 1)]
        if self.max_units_b is not None:
            limits.append(self.max_units_b / max(abs(unit_nq), 1))
        scale = min(scale, *limits)
        return integerize(unit_es * scale, unit_nq * scale)


@dataclass(frozen=True)
class SizerSpec:
    name: str = "fixed"
    kwargs: dict[str, object] = field(default_factory=dict)


def build_sizer(
    spec: SizerSpec,
    spec_a: ProductSpec | None = None,
    spec_b: ProductSpec | None = None,
) -> Sizer:
    kwargs: dict[str, Any] = dict(spec.kwargs)
    spec_a = spec_a or PRODUCTS["ES"]
    spec_b = spec_b or PRODUCTS["NQ"]
    if spec.name == "fixed":
        return FixedContracts(
            n_es=int(kwargs.get("n_es", 1)),
            n_nq=int(kwargs.get("n_nq", 1)),
        )
    if spec.name == "dollar_neutral":
        return DollarNeutralSizer(
            es_contracts=int(kwargs.get("es_contracts", 1)),
            max_units_a=int(kwargs.get("max_units_a", kwargs.get("max_contracts", 20))),
            max_units_b=(
                int(kwargs["max_units_b"]) if kwargs.get("max_units_b") is not None else None
            ),
            spec_a=spec_a,
            spec_b=spec_b,
        )
    if spec.name == "vol_target":
        inner = kwargs.pop("inner", FixedContracts())
        if isinstance(inner, dict):
            inner_name = str(inner.get("name", "fixed"))
            inner_kwargs = {key: value for key, value in inner.items() if key != "name"}
            inner = build_sizer(SizerSpec(name=inner_name, kwargs=inner_kwargs), spec_a, spec_b)
        if not hasattr(inner, "size"):
            raise TypeError("inner must implement Sizer")
        return VolTargetSizer(
            target_daily_vol_usd=float(kwargs["target_daily_vol_usd"]),
            lookback_sessions=int(kwargs.get("lookback_sessions", 20)),
            max_units_a=int(kwargs.get("max_units_a", kwargs.get("max_contracts", 20))),
            max_units_b=(
                int(kwargs["max_units_b"]) if kwargs.get("max_units_b") is not None else None
            ),
            inner=inner,
        )
    raise ValueError(f"unknown sizer: {spec.name}")
