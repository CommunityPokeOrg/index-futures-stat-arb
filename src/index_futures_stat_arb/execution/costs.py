"""Explicit transaction costs for futures simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from ..contracts import ProductSpec


@dataclass(frozen=True)
class CostModel:
    """Approximate CME ES/NQ retail-style costs; these are assumptions."""

    commission_per_contract_usd: float = 1.25
    exchange_fees_per_contract_usd: float = 1.38
    slippage_ticks: float = 1.0
    half_spread_ticks: float = 0.5

    def fill_price(
        self,
        product_spec: ProductSpec,
        reference_price: float,
        side: Literal[1, -1],
    ) -> float:
        if side not in (1, -1):
            raise ValueError("side must be 1 (buy) or -1 (sell)")
        distance = (self.slippage_ticks + self.half_spread_ticks) * product_spec.tick_size
        raw = reference_price + side * distance
        grid = raw / product_spec.tick_size
        rounded = math.ceil(grid - 1e-12) if side == 1 else math.floor(grid + 1e-12)
        return rounded * product_spec.tick_size

    def fees_usd(self, product_spec: ProductSpec, contracts: int) -> float:
        del product_spec
        return abs(contracts) * (
            self.commission_per_contract_usd + self.exchange_fees_per_contract_usd
        )
