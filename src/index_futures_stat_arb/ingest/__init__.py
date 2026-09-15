"""Interfaces for historical bar ingestion."""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class BarClient(Protocol):
    client_version: str

    def get_range(
        self,
        dataset: str,
        schema: str,
        symbols: list[str],
        stype_in: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Return Databento-like raw bars for a half-open time range."""
