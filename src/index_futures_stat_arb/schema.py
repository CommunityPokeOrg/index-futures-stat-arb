"""Arrow schemas and manifests for ingestion artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa

BARS_SCHEMA = pa.schema(
    [
        pa.field("ts_event", pa.timestamp("ns", tz="UTC")),
        pa.field("ts_recv", pa.timestamp("ns", tz="UTC")),
        pa.field("source", pa.string()),
        pa.field("product", pa.string()),
        pa.field("contract", pa.string()),
        pa.field("instrument_id", pa.int64()),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("volume", pa.int64()),
        pa.field("interval", pa.string()),
        pa.field("session_date", pa.date32()),
        pa.field("is_rth", pa.bool_()),
    ]
)

CONTRACTS_SCHEMA = pa.schema(
    [
        pa.field("product", pa.string()),
        pa.field("contract", pa.string()),
        pa.field("month_code", pa.string()),
        pa.field("year", pa.int16()),
        pa.field("first_trade_date", pa.date32()),
        pa.field("last_trade_date", pa.date32()),
        pa.field("expiration_ts", pa.timestamp("ns", tz="UTC")),
        pa.field("multiplier_usd", pa.float64()),
        pa.field("tick_size", pa.float64()),
        pa.field("tick_value_usd", pa.float64()),
        pa.field("exchange", pa.string()),
        pa.field("currency", pa.string()),
        pa.field("source", pa.string()),
        pa.field("asof", pa.date32()),
    ]
)

ROLL_CALENDAR_SCHEMA = pa.schema(
    [
        pa.field("product", pa.string()),
        pa.field("rule", pa.string()),
        pa.field("from_contract", pa.string()),
        pa.field("to_contract", pa.string()),
        pa.field("roll_session_date", pa.date32()),
        pa.field("decision_basis", pa.string()),
        pa.field("panama_offset", pa.float64()),
        pa.field("ratio_factor", pa.float64()),
        pa.field("joint_roll_session_date", pa.date32()),
        pa.field("asof", pa.date32()),
    ]
)


@dataclass(frozen=True)
class FileEntry:
    path: str
    rows: int
    sha256: str


@dataclass
class Manifest:
    dataset_id: str
    source: str
    dataset: str
    schema: str
    stype_in: str
    symbols: list[str]
    start: str
    end: str
    retrieved_at: str
    client_version: str
    price_scale: float
    timezone_in: str
    files: list[FileEntry]
    row_count: int
    notes: str

    def to_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    @classmethod
    def from_json(cls, path: str | Path) -> Manifest:
        payload: dict[str, Any] = json.loads(Path(path).read_text())
        payload["files"] = [FileEntry(**entry) for entry in payload["files"]]
        return cls(**payload)


def make_dataset_id(
    source: str,
    dataset: str,
    schema: str,
    symbols: list[str] | tuple[str, ...],
    start: str,
    end: str,
) -> str:
    """Build a stable, order-independent dataset identifier."""
    symbol_text = ",".join(sorted(symbols))
    return f"{source}-{dataset}-{schema}-{symbol_text}-{start[:10]}-{end[:10]}"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def product_from_contract(contract: str) -> str:
    """Extract an equity-index root from a CME raw symbol."""
    match = re.fullmatch(r"([A-Z]+)[HMUZ]\d{1,2}", contract)
    if not match:
        raise ValueError(f"invalid futures contract symbol: {contract!r}")
    return match.group(1)
