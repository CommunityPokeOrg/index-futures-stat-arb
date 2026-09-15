"""Databento ingestion, validation, and Parquet persistence."""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import TypeVar

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from ..config import IngestConfig, api_key_from_env
from ..schema import (
    BARS_SCHEMA,
    FileEntry,
    Manifest,
    make_dataset_id,
    product_from_contract,
    sha256_file,
)
from ..sessions import is_rth, session_bounds_utc, session_date
from . import BarClient

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


class IngestError(RuntimeError):
    """Raised when a retryable ingestion operation exhausts its retries."""


class DataValidationError(ValueError):
    """Raised when bars contain a hard validation failure."""


@dataclass(frozen=True)
class NormalizeReport:
    n_rows: int
    n_duplicates: int


@dataclass
class ValidationReport:
    n_rows: int
    n_duplicates: int
    n_ohlc_violations: int
    n_nonpositive_prices: int
    n_negative_volume: int
    n_unsorted: int
    gaps: list[tuple[str, pd.Timestamp, pd.Timestamp, int]] = field(default_factory=list)
    ok: bool = False


class DatabentoClient:
    """Thin adapter around ``databento.Historical``."""

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or api_key_from_env()
        if not key:
            raise RuntimeError("DATABENTO_API_KEY is not set")
        try:
            import databento as db
        except ImportError as exc:
            raise ImportError(
                "databento is required for DatabentoClient; install the databento extra"
            ) from exc
        self._client = db.Historical(key)
        self.client_version = getattr(db, "__version__", "unknown")

    def get_range(
        self,
        dataset: str,
        schema: str,
        symbols: list[str],
        stype_in: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        result = self._client.timeseries.get_range(
            dataset=dataset,
            schema=schema,
            symbols=symbols,
            stype_in=stype_in,
            start=start,
            end=end,
        )
        return result.to_df(pretty_px=False, pretty_ts=True)


def normalize_bars(
    raw: pd.DataFrame,
    price_scale: float,
    source: str,
    interval: str,
) -> pd.DataFrame:
    """Convert Databento wire values into the canonical bars schema."""
    if raw.empty:
        return pd.DataFrame(
            {
                name: pd.Series(dtype=_pandas_dtype(field.type))
                for name, field in zip(BARS_SCHEMA.names, BARS_SCHEMA, strict=True)
            }
        )
    out = raw.reset_index() if "ts_event" not in raw.columns else raw.copy()
    out["ts_event"] = pd.to_datetime(out["ts_event"], utc=True)
    if "ts_recv" not in out:
        out["ts_recv"] = pd.NaT
    else:
        out["ts_recv"] = pd.to_datetime(out["ts_recv"], utc=True)
    out["contract"] = out["symbol"].astype("string")
    out["product"] = out["contract"].map(product_from_contract)
    out["source"] = source
    out["instrument_id"] = out["instrument_id"].astype("Int64")
    for column in ("open", "high", "low", "close"):
        out[column] = out[column].astype("float64") * price_scale
    out["volume"] = out["volume"].astype("int64")
    out["interval"] = interval
    out["session_date"] = session_date(pd.DatetimeIndex(out["ts_event"]))
    out["is_rth"] = is_rth(pd.DatetimeIndex(out["ts_event"]))
    out = out[list(BARS_SCHEMA.names)]
    out = out.sort_values(["product", "contract", "ts_event"], kind="stable")
    duplicate_mask = out.duplicated(["contract", "ts_event"], keep="first")
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count:
        LOGGER.info("dropped %d duplicate bars during normalization", duplicate_count)
        out = out.loc[~duplicate_mask]
    return out.reset_index(drop=True)


def _pandas_dtype(dtype: pa.DataType) -> str:
    if pa.types.is_timestamp(dtype):
        return "datetime64[ns, UTC]"
    if pa.types.is_date32(dtype):
        return "object"
    if pa.types.is_boolean(dtype):
        return "bool"
    if pa.types.is_integer(dtype):
        return "int64"
    if pa.types.is_floating(dtype):
        return "float64"
    return "string"


def _interval_delta(interval: str) -> pd.Timedelta:
    match = re.fullmatch(r"(\d+)([smhd])", interval)
    if not match:
        raise ValueError(f"unsupported bar interval: {interval!r}")
    unit = {"s": "s", "m": "m", "h": "h", "d": "d"}[match.group(2)]
    return pd.Timedelta(f"{match.group(1)}{unit}")


def validate_bars(df: pd.DataFrame, interval: str) -> ValidationReport:
    """Check hard bar invariants and report non-fatal timestamp gaps."""
    if df.empty:
        return ValidationReport(0, 0, 0, 0, 0, 0, [], True)
    keys = ["contract", "ts_event"]
    n_duplicates = int(df.duplicated(keys).sum())
    n_ohlc = int(
        (
            (df["low"] > df[["open", "close"]].min(axis=1))
            | (df["high"] < df[["open", "close"]].max(axis=1))
        ).sum()
    )
    prices = df[["open", "high", "low", "close"]]
    n_nonpositive = int((prices <= 0).any(axis=1).sum())
    n_negative_volume = int((df["volume"] < 0).sum())
    event = pd.to_datetime(df["ts_event"], utc=True)
    n_unsorted = int((event.groupby(df["contract"], sort=False).diff() <= pd.Timedelta(0)).sum())
    delta = _interval_delta(interval)
    grouped = (
        pd.DataFrame(
            {
                "contract": df["contract"].to_numpy(),
                "session_date": df["session_date"].to_numpy(),
                "_ts": event.to_numpy(),
            }
        )
        .sort_values(["contract", "session_date", "_ts"])
        .reset_index(drop=True)
    )
    grouped["_previous"] = grouped.groupby(["contract", "session_date"], sort=False)["_ts"].shift()
    grouped["_delta"] = grouped["_ts"] - grouped["_previous"]
    gap_rows = grouped.loc[grouped["_delta"] > delta]
    gaps: list[tuple[str, pd.Timestamp, pd.Timestamp, int]] = []
    delta_ns = delta.value
    missing_counts = gap_rows["_delta"].astype("timedelta64[ns]").astype("int64") // delta_ns - 1
    for contract, previous, current, missing in zip(
        gap_rows["contract"],
        pd.to_datetime(gap_rows["_previous"], utc=True),
        pd.to_datetime(gap_rows["_ts"], utc=True),
        missing_counts,
        strict=True,
    ):
        gaps.append((str(contract), pd.Timestamp(previous), pd.Timestamp(current), int(missing)))
    ok = not any((n_duplicates, n_ohlc, n_nonpositive, n_negative_volume, n_unsorted))
    return ValidationReport(
        n_rows=len(df),
        n_duplicates=n_duplicates,
        n_ohlc_violations=n_ohlc,
        n_nonpositive_prices=n_nonpositive,
        n_negative_volume=n_negative_volume,
        n_unsorted=n_unsorted,
        gaps=gaps,
        ok=ok,
    )


def raise_on_invalid(report: ValidationReport) -> None:
    if not report.ok:
        raise DataValidationError(f"bar validation failed: {report}")


def write_partitioned(
    df: pd.DataFrame,
    data_root: str | Path,
    source: str,
    dataset: str,
    schema: str,
) -> list[Path]:
    """Write one deterministic Parquet file per product/contract/session date."""
    paths: list[Path] = []
    base = Path(data_root) / "raw" / f"source={source}" / f"dataset={dataset}" / f"schema={schema}"
    if df.empty:
        return paths
    for (product, contract, day), group in df.groupby(
        ["product", "contract", "session_date"], sort=True
    ):
        destination = (
            base / f"product={product}" / f"contract={contract}" / f"date={day}" / "part-0.parquet"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(
            group[list(BARS_SCHEMA.names)].reset_index(drop=True),
            schema=BARS_SCHEMA,
            preserve_index=False,
        )
        pq.write_table(table, destination, compression="zstd", compression_level=3)
        paths.append(destination)
    return paths


def read_partitioned(
    data_root: str | Path,
    source: str,
    dataset: str,
    schema: str,
    products: Iterable[str] | None = None,
    contracts: Iterable[str] | None = None,
    start: str | date | None = None,
    end: str | date | None = None,
) -> pd.DataFrame:
    base = Path(data_root) / "raw" / f"source={source}" / f"dataset={dataset}" / f"schema={schema}"
    if not base.exists():
        return pd.DataFrame(columns=BARS_SCHEMA.names)
    table_dataset = ds.dataset(base, format="parquet", partitioning="hive")
    predicate = None
    if products is not None:
        predicate = ds.field("product").isin(list(products))
    if contracts is not None:
        expression = ds.field("contract").isin(list(contracts))
        predicate = expression if predicate is None else predicate & expression
    if start is not None:
        expression = ds.field("date") >= pd.Timestamp(start).date().isoformat()
        predicate = expression if predicate is None else predicate & expression
    if end is not None:
        expression = ds.field("date") < pd.Timestamp(end).date().isoformat()
        predicate = expression if predicate is None else predicate & expression
    result = table_dataset.to_table(filter=predicate).to_pandas()
    if "date" in result:
        result = result.drop(columns=["date"])
    result = result[list(BARS_SCHEMA.names)]
    return result.sort_values(["product", "contract", "ts_event"]).reset_index(drop=True)


@dataclass
class Checkpoint:
    dataset_id: str
    data_root: Path = Path("data")
    completed_chunks: list[str] = field(default_factory=list)

    @property
    def path(self) -> Path:
        return self.data_root / "checkpoints" / f"{self.dataset_id}.json"

    @classmethod
    def load(cls, dataset_id: str, data_root: str | Path) -> Checkpoint:
        checkpoint = cls(dataset_id, Path(data_root))
        if checkpoint.path.exists():
            payload = json.loads(checkpoint.path.read_text())
            checkpoint.completed_chunks = list(payload.get("completed_chunks", []))
        return checkpoint

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"completed_chunks": self.completed_chunks}, indent=2) + "\n"
        )
        os.replace(temporary, self.path)


def iter_chunks(start: str, end: str, chunk_days: int) -> list[tuple[str, str]]:
    if chunk_days < 1:
        raise ValueError("chunk_days must be at least 1")
    current = pd.Timestamp(start).date()
    finish = pd.Timestamp(end).date()
    chunks: list[tuple[str, str]] = []
    while current < finish:
        next_date = min(current + timedelta(days=chunk_days), finish)
        chunks.append((current.isoformat(), next_date.isoformat()))
        current = next_date
    return chunks


def with_retries(
    fn: Callable[[], T],
    max_retries: int,
    backoff_base_s: float,
    backoff_max_s: float,
    sleep: Callable[[float], None] = time.sleep,
    retry_on: tuple[type[BaseException], ...] = (ConnectionError, TimeoutError, RuntimeError),
    rng: random.Random | None = None,
) -> T:
    random_source = rng or random.Random()
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except retry_on as exc:
            if attempt >= max_retries:
                raise IngestError("operation failed after retries") from exc
            ceiling = min(backoff_max_s, backoff_base_s * (2**attempt))
            sleep(random_source.uniform(0.0, ceiling))
    raise AssertionError("unreachable")


def run_ingest(
    cfg: IngestConfig,
    client: BarClient,
    *,
    resume: bool = True,
    sleep: Callable[[float], None] = time.sleep,
) -> Manifest:
    dataset_id = make_dataset_id(
        cfg.source, cfg.dataset, cfg.schema, cfg.symbols, cfg.start, cfg.end
    )
    checkpoint = Checkpoint.load(dataset_id, cfg.data_root)
    if not resume:
        checkpoint.completed_chunks = []
        checkpoint.save()
    for chunk_start, chunk_end in iter_chunks(cfg.start, cfg.end, cfg.chunk_days):
        chunk_key = f"{chunk_start}/{chunk_end}"
        if resume and chunk_key in checkpoint.completed_chunks:
            continue
        request_start = session_bounds_utc(pd.Timestamp(chunk_start).date())[0]
        request_end = session_bounds_utc(pd.Timestamp(chunk_end).date() - timedelta(days=1))[1]

        def fetch_chunk(
            request_start: str = request_start.isoformat(),
            request_end: str = request_end.isoformat(),
        ) -> pd.DataFrame:
            return client.get_range(
                cfg.dataset,
                cfg.schema,
                list(cfg.symbols),
                cfg.stype_in,
                request_start,
                request_end,
            )

        raw = with_retries(
            fetch_chunk,
            cfg.max_retries,
            cfg.backoff_base_s,
            cfg.backoff_max_s,
            sleep=sleep,
        )
        interval = cfg.schema.removeprefix("ohlcv-")
        normalized = normalize_bars(raw, cfg.price_scale, cfg.source, interval)
        report = validate_bars(normalized, interval)
        raise_on_invalid(report)
        write_partitioned(normalized, cfg.data_root, cfg.source, cfg.dataset, cfg.schema)
        if chunk_key not in checkpoint.completed_chunks:
            checkpoint.completed_chunks.append(chunk_key)
        checkpoint.save()
    files: list[FileEntry] = []
    for path in sorted(
        (
            cfg.data_root
            / "raw"
            / f"source={cfg.source}"
            / f"dataset={cfg.dataset}"
            / f"schema={cfg.schema}"
        ).glob("product=*/contract=*/date=*/part-0.parquet")
    ):
        table = pq.read_table(path)
        files.append(FileEntry(str(path), table.num_rows, sha256_file(path)))
    manifest = Manifest(
        dataset_id=dataset_id,
        source=cfg.source,
        dataset=cfg.dataset,
        schema=cfg.schema,
        stype_in=cfg.stype_in,
        symbols=list(cfg.symbols),
        start=cfg.start,
        end=cfg.end,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        client_version=client.client_version,
        price_scale=cfg.price_scale,
        timezone_in="UTC",
        files=files,
        row_count=sum(entry.rows for entry in files),
        notes=(
            "synthetic fixture; no network accessed"
            if cfg.source == "synthetic"
            else "credentials via DATABENTO_API_KEY env; not stored"
        ),
    )
    manifest.to_json(cfg.data_root / "manifests" / f"{dataset_id}.json")
    return manifest
