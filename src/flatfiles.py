"""Polygon Flat Files downloader + filter + partitioner.

For each (endpoint, trading day) we:
  1. Download the full-market gzipped CSV from S3
     (e.g. `us_options_opra/trades_v1/2026/03/2026-03-13.csv.gz`).
  2. Lazily scan it with Polars, filtering to rows where
     `ticker` starts with `O:<TICKER>` (e.g. `O:GOOGL`).
  3. Partition the filtered rows by contract ticker and write one
     parquet per (contract, day) to the same cache paths our REST
     fetchers use — so downstream code is unchanged.
  4. Delete the raw download.
  5. Write a marker file indicating this (endpoint, day) is done.

After marker is written, `fetch_trades` / `fetch_quotes` read cached
parquets directly (or return empty if the contract had no activity
that day). No REST calls.

Processing is one-day-at-a-time so peak disk stays small. Downloads are
sequential per day but the S3 client itself is multi-threaded, so a
single-day download saturates the connection.
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import boto3
import polars as pl
from botocore.client import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from config import (
    CLASSIFICATION_MODE,
    FLATFILES_BUCKET,
    FLATFILES_CONCURRENCY,
    FLATFILES_ENDPOINT,
    FLATFILES_QUOTES_PREFIX,
    FLATFILES_TRADES_PREFIX,
)
from src import storage

logger = logging.getLogger(__name__)

load_dotenv()


# ---------- S3 client ------------------------------------------------------

def _get_s3_client():
    key = os.environ.get("MASSIVE_S3_ACCESS_KEY")
    secret = os.environ.get("MASSIVE_S3_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError(
            "MASSIVE_S3_ACCESS_KEY / MASSIVE_S3_SECRET_KEY are not set. "
            "Get them from https://polygon.io/dashboard/flat-files and add "
            "them to .env."
        )
    return boto3.client(
        "s3",
        endpoint_url=FLATFILES_ENDPOINT,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        config=Config(signature_version="s3v4"),
    )


# ---------- Key format -----------------------------------------------------

def _s3_key(endpoint: str, day: date) -> str:
    prefix = (
        FLATFILES_TRADES_PREFIX if endpoint == "trades" else FLATFILES_QUOTES_PREFIX
    )
    return f"{prefix}/{day.year:04d}/{day.month:02d}/{day.isoformat()}.csv.gz"


# ---------- Schemas -------------------------------------------------------
# Narrow to the columns we actually consume. Matches what the REST fetchers
# already cache, so classify_trades etc. work on either source.

_TRADES_SELECT = {
    "ticker": pl.String,
    "sip_timestamp": pl.Int64,
    "price": pl.Float64,
    "size": pl.Float64,
    "exchange": pl.Int64,
}

_QUOTES_SELECT = {
    "ticker": pl.String,
    "sip_timestamp": pl.Int64,
    "bid_price": pl.Float64,
    "ask_price": pl.Float64,
    "bid_size": pl.Float64,
    "ask_size": pl.Float64,
    "sequence_number": pl.Int64,
}


# ---------- Core routine --------------------------------------------------

def _process_one_day(
    endpoint: str,
    day: date,
    underlying: str,
    s3_client,
) -> tuple[int, int]:
    """Download one day's flat file, filter, partition, delete, mark done.

    Returns (n_contracts_written, n_rows_written). No-op if marker exists.
    """
    if storage.flatfile_marker_exists(endpoint, day, underlying):
        return 0, 0

    key = _s3_key(endpoint, day)
    select = _TRADES_SELECT if endpoint == "trades" else _QUOTES_SELECT
    ticker_prefix = f"O:{underlying}"

    with tempfile.NamedTemporaryFile(
        prefix=f"flatfile-{endpoint}-{day}-", suffix=".csv.gz", delete=False
    ) as tmp:
        local_path = Path(tmp.name)

    try:
        t0 = time.perf_counter()
        # Peek at the object size first so we can show download progress.
        # A 403/404 here means Polygon hasn't published the file yet
        # (flat files land by ~11 AM ET the next day). We skip without
        # writing a marker so the downstream pipeline falls back to REST.
        try:
            head = s3_client.head_object(Bucket=FLATFILES_BUCKET, Key=key)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
            if status in (403, 404) or code in ("403", "404", "NoSuchKey"):
                logger.warning(
                    "flatfile %s %s: not available yet on S3 (HTTP %s). "
                    "Skipping; REST will cover this day.",
                    endpoint, day, status,
                )
                return 0, 0
            raise
        total_bytes = int(head.get("ContentLength", 0))
        total_mb = total_bytes / (1024 * 1024)
        logger.info(
            "flatfile %s %s: downloading %.1f MB from s3://%s/%s ...",
            endpoint, day, total_mb, FLATFILES_BUCKET, key,
        )

        # Progress callback: log every ~10% or every 10s, whichever first.
        progress = {"bytes": 0, "last_pct": -1, "last_log": t0}

        def on_progress(n: int) -> None:
            progress["bytes"] += n
            now = time.perf_counter()
            pct = int(100 * progress["bytes"] / max(total_bytes, 1))
            if pct >= progress["last_pct"] + 10 or now - progress["last_log"] >= 10:
                mb_done = progress["bytes"] / (1024 * 1024)
                rate = mb_done / max(now - t0, 0.001)
                logger.info(
                    "flatfile %s %s: %.0f%% (%.1f/%.1f MB, %.1f MB/s)",
                    endpoint, day, pct, mb_done, total_mb, rate,
                )
                progress["last_pct"] = pct
                progress["last_log"] = now

        s3_client.download_file(
            FLATFILES_BUCKET, key, str(local_path), Callback=on_progress
        )
        size_mb = local_path.stat().st_size / (1024 * 1024)
        logger.info(
            "flatfile %s %s: downloaded %.1f MB in %.1fs, filtering...",
            endpoint, day, size_mb, time.perf_counter() - t0,
        )

        # Lazy scan + filter: only rows we want get materialized.
        t1 = time.perf_counter()
        df = (
            pl.scan_csv(local_path, schema_overrides=select)
            .filter(pl.col("ticker").str.starts_with(ticker_prefix))
            .select(list(select.keys()))
            .collect()
        )
        filter_elapsed = time.perf_counter() - t1

        if df.is_empty():
            logger.info(
                "flatfile %s %s: no %s rows after filter (0 contracts)",
                endpoint, day, underlying,
            )
            storage.write_flatfile_marker(endpoint, day, underlying)
            return 0, 0

        # Partition rows by contract ticker and write per-contract parquet.
        t2 = time.perf_counter()
        path_fn = (
            storage.trades_cache_path if endpoint == "trades"
            else storage.quotes_cache_path
        )
        partitions = df.partition_by("ticker", as_dict=True)
        total_rows = 0
        for key_tuple, group in partitions.items():
            # Polars 1.x: partition_by(as_dict=True) returns dict keyed by a
            # tuple of the partition column values.
            ticker = key_tuple[0] if isinstance(key_tuple, tuple) else key_tuple
            per_contract = group.drop("ticker").sort("sip_timestamp")
            per_contract.write_parquet(path_fn(ticker, day))
            total_rows += per_contract.height
        write_elapsed = time.perf_counter() - t2

        logger.info(
            "flatfile %s %s: filter %.1fs, wrote %d contracts / %d rows in %.1fs",
            endpoint, day, filter_elapsed, len(partitions), total_rows, write_elapsed,
        )
        storage.write_flatfile_marker(endpoint, day, underlying)
        return len(partitions), total_rows

    finally:
        # Always delete the raw download, even on error.
        try:
            local_path.unlink(missing_ok=True)
        except OSError:
            pass


def ensure_day(endpoint: str, day: date, underlying: str, s3_client=None) -> None:
    """Public entry: make sure (endpoint, day, underlying) has been processed."""
    if storage.flatfile_marker_exists(endpoint, day, underlying):
        return
    if s3_client is None:
        s3_client = _get_s3_client()
    _process_one_day(endpoint, day, underlying, s3_client)


def default_endpoints() -> tuple[str, ...]:
    """Endpoints needed for the current classification mode.

    Tick rule needs only trades. Quote rule needs both.
    """
    if CLASSIFICATION_MODE == "tick":
        return ("trades",)
    return ("trades", "quotes")


def ensure_range(
    days: list[date],
    underlying: str,
    endpoints: tuple[str, ...] | None = None,
    concurrency: int = FLATFILES_CONCURRENCY,
) -> None:
    """Process a whole list of trading days concurrently. Skips already marked.

    Uses a ThreadPoolExecutor of `concurrency` workers; each worker owns one
    gzipped CSV on disk at a time. Peak disk usage ≈ concurrency × file size.
    boto3 S3 clients are thread-safe so a single client is shared across
    workers. Logs "[n/total] endpoint day" in completion order (not start
    order), so timestamps reflect real progress.
    """
    if endpoints is None:
        endpoints = default_endpoints()
    pending = [
        (ep, d)
        for d in days
        for ep in endpoints
        if not storage.flatfile_marker_exists(ep, d, underlying)
    ]
    total_pairs = len(days) * len(endpoints)
    if not pending:
        logger.info("flatfile: all %d (endpoint, day) pairs already cached", total_pairs)
        return

    s3 = _get_s3_client()
    logger.info(
        "flatfile: %d/%d pairs already cached, processing %d remaining "
        "(concurrency=%d)...",
        total_pairs - len(pending), total_pairs, len(pending), concurrency,
    )

    def _job(endpoint: str, day: date) -> tuple[str, date]:
        _process_one_day(endpoint, day, underlying, s3)
        return endpoint, day

    completed = 0
    n = len(pending)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_job, ep, d): (ep, d) for ep, d in pending}
        for future in as_completed(futures):
            ep, d = futures[future]
            completed += 1
            try:
                future.result()
            except Exception as e:  # noqa: BLE001 — keep going, one bad day shouldn't kill the whole preprocess
                logger.error(
                    "flatfile [%d/%d] %s %s: FAILED — %s",
                    completed, n, ep, d, e,
                )
                continue
            logger.info(
                "flatfile [%d/%d] done: %s %s", completed, n, ep, d
            )
