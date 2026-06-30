"""Disk cache layer.

Every API response (contracts, trades, quotes, aggregates, yields, dividends)
is persisted to a Parquet file under `data/cache/`. Subsequent runs read
from disk instead of re-calling the API.

To force a re-fetch: delete the relevant subfolder of `data/cache/`.
"""
from __future__ import annotations

from datetime import date  # noqa: TC003 — used in signatures
from pathlib import Path

import polars as pl

from config import CACHE_DIR, OUTPUT_DIR


# ----- Cache path helpers --------------------------------------------------

def _safe(ticker: str) -> str:
    """Replace characters unsafe in filenames (`:`, `/`)."""
    return ticker.replace(":", "_").replace("/", "_")


def contracts_cache_path(underlying: str, expiration: date) -> Path:
    d = CACHE_DIR / "contracts" / underlying
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{expiration.isoformat()}.parquet"


def trades_cache_path(option_ticker: str, day: date) -> Path:
    d = CACHE_DIR / "trades" / _safe(option_ticker)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{day.isoformat()}.parquet"


def quotes_cache_path(option_ticker: str, day: date) -> Path:
    d = CACHE_DIR / "quotes" / _safe(option_ticker)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{day.isoformat()}.parquet"


def snapshot_quote_cache_path(option_ticker: str, day: date) -> Path:
    """Cache for the single quote-mid at the 15:30 snapshot.

    Distinct from `quotes_cache_path`: that stores a full day of ticks,
    this stores just the one quote we need for pricing.
    """
    d = CACHE_DIR / "snapshot_quotes" / _safe(option_ticker)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{day.isoformat()}.parquet"


def stock_bars_cache_path(ticker: str, day: date) -> Path:
    d = CACHE_DIR / "stock_bars" / ticker
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{day.isoformat()}.parquet"


def option_daily_bars_cache_path(option_ticker: str) -> Path:
    d = CACHE_DIR / "option_daily_bars"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{_safe(option_ticker)}.parquet"


def dividends_cache_path(ticker: str) -> Path:
    d = CACHE_DIR / "dividends"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{ticker}.parquet"


def treasury_yields_cache_path() -> Path:
    d = CACHE_DIR / "treasury_yields"
    d.mkdir(parents=True, exist_ok=True)
    return d / "all.parquet"


# ----- Flat-file markers --------------------------------------------------
# A marker signals "we have filtered+partitioned the full-market flat file
# for this (endpoint, day, underlying)". Scoped by UNDERLYING because the
# partition is ticker-specific — a marker written during a GOOGL run
# doesn't mean SPY partitions exist. If a marker exists, callers know
# that any contract-day parquet absent from the cache means the contract
# simply had no activity that day (not that we haven't fetched yet).

def flatfile_marker_path(endpoint: str, day: date, underlying: str) -> Path:
    d = CACHE_DIR / "flatfile_done" / endpoint / underlying
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{day.isoformat()}.marker"


def flatfile_marker_exists(endpoint: str, day: date, underlying: str) -> bool:
    return flatfile_marker_path(endpoint, day, underlying).exists()


def write_flatfile_marker(endpoint: str, day: date, underlying: str) -> None:
    flatfile_marker_path(endpoint, day, underlying).touch()


# ----- IO helpers ----------------------------------------------------------

def write_parquet(df: pl.DataFrame, path: Path) -> None:
    """Write (overwrite) a Parquet file. Empty frames are written too — so we
    can distinguish 'no data for this key' from 'never fetched'."""
    df.write_parquet(path)


def read_parquet(path: Path) -> pl.DataFrame | None:
    """Return cached DataFrame, or None if cache miss."""
    if not path.exists():
        return None
    return pl.read_parquet(path)


# ----- Output writers ------------------------------------------------------

def write_output(df: pl.DataFrame, name: str) -> Path:
    path = OUTPUT_DIR / f"{name}.parquet"
    df.write_parquet(path)
    return path
