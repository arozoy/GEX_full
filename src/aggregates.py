"""Aggregate (OHLC) bars for stocks and options.

Two uses:
- Underlying minute bars → spot price at the snapshot time.
- Option daily bars → detect the contract's first active trading day, so that
  trades/quotes fetches are scoped to its actual lifetime.
"""
from __future__ import annotations

from datetime import date, datetime

import polars as pl

from config import MARKET_TZ
from src import storage
from src.client import PolygonClient


# Polygon requires an "I:" prefix on the aggregates endpoint for index
# tickers. The contracts API accepts the plain name, and the flat-file
# ticker prefix is O:SPX / O:SPXW — so TICKER stays plain ("SPX") and we
# only remap when hitting /v2/aggs for the underlying.
_INDEX_TICKERS: frozenset[str] = frozenset({"SPX", "NDX", "RUT", "DJX", "VIX"})


def _aggregate_ticker_for(ticker: str) -> str:
    """Return the ticker form accepted by Polygon's aggregates endpoint."""
    if ticker.startswith("I:") or ticker.startswith("O:") or ":" in ticker:
        return ticker
    if ticker in _INDEX_TICKERS:
        return f"I:{ticker}"
    return ticker

_OPTION_DAILY_SCHEMA = {
    "t": pl.Int64,      # unix ms (bar start)
    "o": pl.Float64,
    "h": pl.Float64,
    "l": pl.Float64,
    "c": pl.Float64,
    "v": pl.Float64,
    "n": pl.Int64,
    "vw": pl.Float64,
}

_STOCK_MINUTE_SCHEMA = {
    "t": pl.Int64,
    "o": pl.Float64,
    "h": pl.Float64,
    "l": pl.Float64,
    "c": pl.Float64,
    "v": pl.Float64,
    "n": pl.Int64,
    "vw": pl.Float64,
}


async def _fetch_bars(
    client: PolygonClient,
    ticker: str,
    multiplier: int,
    timespan: str,
    frm: date,
    to: date,
    schema: dict,
) -> pl.DataFrame:
    path = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{frm.isoformat()}/{to.isoformat()}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000}
    rows: list[dict] = []
    async for row in client.paginate(path, params=params):
        rows.append(row)
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.from_dicts(rows, infer_schema_length=len(rows)).select(
        [pl.col(k).cast(v) for k, v in schema.items() if k in rows[0]]
    )


# ----- Underlying minute bars ---------------------------------------------

async def fetch_underlying_minute_bars(
    client: PolygonClient, ticker: str, day: date
) -> pl.DataFrame:
    """Return 1-minute bars for `ticker` on `day` (regular + extended hours).

    Cached per (ticker, day). For index tickers (SPX, NDX, RUT, ...) the
    Polygon aggregates endpoint expects an `I:` prefix, which is applied
    transparently — the cache key stays the plain ticker.
    """
    cache_path = storage.stock_bars_cache_path(ticker, day)
    cached = storage.read_parquet(cache_path)
    if cached is not None:
        return cached

    agg_ticker = _aggregate_ticker_for(ticker)
    df = await _fetch_bars(client, agg_ticker, 1, "minute", day, day, _STOCK_MINUTE_SCHEMA)
    if not df.is_empty():
        df = df.with_columns(
            pl.from_epoch(pl.col("t"), time_unit="ms")
            .dt.replace_time_zone("UTC")
            .dt.convert_time_zone("America/New_York")
            .alias("ts_et")
        )
    storage.write_parquet(df, cache_path)
    return df


def spot_at(minute_bars: pl.DataFrame, when_et: datetime) -> float | None:
    """Underlying price at (or immediately before) `when_et`.

    Uses the open of the minute bar whose start == `when_et` if present;
    otherwise the close of the latest bar whose start <= `when_et`.
    Returns None if no bar covers that time.
    """
    if minute_bars.is_empty():
        return None

    exact = minute_bars.filter(pl.col("ts_et") == when_et)
    if not exact.is_empty():
        return float(exact["o"][0])

    prior = minute_bars.filter(pl.col("ts_et") <= when_et).sort("ts_et")
    if prior.is_empty():
        return None
    return float(prior["c"][-1])


# ----- Option daily bars (for lifetime detection) --------------------------

async def fetch_option_daily_bars(
    client: PolygonClient,
    option_ticker: str,
    frm: date,
    to: date,
) -> pl.DataFrame:
    """Daily OHLC for an option contract across [frm, to].

    Used to detect the first day the contract actually traded, so we can scope
    tick fetches to its real active lifetime.
    """
    cache_path = storage.option_daily_bars_cache_path(option_ticker)
    cached = storage.read_parquet(cache_path)
    if cached is not None:
        return cached

    df = await _fetch_bars(
        client, option_ticker, 1, "day", frm, to, _OPTION_DAILY_SCHEMA
    )
    if not df.is_empty():
        df = df.with_columns(
            pl.from_epoch(pl.col("t"), time_unit="ms")
            .dt.replace_time_zone("UTC")
            .dt.convert_time_zone("America/New_York")
            .dt.date()
            .alias("session_date")
        )
    storage.write_parquet(df, cache_path)
    return df


def first_active_day(daily_bars: pl.DataFrame) -> date | None:
    """Earliest session with non-zero volume."""
    if daily_bars.is_empty():
        return None
    active = daily_bars.filter(pl.col("v") > 0).sort("session_date")
    if active.is_empty():
        return None
    return active["session_date"][0]
