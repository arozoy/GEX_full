"""Dividend data — used to skip trading days near ex-dividend.

Per the spec: any trading day within `DIVIDEND_SKIP_WINDOW_DAYS` calendar days
of any ex-dividend date is excluded from the backtest.
"""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from config import DIVIDEND_SKIP_WINDOW_DAYS
from src import storage
from src.client import PolygonClient

_DIVIDENDS_SCHEMA = {
    "ticker": pl.String,
    "ex_dividend_date": pl.Date,
    "cash_amount": pl.Float64,
    "frequency": pl.Int64,
}


async def fetch_dividends(client: PolygonClient, ticker: str) -> pl.DataFrame:
    """Full dividend history for `ticker`. Cached once per ticker."""
    cache_path = storage.dividends_cache_path(ticker)
    cached = storage.read_parquet(cache_path)
    if cached is not None:
        return cached

    rows: list[dict] = []
    async for row in client.paginate(
        "/stocks/v1/dividends",
        params={"ticker": ticker, "limit": 5000},
    ):
        rows.append(row)

    if not rows:
        df = pl.DataFrame(schema=_DIVIDENDS_SCHEMA)
    else:
        df = pl.from_dicts(rows, infer_schema_length=len(rows)).select(
            pl.col("ticker"),
            pl.col("ex_dividend_date").str.to_date(),
            pl.col("cash_amount").cast(pl.Float64),
            pl.col("frequency").cast(pl.Int64),
        )

    storage.write_parquet(df, cache_path)
    return df


def skip_dates(
    dividends: pl.DataFrame,
    window_days: int = DIVIDEND_SKIP_WINDOW_DAYS,
) -> set[date]:
    """Calendar dates within ±`window_days` of any ex-dividend date."""
    if dividends.is_empty():
        return set()

    out: set[date] = set()
    for ex_date in dividends["ex_dividend_date"].drop_nulls().to_list():
        for offset in range(-window_days, window_days + 1):
            out.add(ex_date + timedelta(days=offset))
    return out
