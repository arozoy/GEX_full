"""Snapshot inputs at the valuation time: option mid, underlying spot, r.

At `snapshot_time_et` on each trading day we need three numbers per contract
(plus one per day):
    * Option mid      = (bid + ask) / 2 of the latest valid quote <= snapshot
    * Underlying spot = price at the snapshot minute (see aggregates.spot_at)
    * Risk-free rate  = 1-month treasury yield on that calendar day
"""
from __future__ import annotations

from datetime import date, datetime

import polars as pl

from config import MIN_ASK, MIN_BID, TREASURY_TENOR
from src import storage
from src.client import PolygonClient


def option_mid_at_snapshot(
    quotes: pl.DataFrame, snapshot_et: datetime
) -> float | None:
    """Latest valid quote mid at or before `snapshot_et`.

    Filters out quotes with bid<MIN_BID or ask<MIN_ASK. Returns None if no
    valid quote exists before the snapshot time.
    """
    if quotes.is_empty():
        return None

    snapshot_ns = int(snapshot_et.timestamp() * 1_000_000_000)
    valid = quotes.filter(
        (pl.col("sip_timestamp") <= snapshot_ns)
        & (pl.col("bid_price") >= MIN_BID)
        & (pl.col("ask_price") >= MIN_ASK)
    ).sort("sip_timestamp")

    if valid.is_empty():
        return None
    last = valid.tail(1)
    return float((last["bid_price"][0] + last["ask_price"][0]) / 2.0)


# ----- Treasury yields -----------------------------------------------------

_YIELD_FIELDS = [
    "date",
    "yield_1_month",
    "yield_3_month",
    "yield_6_month",
    "yield_1_year",
    "yield_2_year",
    "yield_3_year",
    "yield_5_year",
    "yield_7_year",
    "yield_10_year",
    "yield_20_year",
    "yield_30_year",
]


async def fetch_treasury_yields(
    client: PolygonClient, start: date, end: date
) -> pl.DataFrame:
    """All treasury-yield rows for [start, end]. Cached once per range."""
    cache_path = storage.treasury_yields_cache_path()
    cached = storage.read_parquet(cache_path)
    if cached is not None:
        # Reuse cache if it covers the requested range.
        min_d = cached["date"].min()
        max_d = cached["date"].max()
        if min_d is not None and max_d is not None and min_d <= start and max_d >= end:
            return cached

    rows: list[dict] = []
    async for row in client.paginate(
        "/fed/v1/treasury-yields",
        params={
            "date.gte": start.isoformat(),
            "date.lte": end.isoformat(),
            "limit": 50000,
            "sort": "date.asc",
        },
    ):
        rows.append(row)

    if not rows:
        df = pl.DataFrame(
            schema={k: (pl.Date if k == "date" else pl.Float64) for k in _YIELD_FIELDS}
        )
    else:
        df = pl.from_dicts(rows, infer_schema_length=len(rows))
        # Ensure all expected columns exist (some days may omit long tenors).
        for field in _YIELD_FIELDS:
            if field not in df.columns:
                df = df.with_columns(pl.lit(None).alias(field))
        df = df.select(
            [pl.col("date").str.to_date()]
            + [pl.col(f).cast(pl.Float64) for f in _YIELD_FIELDS if f != "date"]
        ).sort("date")

    storage.write_parquet(df, cache_path)
    return df


def risk_free_rate(
    yields: pl.DataFrame, day: date, tenor_field: str = TREASURY_TENOR
) -> float | None:
    """Yield (decimal, e.g. 0.0525) for `day` at the configured tenor.

    Falls back to the latest prior date if the exact day is missing
    (e.g., weekends or holidays return no yield observation).
    """
    if yields.is_empty():
        return None

    on_or_before = yields.filter(pl.col("date") <= day).sort("date")
    if on_or_before.is_empty():
        return None

    val = on_or_before[tenor_field].drop_nulls()
    if val.is_empty():
        return None
    return float(val[-1]) / 100.0
