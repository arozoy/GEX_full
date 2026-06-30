"""Option trades & quotes: fetch per day, classify initiative direction.

CLASSIFICATION MODE
-------------------
The project supports two trade-classification rules. The active mode is set by
`CLASSIFICATION_MODE` in config.py. Default: "tick".

Quote Rule (Lee-Ready with midpoint) — `classify_trades_quote_rule`
    For each trade, take the midpoint of the contemporaneous bid/ask quote:
        trade_price > mid  → buy-initiated  → MM sold  (mm_delta = -size)
        trade_price < mid  → sell-initiated → MM bought (mm_delta = +size)
        trade_price == mid → dropped (mm_delta = 0)
    Most accurate but requires a full day of tick quotes for every session of
    every contract's lifetime. Flat-file quotes are 10-50× larger than trades
    and downloading them for lifetime history is prohibitive.

Tick Rule — `classify_trades_tick_rule` (ACTIVE MODE)
    Compare each trade's price to the immediately preceding trade's price:
        up-tick       → buy-initiated  (mm_delta = -size)
        down-tick     → sell-initiated (mm_delta = +size)
        zero-tick     → carry the last non-zero tick direction
                        ("zero-up-tick" → buy, "zero-down-tick" → sell)
        first of day  → unclassified (mm_delta = 0, dropped)
    Published benchmarks: ~85% agreement with the quote rule. Cheaper because
    classification needs trades only — no lifetime quotes required. Quotes
    are still fetched at the 15:30 snapshot moment for the IV inversion mid.

SIGN CONVENTION
---------------
A buy-initiated trade means a customer bought `size` contracts, so the MM
sold them → MM position decreases by `size` (mm_delta = -size). A sell-
initiated trade means a customer sold, MM bought → mm_delta = +size. Summed
across the contract's lifetime, this is the MM's net inventory *change* since
the first observed trade (absolute inventory is not knowable from public
data; standard practice is to treat the series as a relative signal).
"""
from __future__ import annotations

from datetime import date

import polars as pl

from config import CLASSIFICATION_MODE, TICKER
from src import storage
from src.client import PolygonClient

_TRADES_SCHEMA = {
    "sip_timestamp": pl.Int64,
    "price": pl.Float64,
    "size": pl.Float64,
    "exchange": pl.Int64,
    "conditions": pl.List(pl.Int64),
}

_QUOTES_SCHEMA = {
    "sip_timestamp": pl.Int64,
    "bid_price": pl.Float64,
    "ask_price": pl.Float64,
    "bid_size": pl.Float64,
    "ask_size": pl.Float64,
    "sequence_number": pl.Int64,
}

_CLASSIFIED_SCHEMA = {
    "sip_timestamp": pl.Int64,
    "price": pl.Float64,
    "size": pl.Float64,
    "mm_delta": pl.Float64,
}


def _select_or_null(rows: list[dict], field: str, dtype: pl.DataType) -> pl.Series:
    """Extract a column from list-of-dicts, filling missing keys with null."""
    return pl.Series(field, [r.get(field) for r in rows], dtype=dtype)


# ----- Fetch ---------------------------------------------------------------

async def fetch_trades(
    client: PolygonClient, option_ticker: str, day: date
) -> pl.DataFrame:
    """All option trades on `day`. Cached per (contract, day).

    If the flat-file marker for this day is set, the cache is authoritative:
    missing parquet ⇒ this contract had no trades that day, return empty.
    REST is only used if flat-file preprocessing hasn't covered this day.
    """
    cache_path = storage.trades_cache_path(option_ticker, day)
    cached = storage.read_parquet(cache_path)
    if cached is not None:
        return cached

    if storage.flatfile_marker_exists("trades", day, TICKER):
        # Authoritatively no activity this day for this contract.
        return pl.DataFrame(schema=_TRADES_SCHEMA)

    rows: list[dict] = []
    async for row in client.paginate(
        f"/v3/trades/{option_ticker}",
        params={"timestamp": day.isoformat(), "limit": 50000, "order": "asc"},
    ):
        rows.append(row)

    if not rows:
        df = pl.DataFrame(schema=_TRADES_SCHEMA)
    else:
        df = pl.DataFrame(
            {
                "sip_timestamp": _select_or_null(rows, "sip_timestamp", pl.Int64),
                "price": _select_or_null(rows, "price", pl.Float64),
                "size": _select_or_null(rows, "size", pl.Float64),
                "exchange": _select_or_null(rows, "exchange", pl.Int64),
                "conditions": _select_or_null(
                    rows, "conditions", pl.List(pl.Int64)
                ),
            }
        ).sort("sip_timestamp")

    storage.write_parquet(df, cache_path)
    return df


async def fetch_snapshot_quote(
    client: PolygonClient,
    option_ticker: str,
    snapshot_day: date,
    snapshot_ns: int,
    n_probe: int = 50,
) -> tuple[float, float] | None:
    """Get the (bid, ask) of the latest valid quote at or before `snapshot_ns`.

    Makes a single targeted REST call — `timestamp.lte=<snapshot_ns>`,
    `order=desc`, `limit=n_probe` — instead of paginating the whole day.
    Filters to rows with bid > 0 and ask > 0 and returns the most recent one.
    Returns None if no valid quote exists.

    Caches a single-row parquet per (contract, day) so subsequent runs
    skip the network.
    """
    cache_path = storage.snapshot_quote_cache_path(option_ticker, snapshot_day)
    cached = storage.read_parquet(cache_path)
    if cached is not None:
        if cached.is_empty():
            return None
        if "bid" in cached.columns and "ask" in cached.columns:
            return (float(cached["bid"][0]), float(cached["ask"][0]))
        # stale cache with old "mid" schema — fall through to re-fetch

    payload = await client.get(
        f"/v3/quotes/{option_ticker}",
        params={
            "timestamp.lte": str(snapshot_ns),
            "order": "desc",
            "sort": "timestamp",
            "limit": n_probe,
        },
    )
    rows = payload.get("results") or []
    result: tuple[float, float] | None = None
    for row in rows:
        bid = row.get("bid_price")
        ask = row.get("ask_price")
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            result = (float(bid), float(ask))
            break

    # Cache the scalar result (one row, or empty if none found).
    if result is None:
        storage.write_parquet(
            pl.DataFrame(schema={"bid": pl.Float64, "ask": pl.Float64}), cache_path,
        )
    else:
        bid, ask = result
        storage.write_parquet(
            pl.DataFrame({"bid": [bid], "ask": [ask]}), cache_path,
        )
    return result


async def fetch_quotes(
    client: PolygonClient, option_ticker: str, day: date
) -> pl.DataFrame:
    """All option quotes on `day`. Cached per (contract, day).

    Used only for the 15:30 snapshot mid under tick-rule mode. (Under quote-
    rule mode, lifetime quotes would also be fetched here.)
    """
    cache_path = storage.quotes_cache_path(option_ticker, day)
    cached = storage.read_parquet(cache_path)
    if cached is not None:
        return cached

    if storage.flatfile_marker_exists("quotes", day, TICKER):
        return pl.DataFrame(schema=_QUOTES_SCHEMA)

    rows: list[dict] = []
    async for row in client.paginate(
        f"/v3/quotes/{option_ticker}",
        params={"timestamp": day.isoformat(), "limit": 50000, "order": "asc"},
    ):
        rows.append(row)

    if not rows:
        df = pl.DataFrame(schema=_QUOTES_SCHEMA)
    else:
        df = pl.DataFrame(
            {
                "sip_timestamp": _select_or_null(rows, "sip_timestamp", pl.Int64),
                "bid_price": _select_or_null(rows, "bid_price", pl.Float64),
                "ask_price": _select_or_null(rows, "ask_price", pl.Float64),
                "bid_size": _select_or_null(rows, "bid_size", pl.Float64),
                "ask_size": _select_or_null(rows, "ask_size", pl.Float64),
                "sequence_number": _select_or_null(
                    rows, "sequence_number", pl.Int64
                ),
            }
        ).sort("sip_timestamp")

    storage.write_parquet(df, cache_path)
    return df


# ----- Classification ------------------------------------------------------

def classify_trades_quote_rule(
    trades: pl.DataFrame, quotes: pl.DataFrame
) -> pl.DataFrame:
    """Lee-Ready midpoint classification. Requires lifetime quotes.

    Kept for reference / switching back. Not used when CLASSIFICATION_MODE
    is "tick".
    """
    if trades.is_empty() or quotes.is_empty():
        return pl.DataFrame(schema=_CLASSIFIED_SCHEMA)

    valid_quotes = (
        quotes.filter((pl.col("bid_price") > 0) & (pl.col("ask_price") > 0))
        .with_columns(
            ((pl.col("bid_price") + pl.col("ask_price")) / 2.0).alias("mid")
        )
        .select(["sip_timestamp", "mid"])
        .sort("sip_timestamp")
    )
    if valid_quotes.is_empty():
        return pl.DataFrame(schema=_CLASSIFIED_SCHEMA)

    matched = trades.sort("sip_timestamp").join_asof(
        valid_quotes, on="sip_timestamp", strategy="backward",
    )

    classified = matched.with_columns(
        pl.when(pl.col("mid").is_null())
        .then(0.0)
        .when(pl.col("price") > pl.col("mid"))
        .then(-pl.col("size"))        # buy-initiated ⇒ MM sold
        .when(pl.col("price") < pl.col("mid"))
        .then(pl.col("size"))         # sell-initiated ⇒ MM bought
        .otherwise(0.0)               # at-mid ⇒ drop
        .alias("mm_delta")
    )
    return classified.select(["sip_timestamp", "price", "size", "mm_delta"])


def classify_trades_tick_rule(trades: pl.DataFrame) -> pl.DataFrame:
    """Tick rule — classify using only prior-trade prices. No quotes needed.

    For each trade (after sorting by sip_timestamp):
      * price  > prev      → uptick       → buy-initiated  → mm_delta = -size
      * price  < prev      → downtick     → sell-initiated → mm_delta = +size
      * price == prev      → zero-tick    → carry forward last non-zero
                                             direction ("zero-up-tick" = buy,
                                             "zero-down-tick" = sell)
      * very first trade   → unclassified → mm_delta = 0 (dropped)

    Implementation: compute the most recent non-zero price move using a
    fill-forward on the diff series.
    """
    if trades.is_empty():
        return pl.DataFrame(schema=_CLASSIFIED_SCHEMA)

    t = trades.sort("sip_timestamp").with_columns(
        pl.col("price").diff().alias("_diff")
    )
    # Replace zero-ticks with null, then forward-fill so each zero-tick inherits
    # the sign of the last non-zero move. Leading nulls (first trade, plus any
    # opening zero-tick streak) stay null → unclassified.
    t = t.with_columns(
        pl.when(pl.col("_diff") == 0)
        .then(None)
        .otherwise(pl.col("_diff"))
        .forward_fill()
        .alias("_signed_move")
    )
    t = t.with_columns(
        pl.when(pl.col("_signed_move").is_null())
        .then(0.0)
        .when(pl.col("_signed_move") > 0)
        .then(-pl.col("size"))   # uptick → MM sold
        .when(pl.col("_signed_move") < 0)
        .then(pl.col("size"))    # downtick → MM bought
        .otherwise(0.0)
        .alias("mm_delta")
    )
    return t.select(["sip_timestamp", "price", "size", "mm_delta"])


def classify_trades(
    trades: pl.DataFrame, quotes: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Dispatch to the configured classifier.

    `quotes` is accepted (and required) only in "quote" mode; it is ignored
    under "tick" mode. Callers can always pass None in tick mode.
    """
    if CLASSIFICATION_MODE == "tick":
        return classify_trades_tick_rule(trades)
    if CLASSIFICATION_MODE == "quote":
        if quotes is None:
            raise ValueError("quote-rule classification requires quotes")
        return classify_trades_quote_rule(trades, quotes)
    raise ValueError(f"Unknown CLASSIFICATION_MODE: {CLASSIFICATION_MODE!r}")


def sum_mm_position(classified: pl.DataFrame) -> float:
    """Total MM net position contribution from a classified trades frame."""
    if classified.is_empty():
        return 0.0
    total = classified["mm_delta"].sum()
    return float(total) if total is not None else 0.0
