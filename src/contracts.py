"""Option contracts reference: enumerate 0DTE contracts per expiration date."""
from __future__ import annotations

from datetime import date

import polars as pl

from src import storage
from src.client import PolygonClient

# Schema used for empty-result parquet writes (keeps column types stable).
_CONTRACTS_SCHEMA = {
    "ticker": pl.String,
    "underlying_ticker": pl.String,
    "contract_type": pl.String,  # "call" or "put"
    "strike_price": pl.Float64,
    "expiration_date": pl.Date,
    "exercise_style": pl.String,
    "shares_per_contract": pl.Int64,
    "primary_exchange": pl.String,
}


async def fetch_contracts_expiring_on(
    client: PolygonClient,
    underlying: str,
    expiration: date,
) -> pl.DataFrame:
    """Return all option contracts of `underlying` that expire on `expiration`.

    Uses the cache if present; otherwise calls `/v3/reference/options/contracts`
    twice (expired=true and expired=false) and unions — because by the time we
    query historical dates, matching contracts will be flagged as expired.
    """
    cache_path = storage.contracts_cache_path(underlying, expiration)
    cached = storage.read_parquet(cache_path)
    if cached is not None:
        return cached

    rows: list[dict] = []
    for expired_flag in ("true", "false"):
        params = {
            "underlying_ticker": underlying,
            "expiration_date": expiration.isoformat(),
            "expired": expired_flag,
            "limit": 1000,
        }
        async for row in client.paginate(
            "/v3/reference/options/contracts", params=params
        ):
            rows.append(row)

    if not rows:
        df = pl.DataFrame(schema=_CONTRACTS_SCHEMA)
    else:
        df = (
            pl.from_dicts(rows, infer_schema_length=len(rows))
            .select(
                pl.col("ticker"),
                pl.col("underlying_ticker"),
                pl.col("contract_type"),
                pl.col("strike_price").cast(pl.Float64),
                pl.col("expiration_date").str.to_date(),
                pl.col("exercise_style"),
                pl.col("shares_per_contract").cast(pl.Int64),
                pl.col("primary_exchange"),
            )
            .unique(subset=["ticker"])
            .sort(["contract_type", "strike_price"])
        )

    storage.write_parquet(df, cache_path)
    return df


def filter_by_strike_band(
    contracts: pl.DataFrame, spot: float, band: float
) -> pl.DataFrame:
    """Keep only contracts whose strike is within `spot * (1 ± band)`."""
    lower = spot * (1.0 - band)
    upper = spot * (1.0 + band)
    return contracts.filter(
        (pl.col("strike_price") >= lower) & (pl.col("strike_price") <= upper)
    )
