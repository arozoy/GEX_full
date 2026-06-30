"""Aggregate per-contract rows into daily GEX totals.

Sign convention (per spec):
    GEX_contract = mm_position × gamma × shares_per_contract × S²

where `mm_position` is the signed net MM inventory:
    + long contracts  (MM is long gamma)
    − short contracts (MM is short gamma)

A **negative** total GEX means the MM is net short gamma. A positive total
means the MM is net long gamma.
"""
from __future__ import annotations

import polars as pl


def compute_contract_gex(detail: pl.DataFrame) -> pl.DataFrame:
    """Attach a `gex` column to a per-contract detail frame.

    Expected columns: mm_position, gamma, shares_per_contract, spot.
    """
    return detail.with_columns(
        (
            pl.col("mm_position")
            * pl.col("gamma")
            * pl.col("shares_per_contract")
            * pl.col("spot") ** 2
        ).alias("gex")
    )


def aggregate_daily(detail: pl.DataFrame) -> pl.DataFrame:
    """Collapse per-contract detail into one row per `date`.

    Produces gex_total, gex_calls, gex_puts, plus the spot at snapshot time
    and the number of contracts that contributed.
    """
    if detail.is_empty():
        return pl.DataFrame(
            schema={
                "date": pl.Date,
                "spot": pl.Float64,
                "n_contracts": pl.UInt32,
                "gex_total": pl.Float64,
                "gex_calls": pl.Float64,
                "gex_puts": pl.Float64,
            }
        )

    return (
        detail.group_by("date")
        .agg(
            pl.col("spot").first().alias("spot"),
            pl.len().alias("n_contracts"),
            pl.col("gex").sum().alias("gex_total"),
            pl.col("gex")
            .filter(pl.col("contract_type") == "call")
            .sum()
            .alias("gex_calls"),
            pl.col("gex")
            .filter(pl.col("contract_type") == "put")
            .sum()
            .alias("gex_puts"),
        )
        .sort("date")
    )
