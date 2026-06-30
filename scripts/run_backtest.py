"""GEX backtest entry point.

For each trading day D in [START_DATE, END_DATE]:
    1. Skip if D is within the dividend-exclusion window.
    2. Fetch 1-minute underlying bars and determine spot at snapshot time.
    3. List all option contracts expiring on D (0DTE); filter ±STRIKE_PCT_BAND
       around spot.
    4. For each surviving contract, detect its first active trading day via
       daily option aggregates, then pull tick trades + quotes for every day
       in [first_active_day, D]. Classify via the quote rule and sum mm_delta
       to obtain the MM net position.
    5. Skip contracts with zero MM position (per spec 6).
    6. At snapshot time, compute option mid from quotes; look up 1-month
       treasury yield; solve BS IV and gamma.
    7. Accumulate per-contract GEX.

Writes:
    data/output/gex_daily.parquet  — 1 row per day
    data/output/gex_detail.parquet — 1 row per (day, contract)
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import logging
# import resource
import time
from datetime import date, timedelta

import polars as pl


# def _raise_fd_limit(target: int = 4096) -> None:
#     """Raise the process file-descriptor soft limit for high concurrency.

#     macOS defaults to 256 soft / 10240 hard. We need ~2× MAX_CONCURRENT_REQUESTS
#     plus headroom for cache parquet files, stdio, etc.
#     """
#     soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
#     desired = max(soft, min(target, hard))
#     if desired > soft:
#         resource.setrlimit(resource.RLIMIT_NOFILE, (desired, hard))


# _raise_fd_limit(4096)

from config import (
    CLASSIFICATION_MODE,
    CONTRACT_TIMEOUT_SEC,
    END_DATE,
    MAX_CONCURRENT_CONTRACTS,
    START_DATE,
    STRIKE_PCT_BAND,
    TICKER,
)
from src import storage
from src.aggregates import (
    fetch_option_daily_bars,
    fetch_underlying_minute_bars,
    first_active_day,
    spot_at,
)
from src.calendar_utils import (
    latest_published_flatfile_day,
    list_trading_days,
    option_expiration_datetime_et,
    snapshot_time_et,
    time_to_expiry_years,
)
from src.client import ApiError, PolygonClient, ServerError
from src import flatfiles
from src.contracts import fetch_contracts_expiring_on, filter_by_strike_band
from src.dividends import fetch_dividends, skip_dates
from src.gex import aggregate_daily, compute_contract_gex
from src.iv_gamma import compute_iv_gamma
from src.snapshot import (
    fetch_treasury_yields,
    risk_free_rate,
)
from src.trades_quotes import (
    classify_trades,
    fetch_quotes,
    fetch_snapshot_quote,
    fetch_trades,
    sum_mm_position,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
# Silence per-request httpx noise; our own logger still prints day-level progress.
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("gex")


# ---------------------------------------------------------------------------
# Per-contract pipeline: discover lifetime, pull ticks, compute MM position.
# ---------------------------------------------------------------------------

async def accumulate_mm_position(
    client: PolygonClient,
    option_ticker: str,
    expiration: date,
    lookback_start: date,
) -> tuple[float, int]:
    """Lifetime MM net position for a contract expiring on `expiration`.

    Steps: detect first active trading day via daily bars → enumerate NYSE
    sessions from that day through expiration → for each session fetch
    trades (and quotes only if classification requires them) → classify
    and sum.

    Under `CLASSIFICATION_MODE == "tick"` we skip the per-session quote
    fetch entirely — the tick rule needs only the trade-price sequence.
    Lifetime quote flat-files are 10-50× bigger than trades and this is
    the main reason "tick" is the default mode.

    Returns (mm_position, n_session_days).
    """
    daily = await fetch_option_daily_bars(
        client, option_ticker, lookback_start, expiration
    )
    start = first_active_day(daily)
    if start is None:
        return 0.0, 0

    sessions = list_trading_days(start, expiration)
    if not sessions:
        return 0.0, 0

    need_quotes = CLASSIFICATION_MODE == "quote"

    async def one_day(day: date) -> float:
        if need_quotes:
            trades, quotes = await asyncio.gather(
                fetch_trades(client, option_ticker, day),
                fetch_quotes(client, option_ticker, day),
            )
            return sum_mm_position(classify_trades(trades, quotes))
        trades = await fetch_trades(client, option_ticker, day)
        return sum_mm_position(classify_trades(trades))

    per_day = await asyncio.gather(*(one_day(d) for d in sessions))
    return float(sum(per_day)), len(sessions)


async def snapshot_option_bid_ask(
    client: PolygonClient,
    option_ticker: str,
    snapshot_day: date,
    snapshot_et,
) -> tuple[float, float] | None:
    """Option (bid, ask) at the snapshot instant.

    Uses a single targeted REST call (`timestamp.lte=<snapshot_ns>`,
    `order=desc`, `limit=50`) to avoid pulling the whole day of quotes
    for ATM contracts (millions of rows). Falls back to the last valid
    quote at or before the snapshot time within the same trading day.
    """
    snapshot_ns = int(snapshot_et.timestamp() * 1_000_000_000)
    return await fetch_snapshot_quote(
        client, option_ticker, snapshot_day, snapshot_ns
    )


# ---------------------------------------------------------------------------
# Per-day driver.
# ---------------------------------------------------------------------------

async def process_day(
    client: PolygonClient, underlying: str, day: date, yields: pl.DataFrame
) -> list[dict]:
    """Produce a per-contract detail list for a single trading day."""
    day_t0 = time.perf_counter()
    weekday = day.strftime("%a")
    logger.info("=== %s (%s): starting ===", day, weekday)

    # 1) Snapshot time on this session (handles early-close days).
    snapshot_et = snapshot_time_et(day)
    expiry_et = option_expiration_datetime_et(day)
    T = time_to_expiry_years(snapshot_et, expiry_et)

    # 2) Underlying spot at snapshot.
    minute_bars = await fetch_underlying_minute_bars(client, underlying, day)
    spot = spot_at(minute_bars, snapshot_et)
    if spot is None:
        logger.warning("%s: no underlying bars at snapshot; skipping day", day)
        return []

    # 3) 0DTE contracts ±band.
    all_contracts = await fetch_contracts_expiring_on(client, underlying, day)
    contracts = filter_by_strike_band(all_contracts, spot, STRIKE_PCT_BAND)
    n_calls = int(contracts.filter(pl.col("contract_type") == "call").height)
    n_puts = int(contracts.filter(pl.col("contract_type") == "put").height)
    logger.info(
        "%s: spot=$%.2f, %d 0DTE contracts in ±%.0f%% band (%d calls, %d puts)",
        day, spot, contracts.height, STRIKE_PCT_BAND * 100, n_calls, n_puts,
    )
    if contracts.is_empty():
        return []

    # 4) Risk-free rate.
    r = risk_free_rate(yields, day)
    if r is None:
        logger.warning("%s: no treasury yield; skipping day", day)
        return []
    logger.info("%s: r (1-mo treasury) = %.4f%%", day, r * 100)

    # 5) Per-contract: lifetime MM position → snapshot mid → IV/Γ.
    #    We run contracts concurrently; the HTTP client's semaphore caps
    #    total in-flight requests. A heartbeat task prints status every
    #    HEARTBEAT_SEC so a stuck contract is visible.
    lookback_start = _months_before(day, months=3)
    total = contracts.height
    progress_counter = {"started": 0, "priced": 0, "skipped": 0, "completed": 0}
    in_flight: dict[str, tuple[float, str]] = {}  # contract_ticker -> (start_ts, stage)

    # Cap concurrent CONTRACT tasks so each running contract has enough of the
    # HTTP pool to make fast progress on its ~44 lifetime-day requests.
    contract_sem = asyncio.Semaphore(MAX_CONCURRENT_CONTRACTS)

    async def one_contract(row: dict) -> dict | None:
        option_ticker = row["ticker"]
        kind = row["contract_type"]  # "call" | "put"
        strike = float(row["strike_price"])

        async with contract_sem:
            progress_counter["started"] += 1
            idx = progress_counter["started"]
            c_t0 = time.perf_counter()
            in_flight[option_ticker] = (c_t0, "fetching lifetime")
            logger.info(
                "%s [%d/%d] %s K=$%.2f: fetching lifetime trades/quotes...",
                day, idx, total, kind, strike,
            )

            try:
                return await asyncio.wait_for(
                    _one_contract_body(
                        row, option_ticker, kind, strike, idx, c_t0,
                    ),
                    timeout=CONTRACT_TIMEOUT_SEC,
                )
            except asyncio.TimeoutError:
                progress_counter["skipped"] += 1
                progress_counter["completed"] += 1
                in_flight.pop(option_ticker, None)
                logger.warning(
                    "%s [%d/%d] %s K=$%.2f: TIMEOUT after %.0fs, skipping",
                    day, idx, total, kind, strike, CONTRACT_TIMEOUT_SEC,
                )
                return None
            except (ApiError, ServerError) as e:
                progress_counter["skipped"] += 1
                progress_counter["completed"] += 1
                in_flight.pop(option_ticker, None)
                logger.warning(
                    "%s [%d/%d] %s K=$%.2f: API error, skipping: %s",
                    day, idx, total, kind, strike, str(e)[:200],
                )
                return None

    async def _one_contract_body(
        row: dict, option_ticker: str, kind: str,
        strike: float, idx: int, c_t0: float,
    ) -> dict | None:
        try:
            mm_position, n_days = await accumulate_mm_position(
                client, option_ticker, day, lookback_start
            )
            fetch_elapsed = time.perf_counter() - c_t0

            if mm_position == 0.0:
                progress_counter["skipped"] += 1
                logger.info(
                    "%s [%d/%d] %s K=$%.2f: done in %.1fs (%d sessions) — "
                    "MM position 0, skipping",
                    day, idx, total, kind, strike, fetch_elapsed, n_days,
                )
                return None

            in_flight[option_ticker] = (c_t0, "snapshot quote")
            bid_ask = await snapshot_option_bid_ask(client, option_ticker, day, snapshot_et)
            if bid_ask is None:
                progress_counter["skipped"] += 1
                logger.info(
                    "%s [%d/%d] %s K=$%.2f: no valid quote at snapshot, skipping",
                    day, idx, total, kind, strike,
                )
                return None

            bid, ask = bid_ask
            mid = (bid + ask) / 2.0
            in_flight[option_ticker] = (c_t0, "solving IV")
            flag = "C" if kind == "call" else "P"
            result = compute_iv_gamma(bid, ask, spot, strike, T, r, 0.0, flag)
            if result.sigma is None:
                progress_counter["skipped"] += 1
                logger.info(
                    "%s [%d/%d] %s K=$%.2f: IV solver failed, skipping",
                    day, idx, total, kind, strike,
                )
                return None

            progress_counter["priced"] += 1
            total_elapsed = time.perf_counter() - c_t0
            gex = (
                mm_position * result.gamma
                * int(row["shares_per_contract"] or 100) * spot ** 2
            )
            logger.info(
                "%s [%d/%d] %s K=$%.2f: done in %.1fs — "
                "mm=%+d, mid=$%.3f, IV=%.1f%%, Γ=%.5f, GEX=%+.3g",
                day, idx, total, kind, strike, total_elapsed,
                int(mm_position), mid, result.sigma * 100, result.gamma, gex,
            )

            return {
                "date": day,
                "contract_ticker": option_ticker,
                "contract_type": kind,
                "strike": strike,
                "expiration": day,
                "shares_per_contract": int(row["shares_per_contract"] or 100),
                "spot": spot,
                "T_years": T,
                "r": r,
                "mid": mid,
                "iv": result.sigma,
                "gamma": result.gamma,
                "mm_position": mm_position,
            }
        finally:
            progress_counter["completed"] += 1
            in_flight.pop(option_ticker, None)

    HEARTBEAT_SEC = 5.0

    async def heartbeat() -> None:
        """Periodic status line so long-running days show liveness."""
        while True:
            await asyncio.sleep(HEARTBEAT_SEC)
            completed = progress_counter["completed"]
            if completed >= total:
                return
            now = time.perf_counter()
            # Show the N oldest in-flight contracts so we can spot stragglers.
            oldest = sorted(in_flight.items(), key=lambda kv: kv[1][0])[:3]
            stragglers = ", ".join(
                f"{t.split(':')[-1]}({stage}, {now - st:.0f}s)"
                for t, (st, stage) in oldest
            ) or "(none)"
            pending = total - completed - len(in_flight)
            logger.info(
                "%s heartbeat: %d/%d done, %d in-flight, %d pending. "
                "Oldest: %s",
                day, completed, total, len(in_flight), pending, stragglers,
            )

    tasks = [one_contract(row) for row in contracts.iter_rows(named=True)]
    hb = asyncio.create_task(heartbeat())
    try:
        results = await asyncio.gather(*tasks)
    finally:
        hb.cancel()
        try:
            await hb
        except asyncio.CancelledError:
            pass
    detail = [r for r in results if r is not None]

    day_elapsed = time.perf_counter() - day_t0
    logger.info(
        "=== %s: complete in %.1fs — %d/%d priced (%d skipped) ===",
        day, day_elapsed, progress_counter["priced"], total,
        progress_counter["skipped"],
    )
    return detail


def _months_before(d: date, months: int) -> date:
    """Approximate date `months` months before `d` (30-day months).

    Used only as a generous lower bound for option daily-bar lookup so we
    catch the contract's first active day.
    """
    return d - timedelta(days=30 * months)


# ---------------------------------------------------------------------------
# Top-level driver.
# ---------------------------------------------------------------------------

async def run_backtest(underlying: str, start: date, end: date) -> None:
    # Clip `end` to the latest fully-published flat-file day so every day
    # in the range is guaranteed to have bulk data available. This avoids
    # the 403 that Polygon returns for days not yet uploaded (~11 AM ET
    # following day). User's configured END_DATE is respected as a ceiling.
    latest_available = latest_published_flatfile_day()
    effective_end = min(end, latest_available)
    if effective_end < end:
        logger.info(
            "Capping END_DATE %s → %s (latest published flat-file day; "
            "later dates would require REST fallback)",
            end, effective_end,
        )
    end = effective_end

    logger.info(
        "Starting backtest: ticker=%s range=%s..%s", underlying, start, end
    )

    async with PolygonClient() as client:
        all_days = list_trading_days(start, end)

        # One-off references (dividends, yield curve) fetched once.
        divs = await fetch_dividends(client, underlying)
        to_skip = skip_dates(divs)
        if to_skip:
            logger.info(
                "Skipping %d dates inside ex-dividend windows", len(to_skip)
            )
        active_days = [d for d in all_days if d not in to_skip]
        logger.info(
            "Processing %d trading days (skipped %d, total %d)",
            len(active_days), len(all_days) - len(active_days), len(all_days),
        )

        yields = await fetch_treasury_yields(client, start, end)

        # ---- Flat-file preprocessing (replaces tick REST calls) -----------
        # For each snapshot day D we need trades for every session from
        # (D - lookback) up to D, to accumulate MM position. Taking the
        # union across all active snapshot days gives the minimum set of
        # flat-file days to download. Under CLASSIFICATION_MODE="tick" we
        # download trades only; quote rule would also download quotes.
        if active_days:
            earliest_session = _months_before(min(active_days), months=3)
            latest_session = max(active_days)
            session_range = list_trading_days(earliest_session, latest_session)
            endpoints = flatfiles.default_endpoints()
            logger.info(
                "Flat files: ensuring %s for %d session days (%s..%s) "
                "[classification_mode=%s]",
                "+".join(endpoints),
                len(session_range), earliest_session, latest_session,
                CLASSIFICATION_MODE,
            )
            await asyncio.to_thread(
                flatfiles.ensure_range, session_range, underlying
            )
            logger.info("Flat files: preprocessing complete")

        all_rows: list[dict] = []
        for day in active_days:
            rows = await process_day(client, underlying, day, yields)
            all_rows.extend(rows)
            logger.info("%s → %d contracts in detail", day, len(rows))

    if not all_rows:
        logger.warning("No detail rows produced; nothing to write.")
        return

    detail = pl.DataFrame(all_rows)
    detail = compute_contract_gex(detail)
    daily = aggregate_daily(detail)

    detail_path = storage.write_output(detail, "gex_detail")
    daily_path = storage.write_output(daily, "gex_daily")
    logger.info("Wrote %s and %s", detail_path, daily_path)

    # Console preview.
    with pl.Config(tbl_rows=50, tbl_cols=10):
        print("\n=== Daily GEX ===")
        print(daily)


def main() -> None:
    asyncio.run(run_backtest(TICKER, START_DATE, END_DATE))


if __name__ == "__main__":
    main()
