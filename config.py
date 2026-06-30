"""Backtest configuration. Edit values here to change scope/parameters."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo

# --- Universe ---
# SPY (American, ETF). Same S&P 500 exposure as SPX, but:
#   * index aggregates (I:SPX) require a separate Polygon plan add-on
#   * SPY is the retail-facing side — cleaner GEX signal, more liquid
#   * daily 0DTE since April 2022
# If you later add index-data entitlement, switch to "SPX".
TICKER: str = "SPY"

# --- Date range (inclusive) ---
# 2-year backtest. Effective end date is capped at the latest published
# flat-file day at run-time (see run_backtest.py).
START_DATE: date = date(2021, 4, 1)
END_DATE: date = date(2026, 4, 23)

# --- Contract filter ---
# 0DTE only (expiration == snapshot date).
# Strike filter: keep contracts with strike within this fraction of underlying
# spot at snapshot time.
STRIKE_PCT_BAND: float = 0.30  # ±30%

# --- Snapshot time ---
# GEX is computed this many minutes before market close.
# 30 = classic "end-of-day" (15:30 ET), dominated by pinning/vol-crush
# 180 = mid-afternoon (13:00 ET), far enough from close that dealer
#       hedging effects aren't swamped by close-specific dynamics
MINUTES_BEFORE_CLOSE: int = 180
MARKET_TZ = ZoneInfo("America/New_York")

# --- Risk-free rate ---
# Which treasury maturity to use. Options: 1m, 3m, 6m, 1y, 2y, 3y, 5y, 7y, 10y, 20y, 30y.
TREASURY_TENOR: str = "yield_1_month"

# --- Dividend skip ---
# Skip trading days that fall within this many calendar days of any ex-dividend date.
DIVIDEND_SKIP_WINDOW_DAYS: int = 3  # => 7-day window centered on ex-div

# --- Trade classification mode ---
# "tick" (default): tick rule — classify by comparing each trade's price to
#     the previous trade's. Requires trades only; no lifetime quotes. Fast.
#     See src/trades_quotes.py for the full algorithm.
# "quote": Lee-Ready midpoint rule — requires tick quotes for the entire
#     lifetime of every contract. More accurate (~15% better agreement with
#     true customer direction) but quote flat-files are 10–50× larger than
#     trades, making lifetime accumulation prohibitively slow.
CLASSIFICATION_MODE: str = "tick"


# --- Flat Files (S3) ---
# Number of (endpoint, day) flat-files downloaded + filtered concurrently.
# Each concurrent job holds one gzipped CSV on disk (~50–150 MB for trades,
# ~500 MB–5 GB for quotes). 4 is a reasonable default for most machines;
# drop to 1–2 if disk-bound, raise to 6–8 on fast connections + SSD.
FLATFILES_CONCURRENCY: int = 4
# S3-compatible bulk dumps. Each options trades/quotes file covers a full
# trading day across every US-listed option contract. We download one file
# per (day, endpoint), filter it to just `TICKER`, partition per contract
# into the existing parquet cache, then delete the raw file.
FLATFILES_ENDPOINT: str = "https://files.polygon.io"
FLATFILES_BUCKET: str = "flatfiles"
FLATFILES_TRADES_PREFIX: str = "us_options_opra/trades_v1"
FLATFILES_QUOTES_PREFIX: str = "us_options_opra/quotes_v1"

# --- HTTP client ---
API_BASE_URL: str = "https://api.polygon.io"
# Sane concurrency. Tested at 200 → stalls under macOS ephemeral-port
# exhaustion / Polygon backpressure when fan-out is large. 100 is the
# sweet spot for this workload.
MAX_CONCURRENT_REQUESTS: int = 100
# Cap on CONTRACTS processed in parallel within one snapshot day. Each
# contract fans out ~44 HTTP requests (22 days × trades+quotes), so
# concurrent_contracts × 44 should stay within 2–3× MAX_CONCURRENT_REQUESTS.
# 10 gives each active contract a healthy share of the HTTP pool.
MAX_CONCURRENT_CONTRACTS: int = 10
# Fine-grained timeouts. `read` is the main one — it's how long we wait
# for Polygon to send the response body. Too short → we hang up on slow
# endpoints (they log 499). Too long → a truly stuck request drags.
HTTP_CONNECT_TIMEOUT_SEC: float = 10.0
HTTP_READ_TIMEOUT_SEC: float = 45.0
HTTP_WRITE_TIMEOUT_SEC: float = 10.0
HTTP_POOL_TIMEOUT_SEC: float = 60.0
HTTP_MAX_RETRIES: int = 3
# Per-contract overall timeout: accumulate_mm_position must complete in
# this many seconds or the whole contract is abandoned (logged + skipped).
CONTRACT_TIMEOUT_SEC: float = 180.0

# --- QuantLib / pricing ---
# IV solver bounds.
IV_MIN: float = 0.01
IV_MAX: float = 5.00
IV_ACCURACY: float = 1e-4
IV_MAX_ITERATIONS: int = 100

# Minimum bid/ask quality for IV inversion; contracts failing this get dropped.
MIN_BID: float = 0.01
MIN_ASK: float = 0.01

# --- Paths ---
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
CACHE_DIR: Path = DATA_DIR / "cache"
OUTPUT_DIR: Path = DATA_DIR / "output"

DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
