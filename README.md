# GEX — 0DTE Gamma Exposure Backtest

A research pipeline for measuring **dealer gamma exposure (GEX)** on a single
underlying, snapshot mid-day, and testing whether GEX predicts the
relationship between morning and afternoon intraday returns.

Built around the Polygon.io (rebranded as Massive) options dataset, with
heavy use of S3 **flat files** for tick-trade history and a thin REST layer
for everything else.

---

## What it does

For each trading day `D` in the configured range, the pipeline:

1. **Lists all 0DTE option contracts** of `TICKER` (those expiring on `D`)
2. **Filters** to strikes within ±`STRIKE_PCT_BAND` of the underlying spot
3. For every surviving contract, **accumulates the market-maker net inventory**
   over the contract's lifetime by classifying every trade with the **tick
   rule**:
   - `price > prev_price` (uptick) → buy-initiated → MM sold (`mm_delta = −size`)
   - `price < prev_price` (downtick) → sell-initiated → MM bought (`mm_delta = +size`)
   - `price = prev_price` (zero-tick) → carry the last non-zero direction
4. At a configurable **snapshot time** (`MINUTES_BEFORE_CLOSE` before the
   session close), pulls the latest valid option mid quote and the
   underlying spot
5. Solves **implied volatility** via QuantLib's `BlackCalculator` (European
   Black-Scholes — appropriate for short-dated cash-settled options and a
   close approximation for short-dated American equity options)
6. Computes **per-contract gamma** and aggregates into a daily **GEX total**:

   ```
   GEX = Σ (mm_position × Γ × shares_per_contract × spot²)
   ```

   * `GEX > 0` ⇒ dealers are net long gamma (stabilizing — sell rallies, buy dips)
   * `GEX < 0` ⇒ dealers are net short gamma (destabilizing — buy rallies, sell dips)

7. **Skips weeks** containing an ex-dividend date (configurable window)
8. Writes daily and per-contract Parquet outputs that the analysis script
   joins against open/close prices to compute morning-vs-afternoon return
   correlation conditional on GEX

---

## Repository layout

```
GEX/
├── config.py                 — every tunable lives here
├── requirements.txt          — Python deps
├── .env.example              — template for API + S3 credentials
├── .env                      — *gitignored*; your real credentials go here
├── api_documentation/        — Polygon REST docs we rely on
├── src/
│   ├── client.py             — async HTTP client (auth, retry, pagination)
│   ├── flatfiles.py          — S3 download → filter → partition (parallel)
│   ├── calendar_utils.py     — NYSE sessions, snapshot time, T calc
│   ├── contracts.py          — list 0DTE contracts, ±band filter
│   ├── aggregates.py         — minute / daily OHLC for stocks & options
│   ├── trades_quotes.py      — tick fetch, tick-rule classifier, snapshot quote
│   ├── dividends.py          — dividend-window skip dates
│   ├── snapshot.py           — treasury yields, snapshot-quote helpers
│   ├── greeks.py             — QuantLib IV inversion + gamma
│   ├── gex.py                — per-contract GEX → daily aggregation
│   └── storage.py            — Parquet cache + marker file helpers
├── scripts/
│   ├── run_backtest.py       — main entry — run the GEX pipeline
│   └── analyze_momentum.py   — load output, compute correlations + plot
└── data/                     — *gitignored*
    ├── cache/                — every API response cached as Parquet
    │   ├── flatfile_done/    — per-(endpoint, ticker, day) marker files
    │   ├── trades/           — per-(contract, day) tick-trade parquets
    │   ├── snapshot_quotes/  — single mid quote at snapshot time
    │   ├── option_daily_bars/ — for first-active-day detection
    │   ├── stock_bars/       — underlying minute bars
    │   ├── contracts/        — option chains by (underlying, expiry)
    │   ├── dividends/        — full dividend history per ticker
    │   └── treasury_yields/  — full yield curve series
    └── output/
        ├── gex_daily.parquet     — one row per snapshot day
        ├── gex_detail.parquet    — one row per (day, contract)
        ├── momentum_table.parquet — analysis-ready joined table
        └── momentum_scatter.png  — visualization
```

---

## Setup

### Prereqs

- macOS or Linux
- Python 3.11+
- A Polygon.io subscription that includes **Options Trades + Quotes** and
  **Flat Files** access (Options Business plan). Index aggregates (e.g.
  `I:SPX`) require a separate addon; without it, this code only works on
  stocks/ETFs.

### Install

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### Credentials

Copy `.env.example` → `.env` and fill in:

```
MASSIVE_API_KEY=<your REST API key>
MASSIVE_S3_ACCESS_KEY=<from polygon.io/dashboard/flat-files>
MASSIVE_S3_SECRET_KEY=<from polygon.io/dashboard/flat-files>
```

`.env` is gitignored.

---

## Configuration

All knobs live in [`config.py`](config.py). The most-edited ones:

| Setting | Default | What it controls |
|---|---|---|
| `TICKER` | `"SPY"` | Underlying. Tested with SPY, GOOGL. SPX needs index-data entitlement. |
| `START_DATE` / `END_DATE` | 2024-01-01 / 2026-04-23 | Backtest range. End is auto-capped to the latest published flat-file day. |
| `STRIKE_PCT_BAND` | `0.30` | Drop contracts beyond ±30% of spot |
| `MINUTES_BEFORE_CLOSE` | `180` | Snapshot taken this many minutes before session close. 30 = "EOD", 180 = "13:00 ET". |
| `CLASSIFICATION_MODE` | `"tick"` | `"tick"` (uses trades only) or `"quote"` (Lee-Ready, requires lifetime quote flat files — ~10× more data) |
| `MAX_CONCURRENT_REQUESTS` | `100` | Async HTTP semaphore cap |
| `MAX_CONCURRENT_CONTRACTS` | `10` | Contracts processed in parallel within one snapshot day |
| `FLATFILES_CONCURRENCY` | `4` | Parallel flat-file downloads. Peak disk ≈ 4 × file size. |
| `DIVIDEND_SKIP_WINDOW_DAYS` | `3` | Skip ±N calendar days around any ex-dividend date |
| `TREASURY_TENOR` | `"yield_1_month"` | Risk-free rate used for IV inversion |

---

## How to run

### Full backtest

```bash
.venv/bin/python -m scripts.run_backtest 2>&1 | tee /tmp/run.log
```

What happens:

1. **Flat-file preprocessing** (one-time per ticker, parallel 4): downloads
   the full-market OPRA tick-trade CSVs for every session day in the
   lookback range, filters to `O:<TICKER>...` rows, partitions per
   contract, writes Parquet, deletes the raw CSV. Marker files prevent
   re-downloading on restart. **Roughly 30–90 min for 2 years of data.**
2. **Per-day GEX pipeline**: for each snapshot day, lists 0DTE contracts,
   accumulates MM position from cached trades, fetches a targeted
   snapshot quote per contract (REST, single tiny call), inverts IV,
   computes Γ, sums to daily GEX. **Roughly 5–30 min for 2 years.**

You can Ctrl-C at any time — partially-completed contract-day work isn't
cached, but everything else is. Re-running picks up from cache.

### Analysis

After a successful backtest:

```bash
.venv/bin/python -m scripts.analyze_momentum
```

Computes the morning-vs-afternoon return correlation overall, conditional
on GEX sign, and conditional on GEX magnitude (top/bottom quartile).
Also runs an interaction OLS:

```
afternoon_ret = α + β₁·morning_ret + β₂·(morning_ret × normalized_GEX) + ε
```

and writes `data/output/momentum_scatter.png` — a two-panel plot showing
the points colored by GEX with per-regime trend lines.

### Switching tickers

Edit `TICKER` in `config.py`. The flat-file markers and per-contract
caches are automatically scoped per ticker, so switching back and forth
is safe.

### Switching snapshot times

Change `MINUTES_BEFORE_CLOSE`. You only need to invalidate the snapshot-
quote cache (everything else is snapshot-time-independent):

```bash
find data/cache/snapshot_quotes -type d -name 'O_<TICKER>*' -prune -exec rm -rf {} +
.venv/bin/python -m scripts.run_backtest
```

---

## Methodology notes & decisions

These are the choices we made; document them when you write up results.

- **Tick rule, not Lee-Ready quote rule.** Quote-rule classification needs
  lifetime quote flat-files (~10–50× larger than trades, 5+ hour
  preprocess). Tick rule agrees with quote rule on ~85% of trades and
  needs trades only. See `src/trades_quotes.py` for both implementations.
- **MM perspective sign convention.** `GEX > 0` ⇒ dealers net long gamma
  (stabilizing). `GEX < 0` ⇒ dealers net short (amplifying). Some public
  GEX dashboards invert this — be careful when comparing.
- **0DTE only.** We accumulate MM position over each 0DTE contract's
  lifetime (typically 5–60 trading days). Aggregated-across-expirations
  GEX is a different signal we don't compute here.
- **European pricing for 0DTE.** At ≤30 min to expiry on a non-dividend-
  paying interval (we skip ex-div weeks), American–European premium gap
  is sub-penny. We use QuantLib's closed-form `BlackCalculator`.
- **Ex-dividend skip.** Any trading day within ±`DIVIDEND_SKIP_WINDOW_DAYS`
  of an ex-dividend date is dropped from the analysis. SPY's quarterly
  cycle drops ~16–20 days/year; SPX (no dividends) drops nothing.
- **Strike filter at 15:30 spot.** Contracts outside ±30% are excluded
  *after* knowing the snapshot spot. There's a tiny lookforward bias here
  (the filter uses end-of-day spot), but contracts ±30% from spot have
  ~0 gamma so the effect is negligible.

---

## What we've found so far

Tested on **SPY 0DTE, Jan 2024 – Apr 2026 (~366 snapshot days)**:

### At 30-min snapshot (15:30 ET)

```
Bottom-25% GEX:  r = −0.25 (p = 0.017)   — both tails mean-revert; dominated
Top-25% GEX:     r = −0.17 (p = 0.10)      by close-specific pinning effects
```

### At 3-hour snapshot (13:00 ET)

```
GEX < 0 (MM short γ):   r = −0.31 (p < 0.001, n = 127)   — mean reversion
GEX > 0 (MM long γ):    r = +0.26 (p < 0.001, n = 239)   — momentum
```

**Interpretation.** Standard dealer-hedging theory predicts the *opposite*
(short-gamma days should amplify moves; long-gamma days should dampen). The
data instead suggests **GEX sign correlates with market regime**:

- Negative-GEX days = customers buying options = fear / event days → morning
  overreacts, afternoon corrects (mean reversion)
- Positive-GEX days = customers selling premium = calm / bullish-trending
  days → morning trend persists into afternoon (momentum)

That is, the cross-sectional relationship between GEX and intraday return
structure is real and significant, but it's driven by *what regime*
correlates with each GEX sign — not by the mechanical hedging flow itself.

The 30-min window obscures this because end-of-day pinning and vol-crush
dominate the final half hour regardless of GEX. The 3-hour window cuts out
that noise and exposes the regime signal cleanly.

### Caveats

- **Direction is opposite to published GEX-momentum literature.** Treat as
  a working hypothesis until validated out-of-sample.
- **0DTE-only.** Aggregated-across-expiry GEX (à la SqueezeMetrics) may
  behave differently.
- **No transaction-cost or slippage modeling.** Any trading strategy
  derived from these correlations needs realistic execution assumptions.
- **No walk-forward validation yet.** Train on 2024, test on 2025–26 to
  check whether the sign survives held-out data.

---

## Adding a new ticker

```python
# config.py
TICKER = "QQQ"
```

then:

```bash
.venv/bin/python -m scripts.run_backtest
```

The flat-file preprocess will download QQQ-relevant trades (re-using any
session-day raw downloads it can — but this code deletes raw CSVs after
filtering, so each ticker is its own download). Markers are per-ticker
under `data/cache/flatfile_done/<endpoint>/<ticker>/`.

For **index** tickers (SPX, NDX, RUT, ...), the underlying minute-bars
endpoint requires the `I:` prefix. The code auto-maps known indices, but
your Polygon plan must include index data — the Options Business tier
alone returns 403 on `I:SPX`.

---

## Background: why GEX?

Market makers who write options to retail investors run delta-hedged books.
A dealer who is long gamma must **sell** as price rises and **buy** as it
falls (to keep delta flat as it changes with spot moves). A dealer short
gamma must do the opposite. In aggregate this hedging flow can dampen or
amplify intraday moves — the GEX number quantifies the size and direction
of that potential flow.

The hypothesis tested here: on days of extreme GEX, the cross-section of
intraday returns should look measurably different from days of mild GEX.
The result is regime-dependent and direction is opposite to the textbook
prediction, but the effect is statistically real on SPY 2024–2026.

---

## Useful one-liners

```bash
# Disk usage by category
du -sh data/cache/* data/output/*

# How many session-days have been preprocessed for the current ticker
ls data/cache/flatfile_done/trades/SPY/ | wc -l

# Inspect daily GEX
.venv/bin/python -c "import polars as pl; print(pl.read_parquet('data/output/gex_daily.parquet'))"

# Re-run analysis without changing anything else
.venv/bin/python -m scripts.analyze_momentum

# Wipe per-contract cache for a ticker (forces fresh flat-file partitioning on next run)
find data/cache/trades -type d -name 'O_SPY*' -prune -exec rm -rf {} +
find data/cache/snapshot_quotes -type d -name 'O_SPY*' -prune -exec rm -rf {} +
rm -rf data/cache/flatfile_done/trades/SPY
```
