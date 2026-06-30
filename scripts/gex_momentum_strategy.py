"""GEX-driven intraday momentum strategy for SPY.

Trading rule (one trade per day):
    At 13:00 ET (GEX snapshot time):
        If GEX < 0 AND morning return (open → 13:00 spot) bullish → LONG
        If GEX < 0 AND morning return bearish → SHORT
        Otherwise → FLAT

    Exit at 15:55 ET.

GEX sign and morning return both come from the momentum table snapshot at 13:00.
No look-ahead: both are known at the moment of entry.

Run:
    python -m scripts.gex_momentum_strategy
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from datetime import time as dtime

import matplotlib
import numpy as np
import pandas as pd
import polars as pl

matplotlib.use("Agg")
import backtrader as bt
import backtrader.analyzers as btanalyzers

from config import CACHE_DIR, OUTPUT_DIR, TICKER

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("gex_strategy")

# ── Constants ────────────────────────────────────────────────────────────────
ENTRY_TIME       = dtime(13,  0)   # GEX snapshot time — enter immediately after
EXIT_TIME        = dtime(15, 55)   # 5 min before close

MIN_MORNING_MOVE = 0.0001  # skip near-flat mornings (noise filter)


# ── 1. Load minute bars from cache ───────────────────────────────────────────

def load_minute_bars(ticker: str, trading_dates: list) -> pd.DataFrame:
    """Concatenate per-day parquet files; return sorted DataFrame."""
    bars_dir = CACHE_DIR / "stock_bars" / ticker
    dfs = []
    for d in trading_dates:
        p = bars_dir / f"{d.isoformat()}.parquet"
        if not p.exists():
            logger.warning("No minute-bar cache for %s, skipping", d)
            continue
        raw = pl.read_parquet(p)
        if raw.is_empty():
            continue
        raw = raw.filter(
            (pl.col("ts_et").dt.time() >= dtime(9, 30))
            & (pl.col("ts_et").dt.time() <= dtime(16, 0))
        )
        dfs.append(raw.to_pandas())

    if not dfs:
        raise RuntimeError(
            f"No minute-bar cache files found under {bars_dir}. "
            "Run the backtest first to populate the cache."
        )

    combined = pd.concat(dfs, ignore_index=True)
    combined["ts_et"] = (
        pd.to_datetime(combined["ts_et"])
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)          # make naive ET for backtrader
    )
    return combined.sort_values("ts_et").reset_index(drop=True)


# ── 3. Inject signal into minute bars ────────────────────────────────────────

def build_feed_df(
    bars: pd.DataFrame,
    mom: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return minute-bar DataFrame (indexed by ts_et) with a 'signal' column.
    signal is non-NaN only at the 13:00 bar on tradeable days:
        +1  = LONG   (GEX < 0 + bullish morning)
        -1  = SHORT  (GEX < 0 + bearish morning)
        NaN = no trade
    """
    bars = bars.copy()
    bars["date"]   = bars["ts_et"].dt.date
    bars["signal"] = np.nan

    for _, row in mom[["date", "gex_total", "morning_ret"]].iterrows():
        if row["gex_total"] >= 0:
            continue  # only trade when GEX is negative
        if abs(row["morning_ret"]) < MIN_MORNING_MOVE:
            continue

        direction = np.sign(row["morning_ret"])

        mask = (
            (bars["date"] == row["date"])
            & (bars["ts_et"].dt.hour   == ENTRY_TIME.hour)
            & (bars["ts_et"].dt.minute == ENTRY_TIME.minute)
        )
        bars.loc[mask, "signal"] = direction

    bars = bars.rename(columns={"o": "open", "h": "high",
                                 "l": "low",  "c": "close", "v": "volume"})
    bars = bars.set_index("ts_et")
    bars.index = pd.DatetimeIndex(bars.index)
    return bars


# ── 4. Custom PandasData feed ────────────────────────────────────────────────

class GEXMinuteData(bt.feeds.PandasData):
    lines = ("signal",)
    params = (
        ("datetime",     None),
        ("open",         "open"),
        ("high",         "high"),
        ("low",          "low"),
        ("close",        "close"),
        ("volume",       "volume"),
        ("openinterest", -1),
        ("signal",       "signal"),
    )


# ── 5. Strategy ──────────────────────────────────────────────────────────────

class GEXMomentumStrategy(bt.Strategy):
    params = dict(
        size_pct=0.95,
        stop_pct=0.01,   # 1% trailing stop from best price seen since entry
        verbose=True,
    )

    def log(self, msg: str) -> None:
        if self.p.verbose:
            dt = self.datas[0].datetime.datetime(0)
            logger.info("%s  %s", dt.strftime("%Y-%m-%d %H:%M"), msg)

    def __init__(self) -> None:
        self.order         = None
        self.stop_price    = None
        self.trail_extreme = None  # highest high (long) or lowest low (short) since entry

    def next(self) -> None:
        if self.order:
            return

        bar_t  = self.datas[0].datetime.time(0)
        signal = self.data.signal[0]

        # ── Trailing stop update + check ──────────────────────────────────
        if self.position and self.stop_price is not None:
            if self.position.size > 0:
                if self.data.high[0] > self.trail_extreme:
                    self.trail_extreme = self.data.high[0]
                    self.stop_price    = self.trail_extreme * (1 - self.p.stop_pct)
                if self.data.low[0] <= self.stop_price:
                    self.log(f"TRAIL STOP HIT @ {self.stop_price:.2f}  low={self.data.low[0]:.2f}")
                    self.order = self.close()
                    return
            else:
                if self.data.low[0] < self.trail_extreme:
                    self.trail_extreme = self.data.low[0]
                    self.stop_price    = self.trail_extreme * (1 + self.p.stop_pct)
                if self.data.high[0] >= self.stop_price:
                    self.log(f"TRAIL STOP HIT @ {self.stop_price:.2f}  high={self.data.high[0]:.2f}")
                    self.order = self.close()
                    return

        # ── Entry ──────────────────────────────────────────────────────────
        if (
            bar_t.hour   == ENTRY_TIME.hour
            and bar_t.minute == ENTRY_TIME.minute
            and not np.isnan(signal)
            and not self.position
        ):
            price = self.data.open[0]
            size  = int(self.broker.getvalue() * self.p.size_pct / price)
            if size < 1:
                return

            if signal > 0:
                self.log(f"LONG  {size} @ {price:.2f}  signal={signal:+.0f}")
                self.order = self.buy(size=size)
            else:
                self.log(f"SHORT {size} @ {price:.2f}  signal={signal:+.0f}")
                self.order = self.sell(size=size)

        # ── Time-based exit ────────────────────────────────────────────────
        elif (
            bar_t.hour   == EXIT_TIME.hour
            and bar_t.minute == EXIT_TIME.minute
            and self.position
        ):
            self.log(f"CLOSE @ {self.data.close[0]:.2f}")
            self.order = self.close()

    def notify_order(self, order: bt.Order) -> None:
        if order.status == order.Completed:
            if self.position:  # entry just filled — initialise trailing stop
                exec_price         = order.executed.price
                self.trail_extreme = exec_price
                if order.isbuy():
                    self.stop_price = exec_price * (1 - self.p.stop_pct)
                else:
                    self.stop_price = exec_price * (1 + self.p.stop_pct)
                self.log(f"TRAIL STOP INIT @ {self.stop_price:.2f}")
            else:              # exit filled — clear trailing state
                self.stop_price    = None
                self.trail_extreme = None
            self.order = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    def notify_trade(self, trade: bt.Trade) -> None:
        if trade.isclosed:
            self.log(
                f"TRADE  gross={trade.pnl:+.2f}  net={trade.pnlcomm:+.2f}"
            )


# ── 6. Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    # Load and prepare momentum table
    mom_path = OUTPUT_DIR / "momentum_table.parquet"
    if not mom_path.exists():
        raise FileNotFoundError(
            "Run analyze_momentum.py first to generate momentum_table.parquet."
        )
    mom = pl.read_parquet(mom_path).to_pandas()
    mom["date"] = pd.to_datetime(mom["date"]).dt.date
    mom = mom.sort_values("date").reset_index(drop=True)
    logger.info("Loaded %d trading days from momentum_table", len(mom))

    trading_dates = list(mom["date"])
    logger.info("Loading minute bars for %d days…", len(trading_dates))
    bars = load_minute_bars(TICKER, trading_dates)

    feed_df = build_feed_df(bars, mom)
    n_signals = int(feed_df["signal"].notna().sum())
    logger.info(
        "Entry signal present on %d bars  (entry=%s, exit=%s ET)",
        n_signals, ENTRY_TIME.strftime("%H:%M"), EXIT_TIME.strftime("%H:%M"),
    )

    # ── Cerebro setup ──────────────────────────────────────────────────────
    cerebro = bt.Cerebro()
    cerebro.broker.set_cash(100_000.0)
    cerebro.broker.setcommission(commission=0.0005)
    cerebro.broker.set_shortcash(False)
    data_feed = GEXMinuteData(
        dataname=feed_df,
        timeframe=bt.TimeFrame.Minutes,
        compression=1,
    )
    cerebro.adddata(data_feed)
    cerebro.addstrategy(GEXMomentumStrategy)

    cerebro.addanalyzer(
        btanalyzers.SharpeRatio, _name="sharpe",
        riskfreerate=0.05, annualize=True, timeframe=bt.TimeFrame.Days,
    )
    cerebro.addanalyzer(btanalyzers.DrawDown,      _name="drawdown")
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name="trades")

    logger.info("Running backtest — initial capital $100,000")
    results = cerebro.run()
    strat   = results[0]

    # ── Performance report ──────────────────────────────────────────────────
    final_val  = cerebro.broker.getvalue()
    total_ret  = (final_val / 100_000.0 - 1.0) * 100.0
    sharpe     = strat.analyzers.sharpe.get_analysis().get("sharperatio")
    dd_an      = strat.analyzers.drawdown.get_analysis()
    trade_an   = strat.analyzers.trades.get_analysis()

    total_t = trade_an.get("total", {}).get("total", 0)
    won     = trade_an.get("won",   {}).get("total", 0)
    lost    = trade_an.get("lost",  {}).get("total", 0)
    win_pct = won / total_t * 100 if total_t else 0.0
    avg_win  = trade_an.get("won",  {}).get("pnl", {}).get("average", float("nan"))
    avg_loss = trade_an.get("lost", {}).get("pnl", {}).get("average", float("nan"))

    print("\n" + "=" * 60)
    print(f"  GEX Momentum Strategy — {TICKER}")
    print(f"  Entry: {ENTRY_TIME.strftime('%H:%M')} ET  |  "
          f"Exit: {EXIT_TIME.strftime('%H:%M')} ET")
    print(f"  Rule: GEX < 0 → follow morning direction (open → 13:00)")
    print("=" * 60)
    print(f"  Initial capital : $100,000.00")
    print(f"  Final value     : ${final_val:,.2f}")
    print(f"  Total return    : {total_ret:+.2f}%")
    if sharpe is not None:
        print(f"  Sharpe ratio    : {sharpe:.3f}")
    max_dd = dd_an.get("max", {}).get("drawdown", float("nan"))
    print(f"  Max drawdown    : {max_dd:.2f}%")
    print(f"  Trades          : {total_t}  (won {won}, lost {lost})")
    print(f"  Win rate        : {win_pct:.1f}%")
    print(f"  Avg win / loss  : {avg_win:+.2f} / {avg_loss:+.2f}")
    print("=" * 60)

    # Save equity-curve plot
    out_fig = OUTPUT_DIR / "gex_strategy_equity.png"
    try:
        import matplotlib.pyplot as plt
        from backtrader.plot import Plot
        plt.rcParams["figure.max_open_warning"] = 0
        plotter = Plot(style="bar")
        figs = plotter.plot(strat, iplot=False)
        figs[0].savefig(out_fig, dpi=130, bbox_inches="tight")
        plt.close("all")
        logger.info("Equity curve saved → %s", out_fig)
    except Exception as exc:
        logger.warning("Could not save plot (%s) — skipping", exc)


if __name__ == "__main__":
    main()
