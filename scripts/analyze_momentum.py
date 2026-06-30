"""Does GEX predict intraday momentum?

HYPOTHESIS (standard dealer-hedging story):
    GEX < 0   (MM net SHORT gamma, amplifies moves)
              → morning & afternoon returns have SAME sign (momentum)
              → correlation is POSITIVE
    GEX > 0   (MM net LONG gamma, dampens moves)
              → morning & afternoon returns have OPPOSITE signs (mean rev)
              → correlation is NEGATIVE

Variables (one row per trading day D):
    morning_ret   = (spot_1530 − open) / open
    afternoon_ret = (close − spot_1530) / spot_1530
    gex_total     = dollar-gamma sum at 15:30 (sign: + = MM long gamma)

Outputs:
    stdout summary table + statistics
    data/output/momentum_scatter.png   — scatter colored by GEX w/ trendlines
    data/output/momentum_table.parquet — merged per-day table

Run:
    .venv/bin/python -m scripts.analyze_momentum
"""
from __future__ import annotations

import logging
import sys
from datetime import date, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats

from config import OUTPUT_DIR, TICKER
from src import storage

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("analyze")


# ----- Data loading --------------------------------------------------------

def _stock_open_close(ticker: str, day: date) -> tuple[float | None, float | None]:
    """Return (open, close) for `ticker` on `day` from cached minute bars.

    Open = first minute bar's open after 09:30 ET.
    Close = last minute bar's close at or before 16:00 ET.
    """
    path = storage.stock_bars_cache_path(ticker, day)
    df = storage.read_parquet(path)
    if df is None or df.is_empty():
        return None, None

    # Filter to regular-hours bars (9:30 ET open → 16:00 ET close). Compare
    # via `.dt.time()` rather than hour*60+minute arithmetic because
    # `.dt.hour()` returns Int8, which overflows on the × 60.
    ny = df.filter(
        (pl.col("ts_et").dt.time() >= time(9, 30))
        & (pl.col("ts_et").dt.time() <= time(16, 0))
    ).sort("ts_et")
    if ny.is_empty():
        return None, None
    return float(ny["o"][0]), float(ny["c"][-1])


def build_dataset() -> pl.DataFrame:
    """Join daily GEX with open/close to get momentum variables."""
    gex = storage.read_parquet(OUTPUT_DIR / "gex_daily.parquet")
    if gex is None or gex.is_empty():
        raise RuntimeError(
            "data/output/gex_daily.parquet not found. Run the backtest first."
        )

    rows: list[dict] = []
    for row in gex.iter_rows(named=True):
        d = row["date"]
        open_px, close_px = _stock_open_close(TICKER, d)
        if open_px is None or close_px is None:
            logger.warning("%s: missing stock bars, skipping", d)
            continue

        spot_1530 = row["spot"]
        morning_ret = (spot_1530 - open_px) / open_px
        afternoon_ret = (close_px - spot_1530) / spot_1530
        rows.append({
            "date": d,
            "open": open_px,
            "spot_1530": spot_1530,
            "close": close_px,
            "morning_ret": morning_ret,
            "afternoon_ret": afternoon_ret,
            "gex_total": row["gex_total"],
            "gex_calls": row["gex_calls"],
            "gex_puts": row["gex_puts"],
            "n_contracts": row["n_contracts"],
        })

    return pl.DataFrame(rows).sort("date")


# ----- Statistics ---------------------------------------------------------

def _corr_with_p(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """Pearson r + two-sided p-value + sample size. Skips NaNs pairwise."""
    mask = np.isfinite(x) & np.isfinite(y)
    xc, yc = x[mask], y[mask]
    if len(xc) < 3:
        return float("nan"), float("nan"), len(xc)
    r, p = stats.pearsonr(xc, yc)
    return float(r), float(p), len(xc)


def _ols_with_tstats(
    X: np.ndarray, y: np.ndarray, names: list[str]
) -> list[dict]:
    """Plain OLS with coefficient t-stats and p-values.

    Returns one dict per coefficient: {name, beta, se, t, p}.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma2 = (resid @ resid) / max(n - k, 1)
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    tstat = beta / se
    pvals = 2 * (1 - stats.t.cdf(np.abs(tstat), df=max(n - k, 1)))
    return [
        {"name": n_, "beta": float(b), "se": float(s),
         "t": float(t), "p": float(p)}
        for n_, b, s, t, p in zip(names, beta, se, tstat, pvals)
    ]


# ----- Reporting ----------------------------------------------------------

def print_report(df: pl.DataFrame) -> None:
    n = df.height
    mr = df["morning_ret"].to_numpy()
    ar = df["afternoon_ret"].to_numpy()
    gex = df["gex_total"].to_numpy()

    print("\n" + "=" * 72)
    print(f"GEX momentum analysis — {TICKER}, N = {n} trading days")
    print("=" * 72)

    # Per-day table
    with pl.Config(tbl_rows=50, tbl_cols=10, float_precision=4):
        print("\nPer-day data:")
        print(df.select([
            "date",
            pl.col("morning_ret").round(4),
            pl.col("afternoon_ret").round(4),
            (pl.col("gex_total") / 1e6).round(1).alias("gex_$M"),
        ]))

    # (a) Overall correlation
    r_all, p_all, n_all = _corr_with_p(mr, ar)
    print(f"\n[Overall]    corr(morning, afternoon) = {r_all:+.3f}  "
          f"(p={p_all:.3f}, n={n_all})")

    # (b) Conditional on GEX sign.
    neg_mask = gex < 0
    pos_mask = gex > 0
    r_neg, p_neg, n_neg = _corr_with_p(mr[neg_mask], ar[neg_mask])
    r_pos, p_pos, n_pos = _corr_with_p(mr[pos_mask], ar[pos_mask])
    print(f"[GEX < 0]    corr = {r_neg:+.3f}  (p={p_neg:.3f}, n={n_neg})"
          "   — hyp: POSITIVE  (MM short gamma → momentum)")
    print(f"[GEX > 0]    corr = {r_pos:+.3f}  (p={p_pos:.3f}, n={n_pos})"
          "   — hyp: NEGATIVE  (MM long gamma → mean reversion)")

    # (c) Strongly-positive vs strongly-negative |GEX| buckets. Useful at
    # larger N — isolates days where dealer hedging most matters.
    if n >= 8:
        q_lo = np.quantile(gex, 0.25)
        q_hi = np.quantile(gex, 0.75)
        strong_pos = gex >= q_hi
        strong_neg = gex <= q_lo
        r_sp, p_sp, n_sp = _corr_with_p(mr[strong_pos], ar[strong_pos])
        r_sn, p_sn, n_sn = _corr_with_p(mr[strong_neg], ar[strong_neg])
        print(
            f"[Bot-25% GEX] corr = {r_sn:+.3f}  (p={p_sn:.3f}, n={n_sn})"
            f"   GEX ≤ ${q_lo / 1e6:+.0f}M  — hyp: strongly POSITIVE"
        )
        print(
            f"[Top-25% GEX] corr = {r_sp:+.3f}  (p={p_sp:.3f}, n={n_sp})"
            f"   GEX ≥ ${q_hi / 1e6:+.0f}M  — hyp: strongly NEGATIVE"
        )

    # (d) Interaction regression:
    #     afternoon = α + β₁·morning + β₂·(morning × gex_norm) + ε
    # Hypothesis: β₂ < 0  (more-positive GEX ⇒ more dampening).
    gex_scale = np.median(np.abs(gex)) or 1.0
    gex_norm = gex / gex_scale
    X = np.column_stack([np.ones(n), mr, mr * gex_norm])
    y = ar
    results = _ols_with_tstats(X, y, ["intercept", "morning_ret", "morning×GEX_norm"])
    print(f"\n[Interaction regression]  gex_norm = gex / {gex_scale:.2e}")
    print(f"  afternoon_ret = α + β₁·morning_ret + β₂·(morning_ret × gex_norm)")
    print("  {:<22s} {:>10s} {:>10s} {:>8s} {:>8s}".format(
        "term", "beta", "se", "t", "p"
    ))
    for r in results:
        print("  {:<22s} {:>+10.4f} {:>10.4f} {:>+8.2f} {:>8.3f}".format(
            r["name"], r["beta"], r["se"], r["t"], r["p"]
        ))
    print("  Hypothesis: β₂ < 0  (positive GEX dampens momentum).")


# ----- Plot ---------------------------------------------------------------

def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (x_line, y_line) for an OLS fit. None if < 2 points."""
    if len(x) < 2:
        return None
    slope, intercept, *_ = stats.linregress(x, y)
    x_line = np.array([x.min(), x.max()])
    y_line = slope * x_line + intercept
    return x_line, y_line


def save_scatter(df: pl.DataFrame, path: str) -> None:
    """Two-panel scatter:
      left  — all points colored by GEX value (diverging colormap + colorbar)
      right — split by GEX sign with per-regime OLS trend lines
    Axes in percent. Uniform marker size so every point is clearly visible.
    """
    mr = df["morning_ret"].to_numpy() * 100   # %
    ar = df["afternoon_ret"].to_numpy() * 100  # %
    gex = df["gex_total"].to_numpy() / 1e9    # $B

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Panel 1: continuous color by GEX ---------------------------------
    ax = axes[0]
    vmax = max(abs(gex.min()), abs(gex.max())) or 1.0
    sc = ax.scatter(
        mr, ar, c=gex, cmap="RdBu", vmin=-vmax, vmax=vmax,
        s=70, alpha=0.85, edgecolor="black", linewidth=0.4,
    )
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("Morning return (open → 15:30), %")
    ax.set_ylabel("Afternoon return (15:30 → close), %")
    ax.set_title(f"{TICKER}: afternoon vs morning, colored by GEX")
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("GEX at 15:30 ($B)")

    # --- Panel 2: split + trend lines -------------------------------------
    ax = axes[1]
    neg = gex < 0
    pos = gex >= 0
    ax.scatter(
        mr[neg], ar[neg], c="crimson", s=70, alpha=0.75,
        edgecolor="black", linewidth=0.4,
        label=f"GEX < 0  (n={int(neg.sum())})",
    )
    ax.scatter(
        mr[pos], ar[pos], c="steelblue", s=70, alpha=0.75,
        edgecolor="black", linewidth=0.4,
        label=f"GEX ≥ 0  (n={int(pos.sum())})",
    )

    # Per-regime trend lines, labelled with slope and r.
    if neg.sum() >= 2:
        xy = _fit_line(mr[neg], ar[neg])
        r_neg, *_ = stats.pearsonr(mr[neg], ar[neg])
        if xy is not None:
            ax.plot(xy[0], xy[1], color="crimson", lw=2,
                    label=f"GEX<0 fit  r={r_neg:+.2f}")
    if pos.sum() >= 2:
        xy = _fit_line(mr[pos], ar[pos])
        r_pos, *_ = stats.pearsonr(mr[pos], ar[pos])
        if xy is not None:
            ax.plot(xy[0], xy[1], color="steelblue", lw=2,
                    label=f"GEX≥0 fit  r={r_pos:+.2f}")

    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("Morning return (open → 15:30), %")
    ax.set_ylabel("Afternoon return (15:30 → close), %")
    ax.set_title("Split by GEX sign + per-regime OLS")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.25)

    fig.suptitle(
        f"Hypothesis: GEX < 0 → momentum (r > 0), "
        f"GEX > 0 → mean reversion (r < 0)",
        y=1.00, fontsize=10, color="dimgray",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    logger.info("Saved scatter to %s", path)


# ----- Main --------------------------------------------------------------

def main() -> None:
    df = build_dataset()
    if df.is_empty():
        logger.error("No overlap between GEX output and stock bars cache.")
        return

    table_path = OUTPUT_DIR / "momentum_table.parquet"
    df.write_parquet(table_path)
    logger.info("Saved merged per-day table to %s", table_path)

    print_report(df)

    scatter_path = OUTPUT_DIR / "momentum_scatter.png"
    save_scatter(df, str(scatter_path))


if __name__ == "__main__":
    main()
