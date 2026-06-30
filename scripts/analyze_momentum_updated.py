"""GEX momentum analysis — rolling 2-year windows.

Same hypothesis and statistics as analyze_momentum.py, but run separately
for each consecutive 2-year calendar window found in the data:
    2021-2022, 2022-2023, 2023-2024, 2024-2025, 2025-2026, …

Outputs per window (Y1_Y2):
    data/output/momentum_scatter_Y1_Y2.png
    data/output/momentum_table_Y1_Y2.parquet

The full-dataset table (momentum_table.parquet) is also written for
compatibility with gex_momentum_strategy.py.

Run:
    python -m scripts.analyze_momentum_updated
"""
from __future__ import annotations

import logging
import sys
from datetime import date, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

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
    path = storage.stock_bars_cache_path(ticker, day)
    df = storage.read_parquet(path)
    if df is None or df.is_empty():
        return None, None

    ny = df.filter(
        (pl.col("ts_et").dt.time() >= time(9, 30))
        & (pl.col("ts_et").dt.time() <= time(16, 0))
    ).sort("ts_et")
    if ny.is_empty():
        return None, None
    return float(ny["o"][0]), float(ny["c"][-1])


def build_dataset() -> pl.DataFrame:
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
    mask = np.isfinite(x) & np.isfinite(y)
    xc, yc = x[mask], y[mask]
    if len(xc) < 3:
        return float("nan"), float("nan"), len(xc)
    r, p = stats.pearsonr(xc, yc)
    return float(r), float(p), len(xc)


def _ols_with_tstats(
    X: np.ndarray, y: np.ndarray, names: list[str]
) -> list[dict]:
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

    r_all, p_all, n_all = _corr_with_p(mr, ar)
    print(f"\n[Overall]    corr(morning, afternoon) = {r_all:+.3f}  "
          f"(p={p_all:.3f}, n={n_all})")

    neg_mask = gex < 0
    pos_mask = gex > 0
    r_neg, p_neg, n_neg = _corr_with_p(mr[neg_mask], ar[neg_mask])
    r_pos, p_pos, n_pos = _corr_with_p(mr[pos_mask], ar[pos_mask])
    print(f"[GEX < 0]    corr = {r_neg:+.3f}  (p={p_neg:.3f}, n={n_neg})"
          "   — hyp: POSITIVE  (MM short gamma → momentum)")
    print(f"[GEX > 0]    corr = {r_pos:+.3f}  (p={p_pos:.3f}, n={n_pos})"
          "   — hyp: NEGATIVE  (MM long gamma → mean reversion)")

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
    if len(x) < 2:
        return None
    slope, intercept, *_ = stats.linregress(x, y)
    x_line = np.array([x.min(), x.max()])
    y_line = slope * x_line + intercept
    return x_line, y_line


def save_scatter(df: pl.DataFrame, path: str, title_suffix: str = "") -> None:
    mr  = df["morning_ret"].to_numpy()  * 100
    ar  = df["afternoon_ret"].to_numpy() * 100
    gex = df["gex_total"].to_numpy()    / 1e9

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

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
    ax.set_title(f"{TICKER}: afternoon vs morning, colored by GEX{title_suffix}")
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("GEX at 15:30 ($B)")

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
    ax.set_title(f"Split by GEX sign + per-regime OLS{title_suffix}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.25)

    fig.suptitle(
        f"Hypothesis: GEX < 0 → momentum (r > 0), "
        f"GEX > 0 → mean reversion (r < 0)",
        y=1.00, fontsize=10, color="dimgray",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved scatter to %s", path)


# ----- Main --------------------------------------------------------------

def main() -> None:
    df = build_dataset()
    if df.is_empty():
        logger.error("No overlap between GEX output and stock bars cache.")
        return

    # Save full table for backward compatibility with gex_momentum_strategy.py
    table_path = OUTPUT_DIR / "momentum_table.parquet"
    df.write_parquet(table_path)
    logger.info("Saved full merged per-day table to %s", table_path)

    # Determine rolling 2-year windows from the years present in the data
    years = sorted(df["date"].dt.year().unique().to_list())
    if len(years) < 2:
        logger.error("Need at least 2 years of data for rolling windows.")
        return

    windows = [(years[i], years[i + 1]) for i in range(len(years) - 1)]
    logger.info(
        "Found %d calendar years (%d–%d) → %d rolling 2-year windows",
        len(years), years[0], years[-1], len(windows),
    )

    for y1, y2 in windows:
        label = f"{y1}_{y2}"
        window_df = df.filter(pl.col("date").dt.year().is_in([y1, y2]))

        if window_df.is_empty():
            logger.warning("No data for window %s, skipping", label)
            continue

        print(f"\n{'#' * 72}")
        print(f"# Rolling window: {y1} – {y2}  (N = {window_df.height} trading days)")
        print(f"{'#' * 72}")

        print_report(window_df)

        scatter_path = OUTPUT_DIR / f"momentum_scatter_{label}.png"
        save_scatter(window_df, str(scatter_path), title_suffix=f"  [{y1}–{y2}]")

        window_table_path = OUTPUT_DIR / f"momentum_table_{label}.parquet"
        window_df.write_parquet(window_table_path)
        logger.info("Saved window table to %s", window_table_path)


if __name__ == "__main__":
    main()
