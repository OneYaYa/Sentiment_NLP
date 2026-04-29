"""
Regression analysis and plotting for Crudebert cumulative sentiment (CSS) vs futures prices.

Uses cumulative sentiment computed as:

    CSS_t = SV_t + sum_{i=1}^6 exp(-i/7) * SV_{t-i}

where SV_t is the daily Crudebert sentiment score. This script:

- Merges CSS with WTI and Brent futures.
- Runs OLS with Newey–West (HAC) standard errors of next-day returns on CSS_t.
- Reports beta, t_beta (HAC), R2, correlation, and quintile long-short.
- Plots CSS_t and close prices over time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sentiment_alpha_validation import (
    ols_regression,
    correlation_and_ttest,
    quintile_sorts,
)
from sentiment_strategy import load_yahoo_oil_csv


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
FINAL_ANALYSIS_DIR = ROOT_DIR / "final_analysis"

CUM_SENTIMENT_CSV = FINAL_ANALYSIS_DIR / "daily_cumulative_sentiment_scores_crudebert_2017_2024.csv"
WTI_CSV = DATA_DIR / "wti.csv"
BRENT_CSV = DATA_DIR / "brent.csv"
PLOTS_DIR = DATA_DIR / "backtest_plots"


def prepare_css_data(market_csv: Path, css_csv: Path) -> pd.DataFrame:
    """Load market data and cumulative sentiment, merge on date, and compute next-day returns."""
    mkt = load_yahoo_oil_csv(str(market_csv), recompute_returns=True)
    css = pd.read_csv(css_csv)
    css["date"] = pd.to_datetime(css["date"])

    df = mkt.merge(css[["date", "css_7d_exp"]], on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)
    df["ret_next"] = df["ret"].shift(-1)
    df = df.dropna(subset=["ret_next", "css_7d_exp"]).reset_index(drop=True)
    return df


def run_regression(df: pd.DataFrame, instrument: str) -> dict[str, Any]:
    """Run regression, correlation, and quintile sorts of next-day returns on CSS."""
    y = df["ret_next"].to_numpy()
    x = df["css_7d_exp"].to_numpy()

    reg = ols_regression(y, x)
    ic = correlation_and_ttest(x, y)
    qq = quintile_sorts(x, y)

    result: dict[str, Any] = {
        "instrument": instrument,
        "n": reg["n"],
        "beta": reg["beta"],
        "t_beta": reg["t_beta"],
        "R2": reg["R2"],
        "shapiro_w": reg.get("shapiro_w", np.nan),
        "shapiro_p": reg.get("shapiro_p", np.nan),
        "corr": ic["corr"],
        "t_corr": ic["t_stat"],
        "long_short": qq.get("long_short", np.nan),
        "t_ls": qq.get("t_ls", np.nan),
    }
    return result


def plot_css_and_price(df: pd.DataFrame, instrument: str, out_path: Path) -> None:
    """Plot CSS_t and close price over time."""
    x = df["date"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(f"Crudebert cumulative sentiment and {instrument} futures price", fontsize=14, fontweight="bold")

    # Panel 1: CSS_t
    ax1 = axes[0]
    ax1.plot(x, df["css_7d_exp"], color="C0", lw=0.8, label="CSS (7d exp-weighted)")
    ax1.axhline(0.0, color="gray", ls="--", lw=0.8)
    ax1.set_ylabel("CSS_t")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Crudebert cumulative sentiment (CSS_t)")

    # Panel 2: Close price
    ax2 = axes[1]
    ax2.plot(x, df["close"], color="C1", lw=0.8, label=f"{instrument} close")
    ax2.set_ylabel("Close price")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f"{instrument} futures close price")

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    fig.autofmt_xdate()
    plt.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")


def main() -> None:
    if not CUM_SENTIMENT_CSV.exists():
        print(f"Cumulative sentiment CSV not found: {CUM_SENTIMENT_CSV}")
        print("Run: python final_analysis/calculate_cumulative_sentiment.py --input final_analysis/daily_sentiment_scores_crudebert_2017_2024.csv")
        return

    instruments: list[tuple[str, Path]] = []
    if WTI_CSV.exists():
        instruments.append(("WTI", WTI_CSV))
    if BRENT_CSV.exists():
        instruments.append(("Brent", BRENT_CSV))
    if not instruments:
        print("No futures CSV found (expected data/wti.csv or data/brent.csv).")
        return

    print("=" * 60)
    print("Crudebert CSS vs futures: regression and plots")
    print("=" * 60)
    print(f"CSS input: {CUM_SENTIMENT_CSV}")
    print()

    for name, mkt_path in instruments:
        print(f"--- {name} ---")
        df = prepare_css_data(mkt_path, CUM_SENTIMENT_CSV)
        if df.empty:
            print("  No overlapping dates between CSS and futures.")
            continue
        print(f"  Rows: {len(df)}, {df['date'].min().date()} to {df['date'].max().date()}")

        res = run_regression(df, name)
        print(f"  Regression (ret_next on CSS_t):")
        print(f"    beta = {res['beta']:.6f}, t (Newey-West) = {res['t_beta']:.3f}, R2 = {res['R2']:.6f}, n = {res['n']}")
        if not np.isnan(res.get("shapiro_w", np.nan)):
            print(f"    Shapiro-Wilk (residuals): W = {res['shapiro_w']:.4f}, p = {res['shapiro_p']:.4f}")
        print(f"    Correlation: r = {res['corr']:.4f}, t = {res['t_corr']:.3f}")
        print(f"    Long-short (Q5-Q1): {res['long_short']:.6f}, t = {res['t_ls']:.3f}")

        out_plot = PLOTS_DIR / f"crudebert_css_{name.lower()}.png"
        plot_css_and_price(df, name, out_plot)

    print("\nDone.")


if __name__ == "__main__":
    main()

