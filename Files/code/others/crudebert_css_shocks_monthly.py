"""
Monthly positive/negative news shocks from Crudebert cumulative sentiment, compared with futures prices.

Definitions (for a given CSS_t series over the full sample):

- Positive news shock at t:      CSS_t > q-quantile of {CSS_s} over all s
- Negative news shock at t:      CSS_t < (1 - q)-quantile of {CSS_s} over all s

Default q = 0.8.

For each instrument (WTI, Brent), this script:
  1. Loads daily cumulative Crudebert CSS_t and daily futures prices.
  2. Flags positive / negative shock days by the thresholds above.
  3. Aggregates to monthly counts of positive and negative shocks.
  4. Aggregates futures prices to month-end close.
  5. Produces two plots per instrument:
       - Positive shocks per month vs month-end futures price.
       - Negative shocks per month vs month-end futures price.

Outputs are saved under data/backtest_plots/:
  - crudebert_css_pos_shocks_<instrument>.png
  - crudebert_css_neg_shocks_<instrument>.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sentiment_strategy import load_yahoo_oil_csv


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
FINAL_ANALYSIS_DIR = ROOT_DIR / "final_analysis"

CUM_SENTIMENT_CSV = FINAL_ANALYSIS_DIR / "daily_cumulative_sentiment_scores_crudebert_2017_2024.csv"
WTI_CSV = DATA_DIR / "wti.csv"
BRENT_CSV = DATA_DIR / "brent.csv"
PLOTS_DIR = DATA_DIR / "backtest_plots"


def prepare_shocks(css_csv: Path, q: float = 0.8) -> pd.DataFrame:
    """
    Load cumulative sentiment CSV and flag positive / negative shock days.

    Parameters
    ----------
    css_csv : Path
        Path to daily cumulative sentiment CSV with columns: date, css_7d_exp.
    q : float, optional
        Upper quantile threshold for positive shocks (default 0.8).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, css_7d_exp, pos_shock, neg_shock.
    """
    df = pd.read_csv(css_csv)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    css = df["css_7d_exp"].astype(float)
    css_valid = css[np.isfinite(css)]
    if css_valid.empty:
        raise ValueError("No valid CSS values found in input CSV.")

    upper = css_valid.quantile(q)
    lower = css_valid.quantile(1.0 - q)

    df["pos_shock"] = css > upper
    df["neg_shock"] = css < lower
    return df


def aggregate_monthly_shocks(df_css: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily shock flags to monthly counts.

    Returns DataFrame with columns: month, pos_shocks, neg_shocks.
    """
    tmp = df_css.copy()
    # Use month-end timestamps so they align with futures resampled using "ME"
    tmp["month"] = tmp["date"].dt.to_period("M").dt.to_timestamp("M")
    grouped = (
        tmp.groupby("month", as_index=False)[["pos_shock", "neg_shock"]]
        .sum()
        .rename(columns={"pos_shock": "pos_shocks", "neg_shock": "neg_shocks"})
    )
    return grouped


def aggregate_monthly_price(market_csv: Path) -> pd.DataFrame:
    """
    Load futures CSV and aggregate close price to month-end.

    Returns DataFrame with columns: month, close_month_end.
    """
    mkt = load_yahoo_oil_csv(str(market_csv), recompute_returns=True)
    mkt = mkt.sort_values("date").reset_index(drop=True)
    mkt = mkt.set_index("date")
    monthly = (
        mkt["close"]
        .resample("ME")
        .last()
        .dropna()
        .reset_index()
        .rename(columns={"date": "month", "close": "close_month_end"})
    )
    return monthly


def plot_monthly_shocks_on_ax(
    ax1: plt.Axes,
    monthly: pd.DataFrame,
    instrument: str,
    shock_col: str,
    title_suffix: str,
) -> None:
    """
    Plot monthly shock counts and month-end futures price.

    shock_col: 'pos_shocks' or 'neg_shocks'
    """
    x = monthly["month"]
    shocks = monthly[shock_col]
    price = monthly["close_month_end"]

    # Left axis: shock counts (bars)
    ax1.bar(x, shocks, width=20, color="C0", alpha=0.7, label=title_suffix)
    ax1.set_ylabel("Monthly shocks count", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.grid(axis="y", alpha=0.3)

    # Right axis: price (line)
    ax2 = ax1.twinx()
    ax2.plot(x, price, color="C1", lw=1.2, label=f"{instrument} close (month-end)")
    ax2.set_ylabel("Month-end close", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")

    # X-axis formatting
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())

    # Combined legend
    lines_labels = ax1.get_legend_handles_labels()
    lines2_labels = ax2.get_legend_handles_labels()
    handles = lines_labels[0] + lines2_labels[0]
    labels = lines_labels[1] + lines2_labels[1]
    ax1.legend(handles, labels, loc="upper left", framealpha=0.9)


def main() -> None:
    if not CUM_SENTIMENT_CSV.exists():
        print(f"Cumulative Crudebert CSS CSV not found: {CUM_SENTIMENT_CSV}")
        print(
            "Run: python final_analysis/calculate_cumulative_sentiment.py "
            "--input final_analysis/daily_sentiment_scores_crudebert_2017_2024.csv "
            "--output final_analysis/daily_cumulative_sentiment_scores_crudebert_2017_2024.csv"
        )
        return

    df_css = prepare_shocks(CUM_SENTIMENT_CSV, q=0.8)
    monthly_shocks = aggregate_monthly_shocks(df_css)

    instruments: list[tuple[str, Path]] = []
    if WTI_CSV.exists():
        instruments.append(("WTI", WTI_CSV))
    if BRENT_CSV.exists():
        instruments.append(("Brent", BRENT_CSV))

    if not instruments:
        print("No futures CSV found (expected data/wti.csv or data/brent.csv).")
        return

    print("=" * 60)
    print("Crudebert CSS monthly shocks vs futures prices")
    print("=" * 60)
    print(f"CSS input: {CUM_SENTIMENT_CSV}")
    print(f"Quantile threshold q = 0.8 (upper), 1-q = 0.2 (lower)")
    print()

    # Prepare merged monthly data per instrument
    merged_by_instr: dict[str, pd.DataFrame] = {}
    for name, mkt_path in instruments:
        print(f"--- {name} ---")
        monthly_price = aggregate_monthly_price(mkt_path)
        merged = monthly_shocks.merge(monthly_price, on="month", how="inner")
        if merged.empty:
            print("  No overlapping months between CSS and futures.")
            continue

        print(f"  Months: {len(merged)}, {merged['month'].min().date()} to {merged['month'].max().date()}")
        merged_by_instr[name] = merged

    if not merged_by_instr:
        print("\nNo overlapping months for any instrument; nothing to plot.")
        print("\nDone.")
        return

    # Create a single 2xN figure to hold all subplots (Positive/Negative x instruments)
    instr_names = list(merged_by_instr.keys())
    n_instr = len(instr_names)
    fig, axes = plt.subplots(
        2,
        n_instr,
        figsize=(6 * n_instr, 6),
        sharex=True,
    )

    # Ensure axes is 2D array even when n_instr == 1
    if n_instr == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    fig.suptitle(
        "Crudebert CSS monthly shocks vs month-end futures prices",
        fontsize=14,
        fontweight="bold",
    )

    for col, name in enumerate(instr_names):
        merged = merged_by_instr[name]
        ax_pos = axes[0, col]
        ax_neg = axes[1, col]

        plot_monthly_shocks_on_ax(
            ax_pos,
            merged,
            instrument=name,
            shock_col="pos_shocks",
            title_suffix="Positive",
        )
        ax_pos.set_title(f"{name}: Positive shocks")

        plot_monthly_shocks_on_ax(
            ax_neg,
            merged,
            instrument=name,
            shock_col="neg_shocks",
            title_suffix="Negative",
        )
        ax_neg.set_title(f"{name}: Negative shocks")

    fig.autofmt_xdate()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_all = PLOTS_DIR / "crudebert_css_shocks_all.png"
    plt.savefig(out_all, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved combined plot: {out_all}")

    print("\nDone.")


if __name__ == "__main__":
    main()

