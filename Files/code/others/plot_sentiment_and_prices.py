"""
Plot daily sentiment scores and futures close prices side by side over the common time frame.

Use --sentiment finbert | crudebert to choose sentiment source (default: crudebert).
Loads: daily_sentiment_scores_*.csv, data/wti.csv, data/brent.csv (if present)

Restricts to dates where both sentiment and futures data exist, then plots:
  - Top panel: daily sentiment score
  - Bottom panel: close price(s) for WTI and/or Brent

Output: data/backtest_plots/sentiment_and_prices.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
FINAL_ANALYSIS = ROOT_DIR / "final_analysis"
SENTIMENT_FINBERT = FINAL_ANALYSIS / "daily_sentiment_scores_2017_2024.csv"
SENTIMENT_CRUDEBERT = FINAL_ANALYSIS / "daily_sentiment_scores_crudebert_2017_2024.csv"
WTI_CSV = DATA_DIR / "wti.csv"
BRENT_CSV = DATA_DIR / "brent.csv"
PLOTS_DIR = DATA_DIR / "backtest_plots"
OUTPUT_PATH = PLOTS_DIR / "sentiment_and_prices.png"


def load_sentiment(sentiment_csv: Path) -> pd.DataFrame:
    """Load daily sentiment; standardize date column."""
    df = pd.read_csv(sentiment_csv)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_futures(csv_path: Path) -> pd.DataFrame:
    """Load futures CSV; return date and close (lowercase columns)."""
    df = pd.read_csv(csv_path)
    col_map = {c: c.lower() for c in ["Date", "Close"] if c in df.columns}
    if not col_map:
        return pd.DataFrame()
    df = df.rename(columns=col_map)
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "close"]].dropna().sort_values("date").reset_index(drop=True)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Plot sentiment and futures prices")
    parser.add_argument("--sentiment", "-s", choices=["finbert", "crudebert"], default="crudebert",
                        help="Sentiment source: finbert (oil_news) or crudebert (default: crudebert)")
    args = parser.parse_args()
    sentiment_csv = SENTIMENT_CRUDEBERT if args.sentiment == "crudebert" else SENTIMENT_FINBERT

    if not sentiment_csv.exists():
        print(f"Sentiment file not found: {sentiment_csv}")
        if args.sentiment == "crudebert":
            print("Run: python final_analysis/calculate_daily_sentiment.py --input crudebert")
        else:
            print("Run: python final_analysis/calculate_daily_sentiment.py")
        return

    sent = load_sentiment(sentiment_csv)

    # Merge sentiment with each futures series to get common dates
    merged = sent.copy()
    price_cols: list[str] = []
    for name, path in [("WTI", WTI_CSV), ("Brent", BRENT_CSV)]:
        if not path.exists():
            continue
        fut = load_futures(path)
        if fut.empty:
            continue
        merged = merged.merge(
            fut.rename(columns={"close": name}),
            on="date",
            how="inner",
        )
        price_cols.append(name)

    if not price_cols:
        print("No futures data found (wti.csv or brent.csv).")
        return

    merged = merged.sort_values("date").reset_index(drop=True)
    if merged.empty:
        print("No overlapping dates between sentiment and futures.")
        return

    x = merged["date"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle("Daily sentiment score and futures close prices", fontsize=14, fontweight="bold")

    # Panel 1: Sentiment score
    ax1 = axes[0]
    ax1.plot(x, merged["sentiment_score"], color="C0", label="Sentiment score", lw=0.8)
    ax1.axhline(0, color="gray", ls="--", lw=0.8)
    ax1.set_ylabel("Sentiment score")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Daily sentiment score")

    # Panel 2: Close prices
    ax2 = axes[1]
    for name in price_cols:
        ax2.plot(x, merged[name], label=name, lw=0.8)
    ax2.set_ylabel("Close price")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Futures close price")

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    fig.autofmt_xdate()
    plt.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {OUTPUT_PATH}")
    print(f"  Date range: {merged['date'].min().date()} to {merged['date'].max().date()}")
    print(f"  Observations: {len(merged)}")


if __name__ == "__main__":
    main()
