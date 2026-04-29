"""
Visualize backtest results from strategy backtest CSVs.

Use --strategy zscore (default) for the original z-score backtest, or
--strategy quantile for the quantile + medium-horizon backtest.

Plots for each instrument:
  - Equity curve and close price (normalized)
  - Applied weight (position)
  - Turnover
  - Applied cost

Outputs saved to data/backtest_plots/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PLOTS_DIR = DATA_DIR / "backtest_plots"

# Strategy -> (wti_csv, brent_csv, plot_suffix, title_suffix)
BACKTEST_FILES = {
    "zscore": (
        DATA_DIR / "wti_strategy_backtest.csv",
        DATA_DIR / "brent_strategy_backtest.csv",
        "",  # wti_backtest_plots.png, brent_backtest_plots.png
        "",
    ),
    "quantile": (
        DATA_DIR / "wti_strategy_backtest_quantile.csv",
        DATA_DIR / "brent_strategy_backtest_quantile.csv",
        "_quantile",  # wti_backtest_plots_quantile.png, etc.
        " (quantile + medium-horizon)",
    ),
}


def load_backtest(csv_path: Path) -> pd.DataFrame:
    """Load backtest CSV and ensure date is datetime."""
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def recompute_equity_joined(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute equity so folds join: fold 1 starts at 1, each subsequent fold
    starts at the ending equity of the previous fold.
    """
    if "strat_ret" not in df.columns or "date" not in df.columns:
        return df
    df = df.copy()
    df["year"] = df["date"].dt.year
    years = df["year"].unique()
    years = sorted(years)
    equity_new = []
    start_equity = 1.0
    for yr in years:
        mask = df["year"] == yr
        block = df.loc[mask].copy()
        rets = block["strat_ret"].fillna(0).values
        eq = [start_equity * (1 + rets[0])]
        for r in rets[1:]:
            eq.append(eq[-1] * (1 + r))
        equity_new.extend(eq)
        start_equity = eq[-1]
    df["equity"] = equity_new
    df = df.drop(columns=["year"])
    return df


def plot_backtest(df: pd.DataFrame, instrument: str, out_path: Path, title_suffix: str = "") -> None:
    """Create multi-panel figure: equity & close, weight_applied, turnover, cost_applied."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Backtest: {instrument}{title_suffix}", fontsize=14, fontweight="bold")
    x = df["date"]

    # --- Panel 1: Equity and close price (normalized to 100 at start) ---
    ax1 = axes[0]
    if "equity" in df.columns:
        equity = df["equity"].values
        ax1.plot(x, equity, color="C0", label="Equity", lw=1.5)
    if "close" in df.columns:
        close = df["close"].values
        close_norm = 100 * close / close[0]
        ax1_twin = ax1.twinx()
        ax1_twin.plot(x, close_norm, color="C1", alpha=0.8, label="Close (norm)", lw=1)
        ax1_twin.set_ylabel("Close (norm to 100)", color="C1")
        ax1_twin.tick_params(axis="y", labelcolor="C1")
    ax1.set_ylabel("Equity")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Equity and close price (normalized)")

    # --- Panel 2: Applied weight ---
    ax2 = axes[1]
    if "weight_applied" in df.columns:
        ax2.fill_between(x, 0, df["weight_applied"], color="C2", alpha=0.6, label="Weight applied")
        ax2.axhline(0, color="gray", ls="--", lw=0.8)
    ax2.set_ylabel("Weight")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Applied weight (position)")

    # --- Panel 3: Turnover ---
    ax3 = axes[2]
    if "turnover" in df.columns:
        ax3.bar(x, df["turnover"], width=2, color="C3", alpha=0.7, label="Turnover")
    ax3.set_ylabel("Turnover")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)
    ax3.set_title("Turnover")

    # --- Panel 4: Applied cost ---
    ax4 = axes[3]
    if "cost_applied" in df.columns:
        ax4.bar(x, df["cost_applied"], width=2, color="C4", alpha=0.7, label="Cost applied")
    ax4.set_ylabel("Cost applied")
    ax4.set_xlabel("Date")
    ax4.legend(loc="upper right")
    ax4.grid(True, alpha=0.3)
    ax4.set_title("Applied cost")

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize backtest results (z-score or quantile strategy).")
    parser.add_argument(
        "--strategy", "-s",
        choices=list(BACKTEST_FILES.keys()),
        default="zscore",
        help="Backtest to plot: zscore (default) or quantile",
    )
    args = parser.parse_args()

    wti_csv, brent_csv, suffix, title_suffix = BACKTEST_FILES[args.strategy]
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for label, csv_path in [("WTI", wti_csv), ("Brent", brent_csv)]:
        if not csv_path.exists():
            print(f"Skip {label}: {csv_path} not found")
            continue
        df = load_backtest(csv_path)
        df = recompute_equity_joined(df)
        out_path = PLOTS_DIR / f"{label.lower()}_backtest_plots{suffix}.png"
        plot_backtest(df, label, out_path, title_suffix=title_suffix)
        print(f"Saved: {out_path}")

    print(f"\nPlots directory: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
