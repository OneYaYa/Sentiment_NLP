# ============================================
# BASE OIL TRADING STRATEGY BACKTEST IN PYTHON
# Strategy: Moving Average Crossover on WTI Crude Oil
# Data source: Yahoo Finance via yfinance
# Ticker: CL=F  (WTI Crude Oil futures on Yahoo Finance)
# ============================================

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# -----------------------------
# USER CONFIG
# -----------------------------
TICKER = "CL=F"          # WTI Crude Oil futures; use "BZ=F" for Brent
START_DATE = "2010-01-01"
END_DATE = None          # None = up to latest available data
SHORT_WINDOW = 50
LONG_WINDOW = 200
TRANSACTION_COST_BPS = 5   # cost applied when position changes, in basis points
INITIAL_CAPITAL = 100000

# -----------------------------
# DOWNLOAD DATA
# -----------------------------
def load_price_data(ticker: str, start: str, end=None) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker}.")

    # Flatten columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            if col == "Adj Close" and "Close" in df.columns:
                df["Adj Close"] = df["Close"]
            elif col == "Volume":
                df["Volume"] = 0
            else:
                raise ValueError(f"Missing required column: {col}")

    df = df.dropna(subset=["Adj Close"]).copy()
    df.index = pd.to_datetime(df.index)
    return df


# -----------------------------
# BUILD STRATEGY
# -----------------------------
def build_strategy(df: pd.DataFrame,
                   short_window: int,
                   long_window: int,
                   transaction_cost_bps: float) -> pd.DataFrame:
    out = df.copy()

    # Use adjusted close if present
    out["price"] = out["Adj Close"]

    # Daily returns
    out["asset_return"] = out["price"].pct_change()

    # Moving averages
    out["ma_short"] = out["price"].rolling(short_window).mean()
    out["ma_long"] = out["price"].rolling(long_window).mean()

    # Signal: +1 when short MA > long MA, else 0 (long-only trend baseline)
    # For long/short baseline, replace 0 with -1.
    out["signal"] = np.where(out["ma_short"] > out["ma_long"], 1, 0)

    # Position is taken next day to avoid lookahead bias
    out["position"] = pd.Series(out["signal"], index=out.index).shift(1).fillna(0)

    # Turnover for transaction cost
    out["trade"] = out["position"].diff().abs().fillna(0)

    # Transaction cost in decimal
    tc = transaction_cost_bps / 10000.0
    out["transaction_cost"] = out["trade"] * tc

    # Strategy return
    out["strategy_return_gross"] = out["position"] * out["asset_return"]
    out["strategy_return_net"] = out["strategy_return_gross"] - out["transaction_cost"]

    # Equity curves
    out["buy_hold_curve"] = (1 + out["asset_return"].fillna(0)).cumprod()
    out["strategy_curve"] = (1 + out["strategy_return_net"].fillna(0)).cumprod()

    return out.dropna().copy()


# -----------------------------
# PERFORMANCE METRICS
# -----------------------------
def compute_drawdown(equity_curve: pd.Series) -> pd.Series:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return drawdown

def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    total_return = (1 + returns).prod()
    n_years = len(returns) / periods_per_year
    if n_years == 0:
        return np.nan
    return total_return ** (1 / n_years) - 1

def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    return returns.std() * np.sqrt(periods_per_year)

def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252, rf: float = 0.0) -> float:
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    excess = returns - (rf / periods_per_year)
    vol = excess.std()
    if vol == 0 or np.isnan(vol):
        return np.nan
    return (excess.mean() / vol) * np.sqrt(periods_per_year)

def max_drawdown(returns_or_curve: pd.Series, is_curve: bool = False) -> float:
    curve = returns_or_curve if is_curve else (1 + returns_or_curve.fillna(0)).cumprod()
    dd = compute_drawdown(curve)
    return dd.min()

def summary_stats(df: pd.DataFrame, initial_capital: float = 100000) -> pd.DataFrame:
    strat_rets = df["strategy_return_net"]
    bh_rets = df["asset_return"]

    strat_curve = df["strategy_curve"]
    bh_curve = df["buy_hold_curve"]

    stats = pd.DataFrame({
        "Metric": [
            "Start Date",
            "End Date",
            "Observations",
            "Final Equity ($)",
            "Total Return (%)",
            "Annualized Return (%)",
            "Annualized Volatility (%)",
            "Sharpe Ratio",
            "Max Drawdown (%)",
            "Time in Market (%)",
            "Number of Trades"
        ],
        "Strategy": [
            df.index.min().date(),
            df.index.max().date(),
            len(df),
            round(initial_capital * strat_curve.iloc[-1], 2),
            round((strat_curve.iloc[-1] - 1) * 100, 2),
            round(annualized_return(strat_rets) * 100, 2),
            round(annualized_volatility(strat_rets) * 100, 2),
            round(sharpe_ratio(strat_rets), 2),
            round(max_drawdown(strat_curve, is_curve=True) * 100, 2),
            round(df["position"].mean() * 100, 2),
            int((df["trade"] > 0).sum())
        ],
        "Buy & Hold": [
            df.index.min().date(),
            df.index.max().date(),
            len(df),
            round(initial_capital * bh_curve.iloc[-1], 2),
            round((bh_curve.iloc[-1] - 1) * 100, 2),
            round(annualized_return(bh_rets) * 100, 2),
            round(annualized_volatility(bh_rets) * 100, 2),
            round(sharpe_ratio(bh_rets), 2),
            round(max_drawdown(bh_curve, is_curve=True) * 100, 2),
            100.0,
            1
        ]
    })

    return stats


# -----------------------------
# PLOTTING
# -----------------------------
def plot_results(df: pd.DataFrame, ticker: str, save_dir: str = "output", show_plots: bool = True):
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # -------------------------
    # Price + Moving Averages
    # -------------------------
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df["price"], label="Price")
    plt.plot(df.index, df["ma_short"], label=f"MA {SHORT_WINDOW}")
    plt.plot(df.index, df["ma_long"], label=f"MA {LONG_WINDOW}")
    plt.fill_between(
        df.index,
        df["price"].min(),
        df["price"].max(),
        where=df["position"] > 0,
        alpha=0.10,
        label="Long regime"
    )
    plt.title(f"{ticker} Price with Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()

    price_path = os.path.join(save_dir, f"{ticker}_price_ma.png")
    plt.savefig(price_path)
    if show_plots:
        plt.show()
    plt.close()

    # -------------------------
    # Equity Curves
    # -------------------------
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df["strategy_curve"], label="Strategy")
    plt.plot(df.index, df["buy_hold_curve"], label="Buy & Hold")
    plt.title(f"{ticker} Strategy vs Buy & Hold")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.tight_layout()

    equity_path = os.path.join(save_dir, f"{ticker}_equity_curve.png")
    plt.savefig(equity_path)
    if show_plots:
        plt.show()
    plt.close()

    # -------------------------
    # Drawdowns
    # -------------------------
    strat_dd = compute_drawdown(df["strategy_curve"])
    bh_dd = compute_drawdown(df["buy_hold_curve"])

    plt.figure(figsize=(14, 6))
    plt.plot(df.index, strat_dd * 100, label="Strategy Drawdown (%)")
    plt.plot(df.index, bh_dd * 100, label="Buy & Hold Drawdown (%)")
    plt.title(f"{ticker} Drawdowns")
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.legend()
    plt.tight_layout()

    dd_path = os.path.join(save_dir, f"{ticker}_drawdown.png")
    plt.savefig(dd_path)
    if show_plots:
        plt.show()
    plt.close()

    print(f"\nPlots saved to folder: {os.path.abspath(save_dir)}")


# -----------------------------
# MAIN
# -----------------------------
def main():
    df = load_price_data(TICKER, START_DATE, END_DATE)
    bt = build_strategy(
        df,
        short_window=SHORT_WINDOW,
        long_window=LONG_WINDOW,
        transaction_cost_bps=TRANSACTION_COST_BPS
    )

    stats = summary_stats(bt, initial_capital=INITIAL_CAPITAL)

    print("\n" + "=" * 70)
    print(f"BASE STRATEGY BACKTEST: {TICKER}")
    print(f"Strategy: Long-only moving average crossover ({SHORT_WINDOW}/{LONG_WINDOW})")
    print("=" * 70)
    print(stats.to_string(index=False))

    # Show latest few rows
    print("\nLatest signals:")
    cols = ["price", "ma_short", "ma_long", "signal", "position",
            "asset_return", "strategy_return_net", "strategy_curve"]
    print(bt[cols].tail(10).round(4).to_string())

    plot_results(bt, TICKER)


if __name__ == "__main__":
    main()