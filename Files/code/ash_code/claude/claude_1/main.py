"""
=============================================================
  BASE OIL TRADING STRATEGY — SMA CROSSOVER
  Instrument : WTI Crude Oil Futures  (CL=F via yfinance)
  Strategy   : Dual Simple Moving Average (SMA) Crossover
  Purpose    : Baseline for comparing more complex strategies
=============================================================

STRATEGY LOGIC
--------------
  • Fast SMA (default 20-day) crosses ABOVE slow SMA (50-day)  → BUY  (go long)
  • Fast SMA crosses BELOW slow SMA                             → SELL (exit / go short)
  • Position sizing : fixed-fraction risk using ATR-based stop

RUN MODES
---------
  • Live mode  : requires `pip install yfinance` — downloads real WTI futures data
  • Demo mode  : auto-activates when yfinance is unavailable — uses GBM synthetic data

OUTPUTS
-------
  1. oil_strategy_results.png  — 4-panel performance dashboard
  2. oil_strategy_trades.csv   — full trade log
  3. Console performance report
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import sys, os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving PNG
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG = {
    # ── Data ────────────────────────────────────────────────────────────
    "ticker"         : "CL=F",          # WTI Crude Oil Futures on Yahoo Finance
    "start_date"     : "2018-01-01",
    "end_date"       : datetime.today().strftime("%Y-%m-%d"),
    "interval"       : "1d",

    # ── Strategy parameters ──────────────────────────────────────────────
    "fast_sma"       : 20,              # Fast SMA window (bars)
    "slow_sma"       : 50,              # Slow SMA window (bars)
    "atr_period"     : 14,              # ATR period
    "atr_stop_mult"  : 2.0,             # Stop = entry ± ATR × multiplier
    "risk_per_trade" : 0.01,            # Fraction of equity risked per trade (1%)

    # ── Capital & costs ──────────────────────────────────────────────────
    "initial_capital": 100_000,         # Starting equity in USD
    "commission_pct" : 0.001,           # 0.10% commission per trade leg

    # ── Output ───────────────────────────────────────────────────────────
    "output_dir"     : "./output/",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LAYER
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_data_yfinance(cfg: dict) -> pd.DataFrame:
    """Download OHLCV from Yahoo Finance (requires yfinance)."""
    import yfinance as yf
    df = yf.download(
        cfg["ticker"],
        start=cfg["start_date"],
        end=cfg["end_date"],
        interval=cfg["interval"],
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        raise ValueError(f"No data returned for {cfg['ticker']}.")
    # Flatten MultiIndex columns produced by newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def generate_synthetic_oil_data(cfg: dict) -> pd.DataFrame:
    """
    Produce realistic synthetic WTI OHLCV data via Geometric Brownian Motion.
    Uses volatility and mean-reversion parameters calibrated to WTI history.
    Injects 2 major shocks (COVID-style crash + recovery) to create interesting
    regime changes that test the crossover strategy.
    """
    np.random.seed(42)
    start = pd.Timestamp(cfg["start_date"])
    end   = pd.Timestamp(cfg["end_date"])
    dates = pd.bdate_range(start, end)
    n     = len(dates)

    S0     = 60.0     # starting price ≈ $60/bbl
    mu     = 0.05     # annual drift
    sigma  = 0.35     # annual volatility (WTI is very vol)
    dt     = 1 / 252

    # Simulate log returns
    log_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.randn(n)

    # Inject market regime shocks
    shock_start1, shock_end1 = int(n * 0.35), int(n * 0.45)   # COVID-style crash
    shock_start2, shock_end2 = int(n * 0.60), int(n * 0.70)   # supply squeeze rally
    log_ret[shock_start1:shock_end1] -= 0.018   # sustained sell-off
    log_ret[shock_start2:shock_end2] += 0.012   # sustained rally

    close = S0 * np.exp(np.cumsum(log_ret))
    close = np.maximum(close, 10)               # price floor at $10

    # Build OHLCV
    daily_range = sigma * np.sqrt(dt) * close * np.random.uniform(0.5, 2.0, n)
    high   = close + daily_range * np.random.uniform(0.3, 0.7, n)
    low    = close - daily_range * np.random.uniform(0.3, 0.7, n)
    open_  = low + (high - low) * np.random.rand(n)
    volume = (np.random.randint(300_000, 900_000, n)).astype(float)

    df = pd.DataFrame({
        "Open"  : open_,
        "High"  : high,
        "Low"   : low,
        "Close" : close,
        "Volume": volume,
    }, index=dates)
    return df


def load_data(cfg: dict) -> tuple[pd.DataFrame, bool]:
    """Return (dataframe, is_live_data)."""
    try:
        print("[1/5] Attempting to fetch real data from Yahoo Finance …")
        df = fetch_data_yfinance(cfg)
        print(f"    ✓ Real data loaded: {len(df)} rows "
              f"({df.index[0].date()} → {df.index[-1].date()})")
        return df, True
    except Exception as e:
        print(f"    ⚠  yfinance unavailable ({e})")
        print("    → Falling back to synthetic WTI data for demonstration")
        df = generate_synthetic_oil_data(cfg)
        print(f"    ✓ Synthetic data generated: {len(df)} rows "
              f"({df.index[0].date()} → {df.index[-1].date()})")
        return df, False


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    fast, slow, atr_p = cfg["fast_sma"], cfg["slow_sma"], cfg["atr_period"]
    print(f"[2/5] Computing SMA({fast}), SMA({slow}), ATR({atr_p}) …")

    df = df.copy()
    df[f"SMA_{fast}"] = df["Close"].rolling(fast).mean()
    df[f"SMA_{slow}"] = df["Close"].rolling(slow).mean()

    # Average True Range (Wilder exponential)
    hi_lo = df["High"] - df["Low"]
    hi_pc = (df["High"] - df["Close"].shift(1)).abs()
    lo_pc = (df["Low"]  - df["Close"].shift(1)).abs()
    tr    = pd.concat([hi_lo, hi_pc, lo_pc], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1 / atr_p, adjust=False).mean()

    # Signal: +1 long zone, -1 short zone (NaN during warm-up)
    cond = df[f"SMA_{fast}"] > df[f"SMA_{slow}"]
    df["raw_signal"] = np.where(cond, 1.0, -1.0)
    df.loc[df.index[:slow], "raw_signal"] = np.nan

    df.dropna(subset=[f"SMA_{slow}"], inplace=True)
    print(f"    ✓ {len(df)} usable bars after warm-up")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Event-driven bar-by-bar simulation.
    Executes entries/exits at the CLOSE of the signal bar (next-bar execution
    is a common improvement — adjust by shifting signals if desired).
    """
    print("[3/5] Running backtest …")

    capital      = float(cfg["initial_capital"])
    risk_frac    = cfg["risk_per_trade"]
    atr_mult     = cfg["atr_stop_mult"]
    commission   = cfg["commission_pct"]

    position     = 0        # units held
    entry_price  = 0.0
    stop_price   = 0.0
    direction    = 0        # +1 long / -1 short

    equity_curve : list[dict] = []
    trades       : list[dict] = []
    prev_signal  = 0.0

    for idx, row in df.iterrows():
        price  = float(row["Close"])
        signal = float(row["raw_signal"]) if not pd.isna(row["raw_signal"]) else prev_signal
        atr    = float(row["ATR"])

        # ── Mark-to-market equity ─────────────────────────────────────
        mtm_eq = capital + position * direction * (price - entry_price)

        # ── Stop-loss check (checked before signal logic) ─────────────
        if position > 0:
            hit_stop = (direction ==  1 and price <= stop_price) or \
                       (direction == -1 and price >= stop_price)
            if hit_stop:
                exit_px  = stop_price
                pnl      = direction * position * (exit_px - entry_price)
                fee      = position * exit_px * commission
                capital += pnl - fee
                if trades:
                    trades[-1].update({
                        "exit_date"  : idx,
                        "exit_price" : round(exit_px, 4),
                        "pnl"        : round(pnl - fee, 2),
                        "exit_reason": "stop-loss",
                    })
                position, direction = 0, 0
                mtm_eq = capital

        # ── Signal flip: exit then re-enter ───────────────────────────
        if signal != prev_signal and prev_signal != 0:
            if position > 0:
                pnl      = direction * position * (price - entry_price)
                fee      = position * price * commission
                capital += pnl - fee
                if trades:
                    trades[-1].update({
                        "exit_date"  : idx,
                        "exit_price" : round(price, 4),
                        "pnl"        : round(pnl - fee, 2),
                        "exit_reason": "signal-flip",
                    })
                position, direction = 0, 0
                mtm_eq = capital

            # Open new position in direction of new signal
            if signal in (1.0, -1.0) and atr > 0:
                stop_dist = atr * atr_mult
                units     = max(1, int((capital * risk_frac) / stop_dist))
                entry_fee = units * price * commission

                if capital > entry_fee + stop_dist * units:
                    position    = units
                    direction   = int(signal)
                    entry_price = price
                    stop_price  = price - direction * stop_dist
                    capital    -= entry_fee
                    trades.append({
                        "entry_date"  : idx,
                        "entry_price" : round(price, 4),
                        "direction"   : "long" if direction == 1 else "short",
                        "units"       : units,
                        "stop"        : round(stop_price, 4),
                        "exit_date"   : None,
                        "exit_price"  : None,
                        "pnl"         : None,
                        "exit_reason" : None,
                    })
                    mtm_eq = capital  # entry fee already deducted

        prev_signal = signal
        mtm_eq = capital + position * direction * (price - entry_price)
        equity_curve.append({"date": idx, "equity": mtm_eq, "price": price})

    # ── Close any open position at last bar ───────────────────────────
    if position > 0 and trades:
        last_price = float(df["Close"].iloc[-1])
        last_date  = df.index[-1]
        pnl        = direction * position * (last_price - entry_price)
        fee        = position * last_price * commission
        capital   += pnl - fee
        trades[-1].update({
            "exit_date"  : last_date,
            "exit_price" : round(last_price, 4),
            "pnl"        : round(pnl - fee, 2),
            "exit_reason": "end-of-backtest",
        })

    eq_df    = pd.DataFrame(equity_curve).set_index("date")
    trades_df = pd.DataFrame(trades)
    completed = trades_df.dropna(subset=["pnl"])
    print(f"    ✓ {len(completed)} completed trades  "
          f"({len(trades_df) - len(completed)} open/partial)")
    return eq_df, trades_df


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  PERFORMANCE METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(eq_df: pd.DataFrame,
                    trades_df: pd.DataFrame,
                    cfg: dict) -> dict:
    eq   = eq_df["equity"]
    init = cfg["initial_capital"]
    n_yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-6)

    total_ret = (eq.iloc[-1] / init - 1) * 100
    cagr      = ((eq.iloc[-1] / init) ** (1 / n_yrs) - 1) * 100

    daily_ret = eq.pct_change().dropna()
    sharpe    = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
    neg_ret   = daily_ret[daily_ret < 0]
    sortino   = daily_ret.mean() / neg_ret.std() * np.sqrt(252) if neg_ret.std() > 0 else 0

    roll_max  = eq.cummax()
    drawdown  = (eq - roll_max) / roll_max * 100
    max_dd    = drawdown.min()
    calmar    = cagr / abs(max_dd) if max_dd != 0 else 0

    # Max drawdown duration (consecutive bars in drawdown)
    in_dd     = drawdown < 0
    dd_groups = (in_dd != in_dd.shift()).cumsum()
    dd_lens   = in_dd.groupby(dd_groups).sum()
    max_dd_dur = int(dd_lens.max()) if in_dd.any() else 0

    completed  = trades_df.dropna(subset=["pnl"])
    winners    = completed[completed["pnl"] > 0]
    losers     = completed[completed["pnl"] <= 0]
    win_rate   = len(winners) / len(completed) * 100 if len(completed) > 0 else 0
    avg_win    = winners["pnl"].mean() if len(winners) > 0 else 0
    avg_loss   = losers["pnl"].mean()  if len(losers)  > 0 else 0
    pf_denom   = abs(losers["pnl"].sum()) if len(losers) > 0 else 0
    pf         = winners["pnl"].sum() / pf_denom if pf_denom > 0 else np.inf

    # Expectancy per trade
    expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)

    bh_ret  = (eq_df["price"].iloc[-1] / eq_df["price"].iloc[0] - 1) * 100

    return {
        "total_return_pct"   : round(total_ret,    2),
        "cagr_pct"           : round(cagr,          2),
        "bh_return_pct"      : round(bh_ret,        2),
        "sharpe_ratio"       : round(sharpe,         2),
        "sortino_ratio"      : round(sortino,        2),
        "calmar_ratio"       : round(calmar,         2),
        "max_drawdown_pct"   : round(max_dd,         2),
        "max_dd_duration_days": max_dd_dur,
        "win_rate_pct"       : round(win_rate,       2),
        "total_trades"       : len(completed),
        "winning_trades"     : len(winners),
        "losing_trades"      : len(losers),
        "avg_win_usd"        : round(avg_win,        2),
        "avg_loss_usd"       : round(avg_loss,       2),
        "profit_factor"      : round(pf,             2),
        "expectancy_usd"     : round(expectancy,     2),
        "final_equity_usd"   : round(eq.iloc[-1],    2),
        "n_years"            : round(n_yrs,           2),
        "_drawdown_series"   : drawdown,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════

DARK = {
    "bg"     : "#0d1117",
    "panel"  : "#161b22",
    "border" : "#30363d",
    "text"   : "#c9d1d9",
    "muted"  : "#8b949e",
    "blue"   : "#58a6ff",
    "green"  : "#3fb950",
    "red"    : "#f85149",
    "orange" : "#f0883e",
    "purple" : "#bc8cff",
}


def _style_ax(ax):
    ax.set_facecolor(DARK["panel"])
    ax.tick_params(colors=DARK["muted"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(DARK["border"])


def plot_results(df: pd.DataFrame,
                 eq_df: pd.DataFrame,
                 trades_df: pd.DataFrame,
                 metrics: dict,
                 cfg: dict,
                 is_live: bool) -> str:

    print("[4/5] Generating performance dashboard …")

    fast, slow = cfg["fast_sma"], cfg["slow_sma"]
    dd         = metrics["_drawdown_series"]
    completed  = trades_df.dropna(subset=["pnl"])
    data_label = "WTI Futures (Live)" if is_live else "Synthetic WTI (Demo)"

    fig = plt.figure(figsize=(20, 15), facecolor=DARK["bg"])
    title = (f"WTI Crude Oil — SMA({fast}/{slow}) Crossover  ·  Base Strategy  ·  "
             f"{cfg['start_date']} → {cfg['end_date']}  [{data_label}]")
    fig.suptitle(title, fontsize=13, color="white", fontweight="bold", y=0.985)

    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        hspace=0.50, wspace=0.32,
        left=0.07, right=0.97, top=0.96, bottom=0.05,
    )

    # ── A: Price + SMAs + Trade signals (full width) ─────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    _style_ax(ax1)

    ax1.plot(df.index, df["Close"],       color=DARK["blue"],   lw=1.2, label="WTI Close", zorder=2)
    ax1.plot(df.index, df[f"SMA_{fast}"], color=DARK["orange"], lw=1.2, label=f"SMA {fast}", zorder=3)
    ax1.plot(df.index, df[f"SMA_{slow}"], color=DARK["green"],  lw=1.2, label=f"SMA {slow}", zorder=3)

    # Fill between SMAs
    sma_f = df[f"SMA_{fast}"]
    sma_s = df[f"SMA_{slow}"]
    ax1.fill_between(df.index, sma_f, sma_s,
                     where=(sma_f >= sma_s), alpha=0.10, color=DARK["green"])
    ax1.fill_between(df.index, sma_f, sma_s,
                     where=(sma_f <  sma_s), alpha=0.10, color=DARK["red"])

    # Trade markers
    longs  = completed[completed["direction"] == "long"]
    shorts = completed[completed["direction"] == "short"]
    ax1.scatter(pd.to_datetime(longs["entry_date"]),   longs["entry_price"],
                marker="^", color=DARK["green"],  s=65, zorder=5, label="Long entry")
    ax1.scatter(pd.to_datetime(longs["exit_date"]),    longs["exit_price"],
                marker="v", color=DARK["red"],    s=65, zorder=5, label="Long exit")
    ax1.scatter(pd.to_datetime(shorts["entry_date"]),  shorts["entry_price"],
                marker="v", color=DARK["purple"], s=65, zorder=5, label="Short entry")
    ax1.scatter(pd.to_datetime(shorts["exit_date"]),   shorts["exit_price"],
                marker="^", color=DARK["orange"], s=65, zorder=5, label="Short exit")

    ax1.set_ylabel("Price (USD/bbl)", color=DARK["text"], fontsize=9)
    ax1.set_title("Price  ·  SMAs  ·  Trade Signals", color=DARK["muted"], fontsize=10)
    ax1.legend(loc="upper left", fontsize=7.5, facecolor=DARK["panel"],
               edgecolor=DARK["border"], labelcolor=DARK["text"], ncol=4)
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, color=DARK["muted"])

    # ── B: Equity Curve ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    _style_ax(ax2)

    eq = eq_df["equity"]
    bh = cfg["initial_capital"] * eq_df["price"] / eq_df["price"].iloc[0]
    ax2.plot(eq.index, eq / 1e3, color=DARK["blue"],  lw=1.6, label="Strategy",   zorder=3)
    ax2.plot(bh.index, bh / 1e3, color=DARK["muted"], lw=1.0, linestyle="--",
             label="Buy & Hold", zorder=2)
    base = cfg["initial_capital"] / 1e3
    ax2.axhline(base, color=DARK["border"], lw=0.8)
    ax2.fill_between(eq.index, base, eq / 1e3,
                     where=(eq / 1e3 >= base), alpha=0.12, color=DARK["green"])
    ax2.fill_between(eq.index, base, eq / 1e3,
                     where=(eq / 1e3 <  base), alpha=0.12, color=DARK["red"])

    ax2.set_ylabel("Equity ($ k)", color=DARK["text"], fontsize=9)
    ax2.set_title("Equity Curve vs Buy & Hold", color=DARK["muted"], fontsize=10)
    ax2.legend(fontsize=8, facecolor=DARK["panel"], edgecolor=DARK["border"],
               labelcolor=DARK["text"])
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, color=DARK["muted"])

    # ── C: Drawdown ───────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    _style_ax(ax3)

    ax3.fill_between(dd.index, dd, 0, color=DARK["red"], alpha=0.55)
    ax3.plot(dd.index, dd, color=DARK["red"], lw=0.8)
    ax3.axhline(metrics["max_drawdown_pct"], color="#ff7b72", lw=1.0,
                linestyle="--", label=f"Max DD: {metrics['max_drawdown_pct']:.1f}%")
    ax3.set_ylabel("Drawdown (%)", color=DARK["text"], fontsize=9)
    ax3.set_title("Strategy Drawdown", color=DARK["muted"], fontsize=10)
    ax3.legend(fontsize=8, facecolor=DARK["panel"], edgecolor=DARK["border"],
               labelcolor=DARK["text"])
    ax3.xaxis.set_major_locator(mdates.YearLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0, color=DARK["muted"])

    # ── D: PnL Distribution ───────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    _style_ax(ax4)

    pnls = completed["pnl"]
    wins = pnls[pnls > 0]
    loss = pnls[pnls <= 0]
    bins = min(40, max(10, len(pnls) // 3))
    ax4.hist(loss, bins=bins, color=DARK["red"],   alpha=0.75, label=f"Losses ({len(loss)})")
    ax4.hist(wins, bins=bins, color=DARK["green"], alpha=0.75, label=f"Wins ({len(wins)})")
    ax4.axvline(0,           color=DARK["text"],   lw=0.9)
    ax4.axvline(pnls.mean(), color=DARK["orange"], lw=1.2, linestyle="--",
                label=f"Mean ${pnls.mean():,.0f}")
    ax4.set_xlabel("Trade PnL (USD)", color=DARK["muted"], fontsize=9)
    ax4.set_ylabel("Count",           color=DARK["text"],  fontsize=9)
    ax4.set_title("Trade PnL Distribution", color=DARK["muted"], fontsize=10)
    ax4.legend(fontsize=8, facecolor=DARK["panel"], edgecolor=DARK["border"],
               labelcolor=DARK["text"])

    # ── E: Scorecard ──────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    _style_ax(ax5)
    ax5.axis("off")
    ax5.set_title("Performance Scorecard", color=DARK["muted"], fontsize=10, pad=8)

    scorecard = [
        ("RETURNS",               None,                        "section"),
        ("Total Return",          f"{metrics['total_return_pct']:+.2f} %",   "return"),
        ("CAGR",                  f"{metrics['cagr_pct']:+.2f} %",           "return"),
        ("Buy & Hold Return",     f"{metrics['bh_return_pct']:+.2f} %",      "neutral"),
        ("Final Equity",          f"${metrics['final_equity_usd']:,.0f}",    "neutral"),
        ("RISK / REWARD",         None,                        "section"),
        ("Sharpe Ratio",          f"{metrics['sharpe_ratio']:.2f}",          "neutral"),
        ("Sortino Ratio",         f"{metrics['sortino_ratio']:.2f}",         "neutral"),
        ("Calmar Ratio",          f"{metrics['calmar_ratio']:.2f}",          "neutral"),
        ("Max Drawdown",          f"{metrics['max_drawdown_pct']:.2f} %",    "risk"),
        ("Max DD Duration",       f"{metrics['max_dd_duration_days']} days", "risk"),
        ("TRADES",                None,                        "section"),
        ("Total Trades",          f"{metrics['total_trades']}",              "neutral"),
        ("Win Rate",              f"{metrics['win_rate_pct']:.1f} %",        "return"),
        ("Profit Factor",         f"{metrics['profit_factor']:.2f}",         "neutral"),
        ("Avg Win / Avg Loss",    f"${metrics['avg_win_usd']:,.0f}  /  ${abs(metrics['avg_loss_usd']):,.0f}", "neutral"),
        ("Expectancy / Trade",    f"${metrics['expectancy_usd']:,.0f}",      "neutral"),
    ]

    y = 1.02
    row_h = 0.063
    for label, val, kind in scorecard:
        y -= row_h
        if kind == "section":
            ax5.text(0.02, y, label, color=DARK["blue"],  fontsize=8.0,
                     fontweight="bold", transform=ax5.transAxes)
            y -= 0.005
            continue
        ax5.text(0.04, y, label, color=DARK["muted"], fontsize=8.5, transform=ax5.transAxes)
        color = (DARK["green"] if kind == "return"
                 else DARK["red"] if kind == "risk"
                 else DARK["text"])
        ax5.text(0.60, y, val, color=color, fontsize=8.5,
                 fontweight="bold", transform=ax5.transAxes)

    out_path = os.path.join(cfg["output_dir"], "oil_strategy_results.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK["bg"])
    plt.close()
    print(f"    ✓ Dashboard saved → {os.path.abspath(out_path)}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  CONSOLE REPORT  &  CSV EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(metrics: dict, cfg: dict, is_live: bool):
    sep  = "─" * 56
    mode = "LIVE (Yahoo Finance)" if is_live else "DEMO (Synthetic Data)"
    print(f"\n{'═'*56}")
    print(f"  WTI CRUDE OIL  ·  SMA({cfg['fast_sma']}/{cfg['slow_sma']})  ·  BASE STRATEGY")
    print(f"  Data mode : {mode}")
    print(f"{'═'*56}")
    print(f"  Period          : {cfg['start_date']}  →  {cfg['end_date']}")
    print(f"  Duration        : {metrics['n_years']:.1f} years")
    print(f"  Initial Capital : ${cfg['initial_capital']:,.0f}")
    print(sep)
    print("  RETURNS")
    print(f"    Total Return  : {metrics['total_return_pct']:+.2f} %")
    print(f"    CAGR          : {metrics['cagr_pct']:+.2f} %")
    print(f"    Buy & Hold    : {metrics['bh_return_pct']:+.2f} %")
    print(f"    Final Equity  : ${metrics['final_equity_usd']:,.2f}")
    print(sep)
    print("  RISK / REWARD")
    print(f"    Sharpe Ratio  : {metrics['sharpe_ratio']:.2f}")
    print(f"    Sortino Ratio : {metrics['sortino_ratio']:.2f}")
    print(f"    Calmar Ratio  : {metrics['calmar_ratio']:.2f}")
    print(f"    Max Drawdown  : {metrics['max_drawdown_pct']:.2f} %  "
          f"(lasted {metrics['max_dd_duration_days']} bars)")
    print(sep)
    print("  TRADES")
    print(f"    Total         : {metrics['total_trades']}")
    print(f"    Win Rate      : {metrics['win_rate_pct']:.1f} %"
          f"  ({metrics['winning_trades']} W / {metrics['losing_trades']} L)")
    print(f"    Avg Win       : ${metrics['avg_win_usd']:,.0f}")
    print(f"    Avg Loss      : ${metrics['avg_loss_usd']:,.0f}")
    print(f"    Profit Factor : {metrics['profit_factor']:.2f}")
    print(f"    Expectancy    : ${metrics['expectancy_usd']:,.0f} / trade")
    print(f"{'═'*56}\n")


def export_trades(trades_df: pd.DataFrame, cfg: dict) -> str:
    out = os.path.join(cfg["output_dir"], "oil_strategy_trades.csv")
    trades_df.dropna(subset=["pnl"]).to_csv(out, index=False)
    print(f"[5/5] Trade log → {os.path.abspath(out)}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    cfg = CONFIG.copy()
    os.makedirs(cfg["output_dir"], exist_ok=True)

    df, is_live = load_data(cfg)
    df          = compute_indicators(df, cfg)
    eq_df, tr   = run_backtest(df, cfg)
    metrics     = compute_metrics(eq_df, tr, cfg)
    plot_results(df, eq_df, tr, metrics, cfg, is_live)
    print_report(metrics, cfg, is_live)
    export_trades(tr, cfg)

    print("─" * 56)
    print("All outputs written. Ready to compare against complex strategies.")
    print("─" * 56)


if __name__ == "__main__":
    main()