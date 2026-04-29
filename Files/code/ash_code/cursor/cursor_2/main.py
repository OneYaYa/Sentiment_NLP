"""
Oil mean-reversion strategy: long-only RSI with hysteresis (oversold in, overbought out).

Run from the repository `code` directory:
  python ash_code/cursor/cursor_2/main.py

With headless plots (save PNGs only):
  python ash_code/cursor/cursor_2/main.py --no-show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd


def _parse_args() -> argparse.Namespace:
    from config import (
        DEFAULT_END_DATE,
        DEFAULT_RSI_OVERBOUGHT,
        DEFAULT_RSI_OVERSOLD,
        DEFAULT_RSI_WINDOW,
        DEFAULT_START_DATE,
        DEFAULT_TICKER,
    )

    p = argparse.ArgumentParser(description="Mean-reversion RSI backtest on oil (yfinance).")
    p.add_argument(
        "--ticker",
        type=str,
        default=DEFAULT_TICKER,
        help="Yahoo symbol (default USO). Use CL=F (WTI) or BZ=F (Brent) for futures.",
    )
    p.add_argument("--rsi-window", type=int, default=DEFAULT_RSI_WINDOW)
    p.add_argument("--oversold", type=float, default=DEFAULT_RSI_OVERSOLD)
    p.add_argument("--overbought", type=float, default=DEFAULT_RSI_OVERBOUGHT)
    p.add_argument("--start", type=str, default=DEFAULT_START_DATE, help="Start date YYYY-MM-DD.")
    p.add_argument("--end", type=str, default=DEFAULT_END_DATE, help="End date YYYY-MM-DD (optional).")
    p.add_argument("--no-show", action="store_true", help="Save figures only, do not open plot windows.")
    p.add_argument("--output-dir", type=str, default=str(_ROOT / "output"))
    return p.parse_args()


def _count_long_entries(signal_position: pd.Series) -> int:
    """Entries into a long leg: signal goes from flat to long."""
    p = signal_position.fillna(0.0)
    return int(((p.shift(1) == 0) & (p == 1)).sum())


def main() -> None:
    args = _parse_args()
    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")

    from backtest import buy_and_hold_returns, run_long_only_backtest, summarize
    from config import COMMISSION_RATE, INITIAL_CASH, TRADING_DAYS_PER_YEAR
    from data_loader import load_oil_ohlcv
    from plot_results import plot_equity_and_drawdown, plot_price_rsi_position
    from strategy import add_rsi_mean_reversion_signals

    start = args.start
    end = args.end if args.end else None
    out_dir = Path(args.output_dir)
    show_plots = not args.no_show

    print(f"Loading {args.ticker!r} (start={start!r}, end={end!r}) ...")
    ohlcv = load_oil_ohlcv(args.ticker, start=start, end=end)
    close = ohlcv["Close"]

    sig = add_rsi_mean_reversion_signals(
        close,
        rsi_window=args.rsi_window,
        oversold=args.oversold,
        overbought=args.overbought,
    )
    strat_ret, equity_mult, _pos = run_long_only_backtest(
        close,
        sig["position"],
        commission_rate=COMMISSION_RATE,
    )
    bh_ret, bh_equity_mult = buy_and_hold_returns(close)

    strategy_equity = equity_mult * INITIAL_CASH
    bh_equity = bh_equity_mult * INITIAL_CASH

    years = len(strat_ret.dropna()) / TRADING_DAYS_PER_YEAR
    m_strategy = summarize(strat_ret, equity_mult, years=years)
    m_bh = summarize(bh_ret, bh_equity_mult, years=years)

    long_entries = _count_long_entries(sig["position"])

    print()
    print("=== Base strategy: mean reversion (RSI hysteresis) ===")
    print(
        f"Ticker: {args.ticker}  |  RSI({args.rsi_window})  |  "
        f"oversold<={args.oversold:g}  overbought>={args.overbought:g}  |  "
        f"Initial cash: ${INITIAL_CASH:,.0f}"
    )
    print(f"Commission (per side, on notional): {COMMISSION_RATE:.4%}")
    print()
    print(f"{'Metric':<22} {'RSI strategy':>16} {'Buy & hold':>16}")
    print("-" * 56)
    print(f"{'Total return':<22} {m_strategy['total_return']:>15.2%} {m_bh['total_return']:>15.2%}")
    print(f"{'CAGR':<22} {m_strategy['cagr']:>15.2%} {m_bh['cagr']:>15.2%}")
    print(f"{'Vol (ann.)':<22} {m_strategy['vol_annual']:>15.2%} {m_bh['vol_annual']:>15.2%}")
    print(f"{'Sharpe (ann.)':<22} {m_strategy['sharpe']:>16.3f} {m_bh['sharpe']:>16.3f}")
    print(f"{'Max drawdown':<22} {m_strategy['max_drawdown']:>15.2%} {m_bh['max_drawdown']:>15.2%}")
    print(f"{'Long entries (count)':<22} {long_entries:>16} {'—':>16}")
    print()

    title = f"{args.ticker} — RSI MR ({args.oversold:g}/{args.overbought:g}) vs buy & hold"
    eq_path = out_dir / "equity_drawdown.png"
    px_path = out_dir / "price_rsi_position.png"

    plot_df = sig.copy()
    plot_df["position"] = sig["position"].shift(1).fillna(0.0)

    plot_equity_and_drawdown(
        strategy_equity / INITIAL_CASH,
        bh_equity / INITIAL_CASH,
        title,
        save_path=eq_path,
        show=show_plots,
    )
    plot_price_rsi_position(
        plot_df,
        oversold=args.oversold,
        overbought=args.overbought,
        title=f"{args.ticker} — price and RSI (green = long, executed)",
        save_path=px_path,
        show=show_plots,
    )

    print(f"Saved: {eq_path}")
    print(f"Saved: {px_path}")


if __name__ == "__main__":
    main()
