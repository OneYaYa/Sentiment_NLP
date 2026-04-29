"""
Oil strategy: walk-forward random forest on lagged returns (predict next-day up move).

Run from the repository `code` directory:
  python ash_code/cursor/cursor_4/main.py

With headless plots (save PNGs only):
  python ash_code/cursor/cursor_4/main.py --no-show
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
        DEFAULT_MAX_DEPTH,
        DEFAULT_N_ESTIMATORS,
        DEFAULT_N_LAGS,
        DEFAULT_PROB_THRESHOLD,
        DEFAULT_RANDOM_STATE,
        DEFAULT_START_DATE,
        DEFAULT_TICKER,
        DEFAULT_TRAIN_WINDOW,
    )

    p = argparse.ArgumentParser(description="Walk-forward random forest backtest on oil (yfinance).")
    p.add_argument(
        "--ticker",
        type=str,
        default=DEFAULT_TICKER,
        help="Yahoo symbol (default USO). Use CL=F (WTI) or BZ=F (Brent) for futures.",
    )
    p.add_argument("--train-window", type=int, default=DEFAULT_TRAIN_WINDOW)
    p.add_argument("--n-lags", type=int, default=DEFAULT_N_LAGS)
    p.add_argument("--prob-threshold", type=float, default=DEFAULT_PROB_THRESHOLD)
    p.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    p.add_argument(
        "--max-depth",
        type=int,
        default=DEFAULT_MAX_DEPTH,
        help="Max tree depth (default from config). Use a large value (e.g. 32) for deeper trees.",
    )
    p.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
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
    from plot_results import plot_equity_and_drawdown, plot_price_prob_position
    from strategy import walk_forward_random_forest_positions

    max_depth = None if args.max_depth <= 0 else args.max_depth

    start = args.start
    end = args.end if args.end else None
    out_dir = Path(args.output_dir)
    show_plots = not args.no_show

    print(f"Loading {args.ticker!r} (start={start!r}, end={end!r}) ...")
    ohlcv = load_oil_ohlcv(args.ticker, start=start, end=end)
    close = ohlcv["Close"]

    sig = walk_forward_random_forest_positions(
        close,
        train_window=args.train_window,
        n_lags=args.n_lags,
        prob_threshold=args.prob_threshold,
        n_estimators=args.n_estimators,
        max_depth=max_depth,
        random_state=args.random_state,
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

    depth_str = "None" if max_depth is None else str(max_depth)
    print()
    print("=== Base strategy: walk-forward random forest (directional) ===")
    print(
        f"Ticker: {args.ticker}  |  train_window={args.train_window}  n_lags={args.n_lags}  "
        f"P(up)>{args.prob_threshold:g}  |  trees={args.n_estimators}  max_depth={depth_str}  "
        f"|  Initial cash: ${INITIAL_CASH:,.0f}"
    )
    print(f"Commission (per side, on notional): {COMMISSION_RATE:.4%}")
    print()
    print(f"{'Metric':<22} {'Random forest':>16} {'Buy & hold':>16}")
    print("-" * 56)
    print(f"{'Total return':<22} {m_strategy['total_return']:>15.2%} {m_bh['total_return']:>15.2%}")
    print(f"{'CAGR':<22} {m_strategy['cagr']:>15.2%} {m_bh['cagr']:>15.2%}")
    print(f"{'Vol (ann.)':<22} {m_strategy['vol_annual']:>15.2%} {m_bh['vol_annual']:>15.2%}")
    print(f"{'Sharpe (ann.)':<22} {m_strategy['sharpe']:>16.3f} {m_bh['sharpe']:>16.3f}")
    print(f"{'Max drawdown':<22} {m_strategy['max_drawdown']:>15.2%} {m_bh['max_drawdown']:>15.2%}")
    print(f"{'Long entries (count)':<22} {long_entries:>16} {'—':>16}")
    print()

    title = f"{args.ticker} — RF (lags={args.n_lags}, W={args.train_window}, T={args.n_estimators}) vs buy & hold"
    eq_path = out_dir / "equity_drawdown.png"
    px_path = out_dir / "price_prob_position.png"

    plot_df = sig.copy()
    plot_df["position"] = sig["position"].shift(1).fillna(0.0)
    plot_df["prob_up"] = sig["prob_up"].shift(1)

    plot_equity_and_drawdown(
        strategy_equity / INITIAL_CASH,
        bh_equity / INITIAL_CASH,
        title,
        save_path=eq_path,
        show=show_plots,
    )
    plot_price_prob_position(
        plot_df,
        prob_threshold=args.prob_threshold,
        title=f"{args.ticker} — price and P(up) (green = long, executed)",
        save_path=px_path,
        show=show_plots,
    )

    print(f"Saved: {eq_path}")
    print(f"Saved: {px_path}")


if __name__ == "__main__":
    main()
