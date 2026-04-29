"""Base strategy: long-only simple moving average (SMA) crossover on close."""

from __future__ import annotations

import pandas as pd


def add_sma_signals(
    close: pd.Series,
    fast_window: int,
    slow_window: int,
) -> pd.DataFrame:
    """
    Compute fast/slow SMAs and a long-only position series.

    Position at date t is 1 if fast_sma(t) > slow_sma(t), else 0.
    Rows before slow_window are NaN for SMAs; position is 0 where SMAs are NaN.
    """
    if fast_window >= slow_window:
        raise ValueError("fast_window must be smaller than slow_window for a classic crossover baseline.")

    out = pd.DataFrame({"close": close.astype(float)})
    out["sma_fast"] = out["close"].rolling(fast_window, min_periods=fast_window).mean()
    out["sma_slow"] = out["close"].rolling(slow_window, min_periods=slow_window).mean()
    valid = out["sma_fast"].notna() & out["sma_slow"].notna()
    out["position"] = 0.0
    out.loc[valid, "position"] = (out.loc[valid, "sma_fast"] > out.loc[valid, "sma_slow"]).astype(float)
    return out
