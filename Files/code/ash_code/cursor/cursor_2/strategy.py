"""Mean-reversion strategy: long-only RSI with hysteresis (oversold entry, overbought exit)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, window: int) -> pd.Series:
    """
    RSI from simple rolling averages of gains and losses (classic SMA-style RSI).
    Returns values in [0, 100] where NaN until window is filled.
    """
    if window < 2:
        raise ValueError("RSI window must be at least 2.")
    c = close.astype(float)
    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi_out = 100.0 - (100.0 / (1.0 + rs))
    return rsi_out


def _rsi_hysteresis_positions(
    rsi_series: pd.Series,
    oversold: float,
    overbought: float,
) -> pd.Series:
    """
    Long-only state machine: flat -> long when RSI <= oversold; long -> flat when RSI >= overbought.
    """
    if oversold >= overbought:
        raise ValueError("oversold must be less than overbought.")

    pos = np.zeros(len(rsi_series), dtype=float)
    state = 0
    for i, r in enumerate(rsi_series.to_numpy()):
        if np.isnan(r):
            pos[i] = 0.0
            continue
        if state == 0 and r <= oversold:
            state = 1
        elif state == 1 and r >= overbought:
            state = 0
        pos[i] = float(state)
    return pd.Series(pos, index=rsi_series.index, dtype=float)


def add_rsi_mean_reversion_signals(
    close: pd.Series,
    rsi_window: int,
    oversold: float,
    overbought: float,
) -> pd.DataFrame:
    """
    Compute RSI and long-only position with hysteresis.

    - Start flat. When RSI drops to oversold or below, enter long.
    - While long, hold until RSI rises to overbought or above, then exit to flat.
    """
    out = pd.DataFrame({"close": close.astype(float)})
    out["rsi"] = rsi(out["close"], rsi_window)
    out["position"] = _rsi_hysteresis_positions(out["rsi"], oversold, overbought)
    return out
