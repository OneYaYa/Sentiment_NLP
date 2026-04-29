"""
Regime switching: if ADX >= threshold → trend regime → SMA crossover; else → RSI hysteresis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Average Directional Index (Wilder-style smoothing via EWM alpha=1/period).
    Returns ADX in [0, 100] with NaNs until enough history.
    """
    if period < 2:
        raise ValueError("ADX period must be at least 2.")
    h = high.astype(float)
    l = low.astype(float)
    c = close.astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)

    up_move = h.diff()
    down_move = -l.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=h.index)
    minus_dm = pd.Series(minus_dm, index=h.index)

    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr.replace(0.0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr.replace(0.0, np.nan)
    denom = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denom
    adx_out = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    return adx_out.clip(lower=0.0, upper=100.0)


def rsi(close: pd.Series, window: int) -> pd.Series:
    c = close.astype(float)
    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _rsi_hysteresis_positions(rsi_series: pd.Series, oversold: float, overbought: float) -> pd.Series:
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


def add_sma_positions(close: pd.Series, fast_window: int, slow_window: int) -> pd.Series:
    if fast_window >= slow_window:
        raise ValueError("fast_window must be smaller than slow_window.")
    c = close.astype(float)
    sma_f = c.rolling(fast_window, min_periods=fast_window).mean()
    sma_s = c.rolling(slow_window, min_periods=slow_window).mean()
    valid = sma_f.notna() & sma_s.notna()
    pos = pd.Series(0.0, index=c.index)
    pos.loc[valid] = (sma_f.loc[valid] > sma_s.loc[valid]).astype(float)
    return pos


def add_regime_switching_signals(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    fast_window: int,
    slow_window: int,
    rsi_window: int,
    oversold: float,
    overbought: float,
    adx_period: int,
    adx_threshold: float,
) -> pd.DataFrame:
    """
    trend_regime: ADX >= adx_threshold → use SMA long-only crossover.
    chop_regime: ADX < adx_threshold → use RSI hysteresis.
    """
    c = close.astype(float)
    adx_s = adx(high, low, c, period=adx_period)
    sma_pos = add_sma_positions(c, fast_window, slow_window)
    rsi_s = rsi(c, rsi_window)
    rsi_pos = _rsi_hysteresis_positions(rsi_s, oversold, overbought)

    trend = (adx_s >= adx_threshold).fillna(False)

    pos = np.where(trend.to_numpy(), sma_pos.to_numpy(), rsi_pos.to_numpy()).astype(float)
    out = pd.DataFrame(
        {
            "close": c,
            "adx": adx_s,
            "rsi": rsi_s,
            "sma_position": sma_pos,
            "rsi_position": rsi_pos,
            "trend_regime": trend.astype(float),
            "position": pos,
        },
        index=c.index,
    )
    return out
