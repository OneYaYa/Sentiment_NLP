"""
Quantile + medium-horizon strategy on Crudebert CSS.

Signal: CSS normalized by previous 60-day mean/std. Long when in Q1 (>= p90),
short when in Q5 (<= p10). Hold for a fixed number of days then exit.
Uses next-day execution and vol targeting. Compatible with the same
walk-forward backtest structure as the z-score strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

# Reuse market loading and performance from original strategy
from sentiment_strategy import (
    add_rolling_volatility,
    load_yahoo_oil_csv,
    performance_summary,
)


# ============================================================
# Config
# ============================================================

@dataclass
class QuantileStrategyConfig:
    """Configuration for quantile + medium-horizon strategy."""

    hold_days: int = 21  # hold position for this many days then exit
    target_vol: float = 0.01  # daily vol target for position sizing
    w_max: float = 2.0  # max absolute weight
    cost_per_unit: float = 0.0002  # cost per unit turnover (e.g. 2 bps)
    next_day_execution: bool = True  # signal at t -> position for t+1
    enable_vol_spike_filter: bool = True
    vol_spike_mult: float = 2.5  # skip sizing when vol20 > this * vol60


# ============================================================
# CSS normalization and quantiles
# ============================================================

def normalize_css_previous_60d(css: pd.Series, window: int = 60) -> pd.Series:
    """
    Normalize CSS by previous 60-day mean and std.
    At t: use mean and std of css at t-60..t-1; (css_t - mean) / std.
    """
    prev = css.shift(1)
    mu = prev.rolling(window=window, min_periods=window).mean()
    sig = prev.rolling(window=window, min_periods=window).std(ddof=0)
    sig = sig.replace(0.0, np.nan)
    return (css - mu) / sig


def compute_quantile_thresholds(
    normalized_css: pd.Series,
    low_pct: float = 10.0,
    high_pct: float = 90.0,
) -> tuple[float, float]:
    """
    Compute percentile thresholds from a series (e.g. train-period normalized CSS).
    Returns (p_low, p_high) e.g. (p10, p90) for Q5 and Q1.
    """
    valid = normalized_css.dropna()
    if len(valid) < 2:
        return np.nan, np.nan
    p_low = float(np.percentile(valid, low_pct))
    p_high = float(np.percentile(valid, high_pct))
    return p_low, p_high


# ============================================================
# Strategy
# ============================================================

def run_quantile_strategy(
    df: pd.DataFrame,
    p10: float,
    p90: float,
    cfg: QuantileStrategyConfig,
    *,
    date_col: str = "date",
    ret_col: str = "ret",
    vol20_col: str = "vol20",
    vol60_col: str = "vol60",
    css_col: str = "css_7d_exp",
) -> pd.DataFrame:
    """
    Run quantile + medium-horizon strategy.

    Expects df with date, ret, vol20, vol60, and css_col (e.g. css_7d_exp).
    Uses p10 and p90 (from train period) to define Q5 and Q1. Enters long
    when normalized CSS >= p90, short when <= p10; holds for hold_days then exits.
    """
    req = [date_col, ret_col, vol20_col, css_col]
    if cfg.enable_vol_spike_filter:
        req.append(vol60_col)
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d = d.sort_values(date_col).reset_index(drop=True)

    # Normalize CSS by previous 60-day mean/std
    d["css_norm"] = normalize_css_previous_60d(d[css_col].astype(float), window=60)

    if cfg.enable_vol_spike_filter:
        vol_ok = d[vol20_col] <= cfg.vol_spike_mult * d[vol60_col]
    else:
        vol_ok = pd.Series(True, index=d.index)

    n = len(d)
    pos_state = np.zeros(n, dtype=np.int8)
    weight = np.zeros(n, dtype=float)

    # State: current position (-1, 0, 1) and days held in that position
    current_pos = 0
    days_held = 0

    css_norm = d["css_norm"].to_numpy(dtype=float)
    vol20 = d[vol20_col].to_numpy(dtype=float)
    vol_ok_np = vol_ok.to_numpy(dtype=bool)

    for i in range(n):
        # Position we hold *during* day i was decided at end of day i-1
        if i == 0:
            pos_state[i] = 0
            weight[i] = 0.0
        else:
            c = css_norm[i - 1]  # signal at end of previous day
            next_pos = current_pos
            next_days_held = days_held

            if current_pos != 0:
                next_days_held = days_held + 1
                if next_days_held >= cfg.hold_days:
                    next_pos = 0
                    next_days_held = 0
                else:
                    next_pos = current_pos
            else:
                if not (np.isfinite(c) and vol_ok_np[i - 1]):
                    next_pos = 0
                elif c >= p90:
                    next_pos = 1
                    next_days_held = 1
                elif c <= p10:
                    next_pos = -1
                    next_days_held = 1
                else:
                    next_pos = 0
                    next_days_held = 0

            pos_state[i] = next_pos
            current_pos = next_pos
            days_held = next_days_held

            p = pos_state[i]
            if p == 0 or not np.isfinite(vol20[i]) or vol20[i] <= 0:
                w = 0.0
            else:
                w = float(p) * min(cfg.target_vol / float(vol20[i]), cfg.w_max)
            weight[i] = w

    d["pos_state"] = pos_state
    d["weight"] = weight

    # Costs and execution (same convention as sentiment_strategy)
    d["turnover"] = d["weight"].diff().abs()
    d["turnover"] = d["turnover"].fillna(d["weight"].abs())
    d["cost"] = d["turnover"] * cfg.cost_per_unit

    if cfg.next_day_execution:
        d["weight_applied"] = d["weight"].shift(1).fillna(0.0)
        d["cost_applied"] = d["cost"].shift(1).fillna(0.0)
    else:
        d["weight_applied"] = d["weight"]
        d["cost_applied"] = d["cost"]

    d["strat_ret"] = d["weight_applied"] * d[ret_col] - d["cost_applied"]
    d["equity"] = (1.0 + d["strat_ret"].fillna(0.0)).cumprod()

    return d


# ============================================================
# Data loading for backtest (CSS + market)
# ============================================================

def load_css_and_market(
    market_csv: Union[str, Path],
    css_csv: Union[str, Path],
) -> pd.DataFrame:
    """
    Load Crudebert cumulative sentiment CSV and market CSV, merge on date,
    add rolling volatility. Forward-fill CSS on market dates.
    """
    mkt = load_yahoo_oil_csv(str(market_csv), recompute_returns=True)
    mkt = add_rolling_volatility(mkt)

    css_df = pd.read_csv(css_csv)
    css_df["date"] = pd.to_datetime(css_df["date"])
    css_df = css_df[["date", "css_7d_exp"]].copy()

    mkt["date"] = pd.to_datetime(mkt["date"])
    out = mkt.merge(css_df, on="date", how="left")
    out["css_7d_exp"] = out["css_7d_exp"].ffill()
    return out
