from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd


# ============================================================
# Config
# ============================================================

@dataclass
class SentimentDirectionalConfig:
    # sentiment combination weights
    w1: float = 0.2
    w3: float = 0.3
    w7: float = 0.5

    # rolling z-score window for sentiment signal
    z_window: int = 120
    z_min_periods: int = 60

    # entry/exit thresholds (hysteresis)
    entry_z: float = 0.5
    exit_z: float = 0.1

    # volatility targeting
    target_vol: float = 0.01  # 1% daily vol target
    w_max: float = 2.0

    # cost model
    cost_per_unit: float = 0.0002  # 2 bps per unit turnover

    # safety filter
    enable_vol_spike_filter: bool = True
    vol_spike_mult: float = 2.5

    # execution convention
    next_day_execution: bool = True

    # contrarian: if True, long when sentiment low (Z < -entry_z), short when high (Z > entry_z)
    contrarian: bool = False


# ============================================================
# Utilities
# ============================================================

def _required_cols_check(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _rolling_zscore(x: pd.Series, window: int, min_periods: int) -> pd.Series:
    mu = x.rolling(window=window, min_periods=min_periods).mean()
    sd = x.rolling(window=window, min_periods=min_periods).std(ddof=0).replace(0.0, np.nan)
    return (x - mu) / sd


def load_yahoo_oil_csv(
    csv_path: str,
    *,
    date_col_in_csv: str | None = None,
    recompute_returns: bool = True,
) -> pd.DataFrame:
    """
    Loads your Yahoo-style CSV and standardizes columns to:
      date, open, high, low, close, volume, ret
    """
    df = pd.read_csv(csv_path)
    date_col = date_col_in_csv
    if date_col is None:
        date_col = "Date" if "Date" in df.columns else ("date" if "date" in df.columns else None)
    if date_col is None:
        raise ValueError(f"Expected date column 'Date' or 'date' in {csv_path}, got {list(df.columns)}")

    # Normalize column names (Date/date -> date, Close -> close, etc.)
    renames = {date_col: "date"}
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            renames[c] = c.lower()
    df = df.rename(columns=renames)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # If your CSV already contains ret, you can set recompute_returns=False
    if recompute_returns:
        if "close" not in df.columns:
            raise ValueError("Cannot recompute returns without 'close' column.")
        df["ret"] = df["close"].pct_change()

    return df


def add_rolling_volatility(
    df: pd.DataFrame,
    *,
    ret_col: str = "ret",
    vol20_window: int = 20,
    vol60_window: int = 60,
    min_periods20: Optional[int] = None,
    min_periods60: Optional[int] = None,
) -> pd.DataFrame:
    out = df.copy()
    _required_cols_check(out, [ret_col])

    mp20 = vol20_window if min_periods20 is None else min_periods20
    mp60 = vol60_window if min_periods60 is None else min_periods60

    out["vol20"] = out[ret_col].rolling(vol20_window, min_periods=mp20).std(ddof=0)
    out["vol60"] = out[ret_col].rolling(vol60_window, min_periods=mp60).std(ddof=0)
    return out


# ============================================================
# Daily sentiment from CSV (real data)
# ============================================================

def load_daily_sentiment_features(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load daily sentiment scores from CSV (date, sentiment_score, article_count)
    and produce features expected by build_sentiment_signal:
      date, oil_sent_1d, oil_sent_3d, oil_sent_7d, oil_news_count_7d
    """
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    s = df["sentiment_score"].fillna(0.0)
    c = df["article_count"].fillna(0.0).astype(int)

    out = pd.DataFrame(
        {
            "date": df["date"],
            "oil_sent_1d": s,
            "oil_sent_3d": s.rolling(3, min_periods=1).mean(),
            "oil_sent_7d": s.rolling(7, min_periods=1).mean(),
            "oil_news_count_7d": c.rolling(7, min_periods=1).sum().astype(int),
        }
    )
    return out


def load_cumulative_sentiment_features(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load cumulative sentiment (CSS) from CSV (date, css_7d_exp, optional article_count)
    and produce features expected by build_sentiment_signal:
      date, oil_sent_1d, oil_sent_3d, oil_sent_7d, oil_news_count_7d

    Uses css_7d_exp for all three sentiment columns so S_raw = w1*css + w3*css + w7*css = css.
    """
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if "css_7d_exp" not in df.columns:
        raise ValueError(f"CSV must contain 'css_7d_exp'. Got columns: {list(df.columns)}")
    css = df["css_7d_exp"].astype(float)
    # Forward-fill NaN CSS (e.g. first 6 days) then fill any remaining with 0
    css = css.ffill().fillna(0.0)

    if "article_count" in df.columns:
        c = df["article_count"].fillna(0.0).astype(int)
        news_7d = c.rolling(7, min_periods=1).sum().astype(int)
    else:
        news_7d = pd.Series(1, index=df.index, dtype=int)

    out = pd.DataFrame(
        {
            "date": df["date"],
            "oil_sent_1d": css,
            "oil_sent_3d": css,
            "oil_sent_7d": css,
            "oil_news_count_7d": news_7d,
        }
    )
    return out


# ============================================================
# Random sentiment generator (fallback / testing)
# ============================================================

def generate_random_oil_sentiment_features(
    dates: pd.Series,
    *,
    seed: int = 7,
    phi: float = 0.92,
    shock_scale: float = 0.20,
    base_news_lambda: float = 18.0,
) -> pd.DataFrame:
    """
    Generates a *realistic-ish* daily sentiment process:
    - AR(1) latent sentiment (slow decay, good for oil narratives)
    - rolling means to get 1d/3d/7d
    - Poisson news counts (with mild co-movement with sentiment magnitude)

    Output columns:
      date, oil_sent_1d, oil_sent_3d, oil_sent_7d, oil_news_count_7d
    """
    d = pd.to_datetime(dates).sort_values().reset_index(drop=True)
    n = len(d)

    rng = np.random.default_rng(seed)

    # AR(1) latent process with occasional shocks
    eps = rng.normal(0.0, shock_scale, size=n)
    latent = np.zeros(n, dtype=float)
    for i in range(1, n):
        latent[i] = phi * latent[i - 1] + eps[i]

    # squash to [-1, 1] like sentiment
    latent = np.tanh(latent)

    s = pd.Series(latent)

    oil_sent_1d = s
    oil_sent_3d = s.rolling(3, min_periods=1).mean()
    oil_sent_7d = s.rolling(7, min_periods=1).mean()

    # News counts: higher around high |sentiment| days
    intensity = np.clip(np.abs(oil_sent_7d.to_numpy()) * 10.0, 0.0, 10.0)
    lam = np.clip(base_news_lambda + intensity, 1.0, None)
    news_count_daily = rng.poisson(lam=lam, size=n).astype(int)

    oil_news_count_7d = pd.Series(news_count_daily).rolling(7, min_periods=1).sum().astype(int)

    out = pd.DataFrame(
        {
            "date": d,
            "oil_sent_1d": oil_sent_1d.to_numpy(),
            "oil_sent_3d": oil_sent_3d.to_numpy(),
            "oil_sent_7d": oil_sent_7d.to_numpy(),
            "oil_news_count_7d": oil_news_count_7d.to_numpy(),
        }
    )
    return out


def merge_sentiment_features(market_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    m = market_df.copy()
    s = sentiment_df.copy()
    m["date"] = pd.to_datetime(m["date"])
    s["date"] = pd.to_datetime(s["date"])
    out = m.merge(s, on="date", how="left")
    return out


# ============================================================
# Strategy logic
# ============================================================

def build_sentiment_signal(
    df: pd.DataFrame,
    cfg: SentimentDirectionalConfig,
    s1_col: str = "oil_sent_1d",
    s3_col: str = "oil_sent_3d",
    s7_col: str = "oil_sent_7d",
    count7_col: str = "oil_news_count_7d",
) -> pd.DataFrame:
    _required_cols_check(df, [s1_col, s3_col, s7_col, count7_col])

    out = df.copy()

    S_raw = cfg.w1 * out[s1_col] + cfg.w3 * out[s3_col] + cfg.w7 * out[s7_col]
    intensity = np.log1p(out[count7_col].clip(lower=0))
    out["S_raw"] = S_raw
    out["S_intensity"] = S_raw * intensity
    out["Z"] = _rolling_zscore(out["S_intensity"], window=cfg.z_window, min_periods=cfg.z_min_periods)

    return out


def run_directional_strategy(
    df: pd.DataFrame,
    cfg: SentimentDirectionalConfig,
    *,
    date_col: str = "date",
    ret_col: str = "ret",
    vol20_col: str = "vol20",
    vol60_col: str = "vol60",
) -> pd.DataFrame:
    req = [date_col, ret_col, vol20_col]
    if cfg.enable_vol_spike_filter:
        req.append(vol60_col)
    _required_cols_check(df, req)

    d0 = df.copy()
    d0[date_col] = pd.to_datetime(d0[date_col])
    d0 = d0.sort_values(date_col).reset_index(drop=True)

    d1 = build_sentiment_signal(d0, cfg)

    if cfg.enable_vol_spike_filter:
        vol_ok = d1[vol20_col] <= cfg.vol_spike_mult * d1[vol60_col]
    else:
        vol_ok = pd.Series(True, index=d1.index)

    # state machine
    pos_state = np.zeros(len(d1), dtype=np.int8)  # -1/0/+1
    weight = np.zeros(len(d1), dtype=float)

    prev_state = 0
    Z = d1["Z"].to_numpy(dtype=float)
    vol20 = d1[vol20_col].to_numpy(dtype=float)
    vol_ok_np = vol_ok.to_numpy(dtype=bool)

    for i in range(len(d1)):
        z = Z[i]
        state = prev_state

        if not np.isnan(z):
            if prev_state == 0:
                if cfg.contrarian:
                    if z > cfg.entry_z and vol_ok_np[i]:
                        state = -1
                    elif z < -cfg.entry_z and vol_ok_np[i]:
                        state = +1
                else:
                    if z > cfg.entry_z and vol_ok_np[i]:
                        state = +1
                    elif z < -cfg.entry_z and vol_ok_np[i]:
                        state = -1
            elif prev_state == +1:
                if cfg.contrarian:
                    if z > -cfg.exit_z:
                        state = 0
                else:
                    if z < cfg.exit_z:
                        state = 0
            elif prev_state == -1:
                if cfg.contrarian:
                    if z < cfg.exit_z:
                        state = 0
                else:
                    if z > -cfg.exit_z:
                        state = 0

        if state == 0 or np.isnan(vol20[i]) or vol20[i] <= 0:
            w = 0.0
        else:
            w = float(state) * min(cfg.target_vol / float(vol20[i]), cfg.w_max)

        pos_state[i] = state
        weight[i] = w
        prev_state = state

    d1["pos_state"] = pos_state
    d1["weight"] = weight

    # costs
    d1["turnover"] = d1["weight"].diff().abs().fillna(d1["weight"].abs())
    d1["cost"] = d1["turnover"] * cfg.cost_per_unit

    # execution
    if cfg.next_day_execution:
        d1["weight_applied"] = d1["weight"].shift(1).fillna(0.0)
        d1["cost_applied"] = d1["cost"].shift(1).fillna(0.0)
    else:
        d1["weight_applied"] = d1["weight"]
        d1["cost_applied"] = d1["cost"]

    d1["strat_ret"] = d1["weight_applied"] * d1[ret_col] - d1["cost_applied"]
    d1["equity"] = (1.0 + d1["strat_ret"].fillna(0.0)).cumprod()

    return d1


def performance_summary(strat_df: pd.DataFrame, strat_ret_col: str = "strat_ret") -> dict:
    r = strat_df[strat_ret_col].dropna().astype(float)
    if r.empty:
        return {"error": "No returns to summarize."}

    ann = 252.0
    mu = r.mean()
    sd = r.std(ddof=0)
    sharpe = (mu / sd) * np.sqrt(ann) if sd > 0 else np.nan

    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    dd = (equity / peak) - 1.0

    total_ret = float(equity.iloc[-1] - 1.0)
    ann_ret = float((1.0 + total_ret) ** (ann / len(r)) - 1.0) if len(r) > 0 else np.nan
    ann_vol = float(sd * np.sqrt(ann))

    return {
        "n_days": int(len(r)),
        "total_return": total_ret,
        "annualized_return": ann_ret,
        "annualized_vol": ann_vol,
        "sharpe": float(sharpe),
        "max_drawdown": float(dd.min()),
        "avg_daily_turnover": float(strat_df["turnover"].mean()) if "turnover" in strat_df else np.nan,
        "avg_daily_cost": float(strat_df["cost"].mean()) if "cost" in strat_df else np.nan,
    }


def optimization_score(
    stats: dict,
    turnover_penalty: float = 0.4,
    param_penalty: float = 0.0,
    entry_z: float | None = None,
    z_window: int | None = None,
) -> float:
    """
    Score = Sharpe - |Max Drawdown| - turnover_penalty * avg_daily_turnover - param_penalty.
    If param_penalty > 0 and entry_z/z_window given, adds penalty for extreme params.
    Used for parameter optimization (maximize this).
    """
    if "error" in stats:
        return -np.inf
    sharpe = stats.get("sharpe") or 0.0
    max_dd = stats.get("max_drawdown") or 0.0
    turnover = stats.get("avg_daily_turnover") or 0.0
    if np.isnan(sharpe):
        sharpe = 0.0
    if np.isnan(max_dd):
        max_dd = 0.0
    if np.isnan(turnover):
        turnover = 0.0
    score = float(sharpe) - abs(float(max_dd)) - turnover_penalty * float(turnover)
    if param_penalty > 0 and entry_z is not None:
        # Penalize distance of entry_z from 0.5 (neutral)
        score -= param_penalty * abs(entry_z - 0.5)
    if param_penalty > 0 and z_window is not None:
        # Penalize very short or very long windows (favor 120)
        score -= param_penalty * 0.01 * abs(z_window - 120)
    return score


# ============================================================
# High-level convenience: run end-to-end for one instrument
# ============================================================

def build_and_backtest_one_instrument(
    market_csv_path: str,
    *,
    cfg: Optional[SentimentDirectionalConfig] = None,
    sentiment_csv_path: Optional[Union[str, Path]] = None,
    sentiment_df: Optional[pd.DataFrame] = None,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Loads market CSV, merges with sentiment, runs strategy, returns (bt_df, stats).

    Sentiment source (use exactly one):
      - sentiment_csv_path: path to daily_sentiment_scores CSV
      - sentiment_df: DataFrame with oil_sent_1d, oil_sent_3d, oil_sent_7d, oil_news_count_7d, date
      - seed: if neither above, use random sentiment (for testing)
    """
    if cfg is None:
        cfg = SentimentDirectionalConfig()

    mkt = load_yahoo_oil_csv(market_csv_path, recompute_returns=True)
    mkt = add_rolling_volatility(mkt)

    if sentiment_df is not None:
        sent = sentiment_df
    elif sentiment_csv_path is not None:
        sent = load_daily_sentiment_features(sentiment_csv_path)
    elif seed is not None:
        sent = generate_random_oil_sentiment_features(mkt["date"], seed=seed)
    else:
        raise ValueError("Provide sentiment_csv_path, sentiment_df, or seed.")

    df = merge_sentiment_features(mkt, sent)
    # Forward-fill sentiment on market days without news (e.g. weekends)
    for col in ["oil_sent_1d", "oil_sent_3d", "oil_sent_7d", "oil_news_count_7d"]:
        if col in df.columns:
            df[col] = df[col].ffill().fillna(0.0)

    bt = run_directional_strategy(df, cfg)
    stats = performance_summary(bt)
    return bt, stats
