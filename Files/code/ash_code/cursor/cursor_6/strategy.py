"""
Meta-labeling on SMA: go long only when the primary SMA says long AND a walk-forward
logistic model predicts P(next-day return > 0) > threshold (trained on past days where SMA was long).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def add_sma_signals(
    close: pd.Series,
    fast_window: int,
    slow_window: int,
) -> pd.DataFrame:
    if fast_window >= slow_window:
        raise ValueError("fast_window must be smaller than slow_window.")
    out = pd.DataFrame({"close": close.astype(float)})
    out["sma_fast"] = out["close"].rolling(fast_window, min_periods=fast_window).mean()
    out["sma_slow"] = out["close"].rolling(slow_window, min_periods=slow_window).mean()
    valid = out["sma_fast"].notna() & out["sma_slow"].notna()
    out["sma_long"] = 0.0
    out.loc[valid, "sma_long"] = (out.loc[valid, "sma_fast"] > out.loc[valid, "sma_slow"]).astype(float)
    return out


def _feature_matrix(close: pd.Series, n_lags: int) -> pd.DataFrame:
    c = close.astype(float)
    ret = c.pct_change()
    parts: list[pd.Series] = []
    for L in range(0, n_lags):
        parts.append(ret.shift(L).rename(f"ret_lag{L}"))
    return pd.concat(parts, axis=1)


def walk_forward_meta_sma_positions(
    close: pd.Series,
    fast_window: int,
    slow_window: int,
    train_window: int,
    n_lags: int,
    prob_threshold: float,
    max_iter: int = 1000,
) -> pd.DataFrame:
    """
    Primary: long when fast SMA > slow SMA.
    Meta: when primary long, long only if logistic P(next return > 0) > threshold,
    trained on prior train_window rows **where primary was long** (and labels observed).

    When primary is flat, position is 0. When primary is long but meta cannot train/predict,
    default to primary (long = 1).
    """
    if train_window < 30:
        raise ValueError("train_window should be at least ~30 for meta fits.")
    if n_lags < 1:
        raise ValueError("n_lags must be >= 1.")

    c = close.astype(float)
    sma_df = add_sma_signals(c, fast_window, slow_window)
    sma_long = sma_df["sma_long"].to_numpy(dtype=float)
    X = _feature_matrix(c, n_lags)
    ret = c.pct_change()
    y_up = (ret.shift(-1) > 0.0).to_numpy(dtype=float)

    n = len(c)
    prob = np.full(n, np.nan, dtype=float)
    pos = np.zeros(n, dtype=float)
    X_np = X.to_numpy(dtype=float)

    # Before the rolling window exists, follow the primary SMA signal only.
    for k in range(0, min(train_window, n)):
        pos[k] = float(sma_long[k] >= 0.5)

    for k in range(train_window, n):
        if sma_long[k] < 0.5:
            pos[k] = 0.0
            continue

        lo = k - train_window
        hi = k
        train_idx = np.arange(lo, hi)
        mask = sma_long[train_idx] >= 0.5
        mask &= ~np.isnan(y_up[train_idx])
        mask &= ~np.isnan(X_np[train_idx]).any(axis=1)
        sel = train_idx[mask]
        if sel.size < 15:
            pos[k] = 1.0
            prob[k] = np.nan
            continue

        x_tr = X_np[sel]
        y_tr = y_up[sel].astype(int)
        if np.unique(y_tr).size < 2:
            pos[k] = 1.0
            prob[k] = np.nan
            continue

        x_row = X_np[k : k + 1]
        if np.any(np.isnan(x_row)):
            pos[k] = 1.0
            continue

        scaler = StandardScaler()
        x_tr_s = scaler.fit_transform(x_tr)
        x_row_s = scaler.transform(x_row)
        clf = LogisticRegression(
            max_iter=max_iter,
            class_weight="balanced",
            solver="lbfgs",
        )
        try:
            clf.fit(x_tr_s, y_tr)
        except ValueError:
            pos[k] = 1.0
            continue

        p_up = float(clf.predict_proba(x_row_s)[0, 1])
        prob[k] = p_up
        pos[k] = 1.0 if p_up > prob_threshold else 0.0

    out = sma_df.copy()
    out["prob_up"] = prob
    out["position"] = pos
    out.rename(columns={"sma_long": "primary_long"}, inplace=True)
    return out
