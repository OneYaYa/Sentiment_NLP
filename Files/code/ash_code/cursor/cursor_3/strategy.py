"""
Walk-forward logistic regression: predict P(next-day return > 0) from lagged returns.

At index k, features use only returns known by end of day k. The model is fit on the prior
`train_window` days only (no lookahead). Position k is 1 if P(up) > threshold, else 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _feature_matrix(close: pd.Series, n_lags: int) -> tuple[pd.DataFrame, pd.Series]:
    """
    Rows aligned with `close.index`. Features at day t: ret[t], ret[t-1], ... ret[t-n_lags+1]
    (ret.shift(0) through ret.shift(n_lags-1)), all known at close t.

    Target y at row t: 1 if next-day return ret[t+1] > 0, else 0.
    """
    c = close.astype(float)
    ret = c.pct_change()
    parts: list[pd.Series] = []
    for L in range(0, n_lags):
        parts.append(ret.shift(L).rename(f"ret_lag{L}"))
    X = pd.concat(parts, axis=1)
    y = (ret.shift(-1) > 0.0).astype(float)
    return X, y


def walk_forward_logistic_positions(
    close: pd.Series,
    train_window: int,
    n_lags: int,
    prob_threshold: float,
    max_iter: int = 1000,
) -> pd.DataFrame:
    """
    For each k >= train_window, fit LogisticRegression on rows [k-train_window, k-1],
    scale features inside the training block, then predict P(up) at k.

    Returns DataFrame with columns: close, prob_up, position (0/1 before execution lag).
    Rows before `train_window` have position 0 and prob_up NaN.
    """
    if train_window < 20:
        raise ValueError("train_window should be at least ~20 for stable logistic fits.")
    if n_lags < 1:
        raise ValueError("n_lags must be >= 1.")

    c = close.astype(float)
    X, y = _feature_matrix(c, n_lags)
    idx = c.index
    n = len(c)
    prob = np.full(n, np.nan, dtype=float)
    pos = np.zeros(n, dtype=float)

    X_np = X.to_numpy(dtype=float)
    y_np = y.to_numpy(dtype=float)

    for k in range(train_window, n):
        lo = k - train_window
        hi = k  # train on [lo, k-1]
        mask = np.ones(hi - lo, dtype=bool)
        x_tr = X_np[lo:hi]
        y_tr = y_np[lo:hi]
        mask &= ~np.isnan(y_tr)
        mask &= ~np.any(np.isnan(x_tr), axis=1)
        if mask.sum() < 10:
            continue
        x_tr = x_tr[mask]
        y_tr = y_tr[mask].astype(int)
        if np.unique(y_tr).size < 2:
            continue
        x_row = X_np[k : k + 1]
        if np.any(np.isnan(x_row)):
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
            continue
        p_up = float(clf.predict_proba(x_row_s)[0, 1])
        prob[k] = p_up
        pos[k] = 1.0 if p_up > prob_threshold else 0.0

    out = pd.DataFrame(
        {
            "close": c.values,
            "prob_up": prob,
            "position": pos,
        },
        index=idx,
    )
    return out
