"""Download and normalize OHLCV oil data via yfinance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


def _repair_nonpositive_bars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Yahoo futures history can include non-positive prints around stress events (e.g. Apr 2020 WTI).
    Forward-fill OHLC from the last valid row so pct-change returns stay finite.
    """
    d = df.copy()
    bad = (d["Close"] <= 0) | d["Close"].isna()
    if not bad.any():
        return d
    d.loc[bad, ["Open", "High", "Low", "Close"]] = np.nan
    d = d.ffill()
    return d.dropna(how="any")


def load_oil_ohlcv(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch daily OHLCV for a commodity symbol (e.g. CL=F for WTI futures).

    Returns a DataFrame indexed by date (timezone-naive) with columns:
    Open, High, Low, Close, Volume (Volume may be 0 for some futures series).
    """
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if df.empty:
        raise ValueError(f"No data returned for {ticker!r} (start={start!r}, end={end!r}).")

    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    expected = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Unexpected columns; missing {missing}. Got: {list(df.columns)}")

    df = df[expected].astype(float)
    df = df.dropna(subset=["Close"])
    df = _repair_nonpositive_bars(df)
    return df
