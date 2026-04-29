"""
Compute a cumulative sentiment score (CSS_t) from daily sentiment scores.

For each day t, the cumulative score is defined as:

    CSS_t = SV_t + sum_{i=1}^6 exp(-i/7) * SV_{t-i}

where SV_t is the daily sentiment score on day t. This is an
exponentially-weighted sum over the current day and the previous 6 days.

Input CSV (same format as daily_sentiment_scores_*.csv):
  - date
  - sentiment_score
  - article_count

Output CSV adds one column:
  - css_7d_exp   (the cumulative sentiment score)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "daily_sentiment_scores_2017_2024.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "daily_cumulative_sentiment_scores_2017_2024.csv"


def compute_css_7d_exp(sentiment: pd.Series, decay: float = 7.0) -> pd.Series:
    """
    Compute CSS_t = SV_t + sum_{i=1}^6 exp(-i/decay) * SV_{t-i}.

    Parameters
    ----------
    sentiment : pd.Series
        Daily sentiment scores indexed by row (assumed sorted by date).
    decay : float, optional
        Decay parameter in the exponential weights, default 7.0
        (matching exp(-i/7) in the provided formula).

    Returns
    -------
    pd.Series
        CSS values aligned with `sentiment`. The first 6 observations
        will be NaN because a full 7-day history is not available.
    """
    s = sentiment.astype(float).to_numpy()
    n = len(s)
    if n == 0:
        return pd.Series([], dtype=float)

    # Weights for a 7-day window: previous 6 days with exp(-i/decay), current day weight = 1.0
    past_lags = np.arange(1, 7, dtype=float)  # 1..6
    w_past = np.exp(-past_lags / decay)      # exp(-i/decay)
    # Order: t-6, ..., t-1, t
    weights = np.concatenate([w_past[::-1], np.array([1.0])])

    def _apply_window(window: np.ndarray) -> float:
        if len(window) != 7 or np.all(np.isnan(window)):
            return np.nan
        return float(np.nansum(window * weights))

    css = (
        pd.Series(s)
        .rolling(window=7, min_periods=7)
        .apply(_apply_window, raw=True)
    )
    return css


def calculate_cumulative_sentiment(
    input_csv: Path | str = DEFAULT_INPUT,
    output_csv: Path | str = DEFAULT_OUTPUT,
    decay: float = 7.0,
) -> pd.DataFrame:
    """
    Load daily sentiment scores and compute the 7-day exponential cumulative score.

    Parameters
    ----------
    input_csv : Path or str
        Path to daily sentiment CSV (date, sentiment_score, article_count).
    output_csv : Path or str
        Path to write the output CSV with an extra css_7d_exp column.
    decay : float
        Decay parameter used in exp(-i/decay).

    Returns
    -------
    pd.DataFrame
        DataFrame with original columns plus css_7d_exp.
    """
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input sentiment CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if "date" not in df.columns or "sentiment_score" not in df.columns:
        raise ValueError("Input CSV must contain columns 'date' and 'sentiment_score'.")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df["css_7d_exp"] = compute_css_7d_exp(df["sentiment_score"], decay=decay)

    df.to_csv(output_csv, index=False)
    print(f"Saved cumulative sentiment to {output_csv}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Observations: {len(df)}")
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute 7-day exponentially-weighted cumulative sentiment score.")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Input daily sentiment CSV (default: daily_sentiment_scores_2017_2024.csv in final_analysis/).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output CSV path (default: daily_cumulative_sentiment_scores_2017_2024.csv in final_analysis/).",
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=7.0,
        help="Decay parameter for exp(-i/decay) (default: 7.0, i.e. exp(-i/7)).",
    )
    args = parser.parse_args()

    calculate_cumulative_sentiment(
        input_csv=Path(args.input),
        output_csv=Path(args.output),
        decay=args.decay,
    )

