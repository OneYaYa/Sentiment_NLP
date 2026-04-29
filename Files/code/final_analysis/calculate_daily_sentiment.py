"""
Calculate daily sentiment scores from oil news sentiment analysis results (2017-2024).

Reads oil_news_YYYY_result.csv files from csv_results/, converts article-level
sentiment (Positive/Negative/Neutral) + confidence to numeric scores, aggregates
by date, and outputs a daily sentiment CSV suitable for the trading strategy.

Score mapping:
  - Positive -> +confidence
  - Negative -> -confidence
  - Neutral  -> 0

Daily score = mean of article scores for that date (range approximately [-1, 1]).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


# Path to csv_results directory (relative to this script)
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_RESULTS_DIR = SCRIPT_DIR / "csv_results"
CRUDEBERT_DIR = SCRIPT_DIR / "Crudebert_results"
OUTPUT_DIR = SCRIPT_DIR
OUTPUT_FILE = OUTPUT_DIR / "daily_sentiment_scores_2017_2024.csv"
OUTPUT_FILE_CRUDEBERT = OUTPUT_DIR / "daily_sentiment_scores_crudebert_2017_2024.csv"


def sentiment_to_score(sentiment: str, confidence: float) -> float:
    """
    Map sentiment label + confidence to numeric score in [-1, 1].
    Positive -> +confidence, Negative -> -confidence, Neutral -> 0.
    """
    s = str(sentiment).strip().lower()
    c = float(confidence)
    if s == "positive":
        return c
    if s == "negative":
        return -c
    return 0.0


def load_year_file(year: int, csv_dir: Path, use_crudebert: bool = False) -> pd.DataFrame | None:
    """Load oil_news_YYYY_result.csv or Crudebert_YYYY_result.csv for a given year."""
    if use_crudebert:
        path = csv_dir / f"Crudebert_{year}_result.csv"
    else:
        path = csv_dir / f"oil_news_{year}_result.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    required = ["date", "sentiment", "confidence"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  Skipping {path.name}: missing columns {missing}")
        return None
    return df


def calculate_daily_sentiment(
    csv_dir: Path | str | None = None,
    output_path: Path | str | None = None,
    start_year: int = 2017,
    end_year: int = 2024,
    use_crudebert: bool = False,
) -> pd.DataFrame:
    """
    Load all year CSVs, compute daily sentiment scores, and optionally save.

    Returns DataFrame with columns: date, sentiment_score, article_count
    """
    csv_dir = Path(csv_dir) if csv_dir else (CRUDEBERT_DIR if use_crudebert else CSV_RESULTS_DIR)
    output_path = Path(output_path) if output_path else (OUTPUT_FILE_CRUDEBERT if use_crudebert else OUTPUT_FILE)

    dfs = []
    for year in range(start_year, end_year + 1):
        df = load_year_file(year, csv_dir, use_crudebert=use_crudebert)
        if df is None:
            continue
        df["year_loaded"] = year
        dfs.append(df)

    if not dfs:
        pattern = "Crudebert_YYYY_result.csv" if use_crudebert else "oil_news_YYYY_result.csv"
        raise FileNotFoundError(
            f"No {pattern} files found in {csv_dir} for years {start_year}-{end_year}"
        )

    all_df = pd.concat(dfs, ignore_index=True)

    # Parse dates and extract date (ignore time)
    all_df["date"] = pd.to_datetime(all_df["date"], errors="coerce")
    all_df = all_df.dropna(subset=["date"])
    all_df["date_only"] = all_df["date"].dt.date

    # Compute numeric score per article
    all_df["score"] = all_df.apply(
        lambda r: sentiment_to_score(r["sentiment"], r["confidence"]),
        axis=1,
    )

    # Aggregate by date
    daily = (
        all_df.groupby("date_only", as_index=False)
        .agg(
            sentiment_score=("score", "mean"),
            article_count=("score", "count"),
        )
        .rename(columns={"date_only": "date"})
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    # Save to CSV
    daily.to_csv(output_path, index=False)
    print(f"Saved {len(daily)} daily sentiment scores to {output_path}")
    print(f"  Date range: {daily['date'].min()} to {daily['date'].max()}")
    print(f"  Total articles: {daily['article_count'].sum()}")

    return daily


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute daily sentiment scores from article CSVs")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Input: 'crudebert' for Crudebert_results/, or path/subdir under csv_results (e.g. 'cleaned')")
    parser.add_argument("--start-year", type=int, default=2017)
    parser.add_argument("--end-year", type=int, default=2024)
    args = parser.parse_args()
    csv_dir = None
    use_crudebert = args.input and str(args.input).lower() == "crudebert"
    if args.input and not use_crudebert:
        p = Path(args.input)
        csv_dir = CSV_RESULTS_DIR / p if not p.is_absolute() else p
    df = calculate_daily_sentiment(
        csv_dir=csv_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        use_crudebert=use_crudebert,
    )
    print("\nFirst 10 rows:")
    print(df.head(10).to_string())
