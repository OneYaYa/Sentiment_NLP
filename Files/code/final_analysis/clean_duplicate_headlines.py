"""
Data cleaning: drop duplicate articles with the same headline within a given day.

Reads oil_news_YYYY_result.csv from csv_results/, deduplicates by (date_day, title),
keeps the first occurrence, and writes to output_dir (or overwrites if output_dir=None).

Usage:
  python clean_duplicate_headlines.py                    # overwrite in place
  python clean_duplicate_headlines.py --output cleaned   # write to csv_results/cleaned/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_RESULTS_DIR = SCRIPT_DIR / "csv_results"


def clean_duplicate_headlines(
    csv_path: Path,
    output_path: Path,
) -> tuple[int, int]:
    """
    Drop duplicate rows with the same headline within a given day.
    Keeps the first occurrence. Returns (original_count, cleaned_count).
    """
    df = pd.read_csv(csv_path)
    original_count = len(df)

    if "date" not in df.columns or "title" not in df.columns:
        raise ValueError(f"Expected columns 'date' and 'title' in {csv_path}")

    df["date"] = pd.to_datetime(df["date"])
    df["date_day"] = df["date"].dt.date

    df_clean = df.drop_duplicates(subset=["date_day", "title"], keep="first")
    df_clean = df_clean.drop(columns=["date_day"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)

    return original_count, len(df_clean)


def main(
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    start_year: int = 2017,
    end_year: int = 2024,
) -> None:
    input_dir = input_dir or CSV_RESULTS_DIR
    output_dir = output_dir or input_dir  # overwrite in place if not specified

    total_removed = 0
    for year in range(start_year, end_year + 1):
        path = input_dir / f"oil_news_{year}_result.csv"
        if not path.exists():
            print(f"  Skip {path.name}: not found")
            continue

        out_path = output_dir / path.name
        try:
            orig, cleaned = clean_duplicate_headlines(path, out_path)
            removed = orig - cleaned
            total_removed += removed
            print(f"  {path.name}: {orig} -> {cleaned} rows ({removed} duplicates removed)")
        except PermissionError:
            print(f"  {path.name}: Permission denied (file may be open elsewhere)")

    print(f"\nTotal duplicates removed: {total_removed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drop duplicate headlines within each day")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output subdir under csv_results (e.g. 'cleaned'). If omitted, overwrites in place.")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Input dir (default: csv_results)")
    parser.add_argument("--start-year", type=int, default=2017)
    parser.add_argument("--end-year", type=int, default=2024)
    args = parser.parse_args()

    inp = Path(args.input) if args.input else CSV_RESULTS_DIR
    out = (CSV_RESULTS_DIR / args.output) if args.output else inp
    main(input_dir=inp, output_dir=out, start_year=args.start_year, end_year=args.end_year)
