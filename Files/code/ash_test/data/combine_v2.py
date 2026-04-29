# ============================================================
# MERGE ALL MONTHLY CSVs (2021-01 → 2025-12) AND SORT BY datetime
# Expected columns in this CSV format:
#   gdelt_seendate (e.g., 20210106T171500Z)
#   gdelt_datetime, publish_date, publish_datetime, event_datetime (string datetimes)
# ============================================================

import pandas as pd
from pathlib import Path

# ---------- CONFIG ----------
DATA_DIR = Path("/Users/aswathsuresh/Documents/Projects/Campbell-B/ash_data/data_v2")

# Output combined file
OUT_FILE = DATA_DIR / "oil_news_2021_01_to_2025_12_combined.csv"

# File naming template for the NEW format
NAME_TMPL = "oil_news_{year:04d}_{month:02d}.csv"

# Preferred sort order (first one found will be used)
SORT_CANDIDATES = ["event_datetime", "publish_datetime", "gdelt_datetime"]

# If you want to dedupe using a stable ID (recommended if present)
DEDUP_KEY = "content_hash"  # set to None to dedupe by full-row
# ----------------------------


def month_iter(start_year=2021, start_month=1, end_year=2025, end_month=12):
    """Generate (year, month) tuples from start to end inclusive."""
    y, m = start_year, start_month
    while (y < end_year) or (y == end_year and m <= end_month):
        yield y, m
        m += 1
        if m == 13:
            m = 1
            y += 1


def parse_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse datetime columns for this file format.
    - gdelt_seendate: 20210106T171500Z -> UTC datetime
    - other datetime columns: let pandas infer, forcing UTC when possible
    """
    # Parse gdelt_seendate if present
    if "gdelt_seendate" in df.columns:
        df["gdelt_seendate_dt"] = pd.to_datetime(
            df["gdelt_seendate"],
            format="%Y%m%dT%H%M%SZ",
            errors="coerce",
            utc=True,
        )

    # Parse other likely datetime columns if they exist
    for col in ["gdelt_datetime", "publish_date", "publish_datetime", "event_datetime"]:
        if col in df.columns:
            # pandas can usually parse:
            # - "2021-01-06 16:30:00 UTC"
            # - "2021-01-06T10:30:00-06:00"
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    return df


def choose_sort_col(df: pd.DataFrame) -> str | None:
    """Pick the best sort column available."""
    for c in SORT_CANDIDATES:
        if c in df.columns:
            return c
    if "gdelt_seendate_dt" in df.columns:
        return "gdelt_seendate_dt"
    return None


def load_all_csvs(data_dir: Path):
    """Load all available monthly CSV files into a list of DataFrames."""
    dfs = []
    missing_files = []

    for year, month in month_iter():
        file_path = data_dir / NAME_TMPL.format(year=year, month=month)

        if file_path.exists():
            print(f"Loading: {file_path.name}")
            df = pd.read_csv(file_path)
            df = parse_datetimes(df)
            dfs.append(df)
        else:
            missing_files.append(file_path.name)

    print(f"\nTotal files loaded: {len(dfs)}")
    if missing_files:
        print(f"Missing files ({len(missing_files)}): showing first 20")
        print(missing_files[:20])

    if not dfs:
        raise RuntimeError("No CSV files were found. Check DATA_DIR and NAME_TMPL.")

    return dfs


def main():
    dfs = load_all_csvs(DATA_DIR)

    print("\nMerging dataframes...")
    combined_df = pd.concat(dfs, ignore_index=True)

    print("Dropping duplicates...")
    if DEDUP_KEY and DEDUP_KEY in combined_df.columns:
        combined_df = combined_df.drop_duplicates(subset=[DEDUP_KEY])
    else:
        combined_df = combined_df.drop_duplicates()

    sort_col = choose_sort_col(combined_df)
    if sort_col:
        print(f"Sorting by {sort_col}...")
        combined_df = combined_df.sort_values(by=sort_col).reset_index(drop=True)
    else:
        print("WARNING: No suitable datetime column found to sort by.")

    print(f"Saving combined CSV → {OUT_FILE}")
    combined_df.to_csv(OUT_FILE, index=False)

    print("\nDone!")
    print(f"Rows: {len(combined_df):,}")
    print(f"Columns: {combined_df.shape[1]}")


if __name__ == "__main__":
    main()