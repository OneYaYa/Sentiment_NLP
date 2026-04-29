# ============================================================
# MERGE ALL MONTHLY CSVs (2021-01 → 2025-12) AND SORT BY date_raw
# date_raw example format: 20210101T213000Z
# ============================================================

import pandas as pd
from pathlib import Path

# ---------- CONFIG ----------
# Folder containing all oil_news_YYYY_MM_filtered.csv files
DATA_DIR = Path("/Users/aswathsuresh/Documents/Projects/Campbell-B/ash_data/data_v1")

# Output combined file
OUT_FILE = DATA_DIR / "oil_news_2021_01_to_2025_12_combined.csv"

# File naming template
NAME_TMPL = "oil_news_{year:04d}_{month:02d}_filtered.csv"

# Date column to sort by
DATE_COL = "date_raw"
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


def load_all_csvs(data_dir: Path):
    """Load all available monthly CSV files into a list of DataFrames."""
    dfs = []
    missing_files = []

    for year, month in month_iter():
        file_path = data_dir / NAME_TMPL.format(year=year, month=month)

        if file_path.exists():
            print(f"Loading: {file_path.name}")
            df = pd.read_csv(file_path)

            # Convert date_raw like 20210101T213000Z → datetime (UTC)
            if DATE_COL in df.columns:
                df[DATE_COL] = pd.to_datetime(
                    df[DATE_COL],
                    format="%Y%m%dT%H%M%SZ",
                    errors="coerce",
                    utc=True,
                )
            else:
                print(f"WARNING: '{DATE_COL}' not found in {file_path.name}")

            dfs.append(df)
        else:
            missing_files.append(file_path.name)

    print(f"\nTotal files loaded: {len(dfs)}")
    if missing_files:
        print(f"Missing files ({len(missing_files)}):")
        print(missing_files[:20])  # show first 20 only

    if not dfs:
        raise RuntimeError("No CSV files were found. Check DATA_DIR.")

    return dfs


def main():
    # Load all monthly CSVs
    dfs = load_all_csvs(DATA_DIR)

    print("\nMerging dataframes...")
    combined_df = pd.concat(dfs, ignore_index=True)

    print("Dropping duplicate rows...")
    combined_df = combined_df.drop_duplicates()

    # Sort by date_raw if available
    if DATE_COL in combined_df.columns:
        print("Sorting by date_raw...")
        combined_df = combined_df.sort_values(by=DATE_COL).reset_index(drop=True)

    # Save output
    print(f"Saving combined CSV → {OUT_FILE}")
    combined_df.to_csv(OUT_FILE, index=False)

    print("\nDone!")
    print(f"Rows: {len(combined_df):,}")
    print(f"Columns: {combined_df.shape[1]}")


if __name__ == "__main__":
    main()
