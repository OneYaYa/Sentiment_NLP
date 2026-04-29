# ============================================================
# Exploratory Data Analysis (EDA) for oil_news_*.csv (your format)
# - Handles datetime columns like gdelt_seendate (20210106T171500Z)
# - Summarizes missingness, duplicates, key categorical distributions
# - Time trends (daily / monthly), top sources, sentiment-ish fields if present
# - Saves plots to an output folder
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
CSV_PATH = Path("/Users/aswathsuresh/Documents/Projects/Campbell-B/ash_data/data_v2/oil_news_2021_01_to_2025_12_combined.csv")  # change to your combined file if needed
OUT_DIR = CSV_PATH.parent / "eda_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Preferred "main datetime" column for time-series EDA
TIME_COL_CANDIDATES = ["event_datetime", "publish_datetime", "gdelt_datetime", "gdelt_seendate_dt"]

# Potential column names often useful in news datasets
TEXT_COL_CANDIDATES = ["title", "headline", "summary", "content", "body", "text"]
SOURCE_COL_CANDIDATES = ["source", "source_name", "domain", "url_domain", "publisher"]
URL_COL_CANDIDATES = ["url", "source_url", "link"]
LANG_COL_CANDIDATES = ["language", "lang"]
HASH_COL_CANDIDATES = ["content_hash", "hash"]

# If present, treat these as numeric indicators
NUMERIC_HINTS = ["tone", "sentiment", "polarity", "subjectivity", "relevance", "score"]
# ---------------------------


def parse_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Parse datetime columns for this dataset's format."""
    if "gdelt_seendate" in df.columns:
        df["gdelt_seendate_dt"] = pd.to_datetime(
            df["gdelt_seendate"],
            format="%Y%m%dT%H%M%SZ",
            errors="coerce",
            utc=True,
        )

    for col in ["gdelt_datetime", "publish_date", "publish_datetime", "event_datetime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    return df


def pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def basic_overview(df: pd.DataFrame) -> None:
    print("\n==================== BASIC OVERVIEW ====================")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.shape[1]}")
    print("\nDtypes:")
    print(df.dtypes.astype(str).sort_values())

    print("\nSample rows:")
    print(df.head(3))

    print("\nMissing values (top 20):")
    miss = df.isna().mean().sort_values(ascending=False)
    print((miss.head(20) * 100).round(2).astype(str) + "%")


def duplicates_report(df: pd.DataFrame) -> None:
    print("\n==================== DUPLICATES ====================")
    hash_col = pick_first_existing(df, HASH_COL_CANDIDATES)
    if hash_col:
        dup = df.duplicated(subset=[hash_col]).sum()
        print(f"Duplicates by {hash_col}: {dup:,} ({dup/len(df):.2%})")
    else:
        dup = df.duplicated().sum()
        print(f"Full-row duplicates: {dup:,} ({dup/len(df):.2%})")


def describe_numeric(df: pd.DataFrame) -> None:
    print("\n==================== NUMERIC SUMMARY ====================")
    # Coerce hinted columns to numeric if they exist but are object
    for col in df.columns:
        if any(h in col.lower() for h in NUMERIC_HINTS):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        print("No numeric columns detected.")
        return
    print(df[num_cols].describe().T)


def top_categories(df: pd.DataFrame, col: str, n: int = 20) -> pd.DataFrame:
    vc = df[col].astype("string").value_counts(dropna=True).head(n)
    out = vc.reset_index()
    out.columns = [col, "count"]
    out["share"] = out["count"] / out["count"].sum()
    return out


def categorical_reports(df: pd.DataFrame) -> None:
    print("\n==================== CATEGORICAL REPORTS ====================")
    source_col = pick_first_existing(df, SOURCE_COL_CANDIDATES)
    lang_col = pick_first_existing(df, LANG_COL_CANDIDATES)

    if source_col:
        print(f"\nTop sources ({source_col}):")
        print(top_categories(df, source_col, n=15).to_string(index=False))
    else:
        print("\nNo obvious source column found.")

    if lang_col:
        print(f"\nLanguages ({lang_col}):")
        print(top_categories(df, lang_col, n=15).to_string(index=False))


def text_length_reports(df: pd.DataFrame) -> None:
    print("\n==================== TEXT LENGTHS ====================")
    text_col = pick_first_existing(df, TEXT_COL_CANDIDATES)
    if not text_col:
        print("No obvious text/title column found.")
        return

    lengths = df[text_col].astype("string").str.len()
    print(f"Using text column: {text_col}")
    print(lengths.describe())

    # Save histogram
    plt.figure()
    plt.hist(lengths.dropna(), bins=50)
    plt.title(f"Distribution of {text_col} length (chars)")
    plt.xlabel("Characters")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"text_length_hist_{text_col}.png", dpi=200)
    plt.close()
    print(f"Saved: {OUT_DIR / f'text_length_hist_{text_col}.png'}")


def time_series_plots(df: pd.DataFrame) -> None:
    print("\n==================== TIME SERIES ====================")
    time_col = pick_first_existing(df, TIME_COL_CANDIDATES)
    if not time_col:
        print("No time column found for time-series plots.")
        return

    # Ensure timezone-aware datetime
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)

    tmp = df.dropna(subset=[time_col]).copy()
    if tmp.empty:
        print(f"Time column {time_col} exists but has no valid datetimes.")
        return

    print(f"Using time column: {time_col}")
    tmp["date"] = tmp[time_col].dt.date
    tmp["month"] = tmp[time_col].dt.to_period("M").astype(str)

    daily = tmp.groupby("date").size()
    monthly = tmp.groupby("month").size()

    # Daily plot
    plt.figure()
    daily.plot(kind="line")
    plt.title("Articles per day")
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "articles_per_day.png", dpi=200)
    plt.close()

    # Monthly plot
    plt.figure()
    monthly.plot(kind="bar")
    plt.title("Articles per month")
    plt.xlabel("Month")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "articles_per_month.png", dpi=200)
    plt.close()

    # Save tables
    daily.to_csv(OUT_DIR / "articles_per_day.csv", header=["count"])
    monthly.to_csv(OUT_DIR / "articles_per_month.csv", header=["count"])

    print(f"Saved: {OUT_DIR / 'articles_per_day.png'}")
    print(f"Saved: {OUT_DIR / 'articles_per_month.png'}")
    print(f"Saved: {OUT_DIR / 'articles_per_day.csv'}")
    print(f"Saved: {OUT_DIR / 'articles_per_month.csv'}")


def correlations(df: pd.DataFrame) -> None:
    print("\n==================== CORRELATIONS ====================")
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        print("Not enough numeric columns for correlation matrix.")
        return

    corr = num.corr(numeric_only=True)

    # Save correlation matrix as CSV
    corr.to_csv(OUT_DIR / "correlation_matrix.csv")

    # Plot heatmap-like image (matplotlib only)
    plt.figure()
    plt.imshow(corr.values)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation matrix (numeric columns)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "correlation_matrix.png", dpi=200)
    plt.close()

    print(f"Saved: {OUT_DIR / 'correlation_matrix.csv'}")
    print(f"Saved: {OUT_DIR / 'correlation_matrix.png'}")


def main():
    print(f"Reading: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    df = parse_datetimes(df)

    basic_overview(df)
    duplicates_report(df)
    describe_numeric(df)
    categorical_reports(df)
    text_length_reports(df)
    time_series_plots(df)
    correlations(df)

    # Save a cleaned "schema snapshot"
    schema = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str)})
    schema.to_csv(OUT_DIR / "schema.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'schema.csv'}")

    print("\nEDA complete.")


if __name__ == "__main__":
    main()