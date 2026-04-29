from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
DATA_DIR = Path("/Users/aswathsuresh/Documents/Projects/Campbell-B/ash_data/data_v1")
COMBINED_CSV = DATA_DIR / "oil_news_2021_01_to_2025_12_combined.csv"

# If your combined file doesn't exist yet, set these to build it from monthly files:
BUILD_FROM_MONTHLY = True
NAME_RE = re.compile(r"^oil_news_(\d{4})_(\d{2})_filtered\.csv$")

# EDA options
# ✅ include date_raw so it gets auto-detected for plots
DATE_COL_CANDIDATES = ["date_raw", "date", "published_at", "published", "pub_date", "datetime", "time", "created_at"]
TEXT_COL_CANDIDATES = ["title", "headline", "summary", "description", "content", "text", "body"]
SOURCE_COL_CANDIDATES = ["source", "news_source", "publisher", "site", "domain", "outlet"]
# ----------------------------------------


def load_or_build_combined() -> pd.DataFrame:
    if COMBINED_CSV.exists() and not BUILD_FROM_MONTHLY:
        return pd.read_csv(COMBINED_CSV)

    # Build from monthly CSVs found in folder (more robust than hardcoding month list)
    monthly_files = []
    for p in DATA_DIR.glob("oil_news_*_*_filtered.csv"):
        m = NAME_RE.match(p.name)
        if m:
            monthly_files.append((int(m.group(1)), int(m.group(2)), p))

    if not monthly_files:
        raise FileNotFoundError(f"No monthly files found in {DATA_DIR} matching oil_news_YYYY_MM_filtered.csv")

    monthly_files.sort(key=lambda x: (x[0], x[1]))

    dfs = []
    for y, m, p in monthly_files:
        df = pd.read_csv(p)
        df["year"] = y
        df["month"] = m
        df["source_file"] = p.name
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True).drop_duplicates()

    combined.to_csv(COMBINED_CSV, index=False)
    print(f"Saved combined -> {COMBINED_CSV}")
    return combined


def pick_first_existing(df: pd.DataFrame, candidates) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ✅ updated: parse date_raw with exact format; otherwise fall back
def safe_to_datetime(s: pd.Series, col_name: str | None = None) -> pd.Series:
    if col_name == "date_raw":
        return pd.to_datetime(s, format="%Y%m%dT%H%M%SZ", errors="coerce", utc=True)
    return pd.to_datetime(s, errors="coerce", utc=True)


def summarize_missingness(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().mean().sort_values(ascending=False)
    out = pd.DataFrame({
        "missing_frac": miss,
        "missing_count": (miss * len(df)).round().astype(int),
        "dtype": df.dtypes.astype(str)
    })
    return out


def basic_eda(df: pd.DataFrame):
    print("\n=== SHAPE ===")
    print(df.shape)

    print("\n=== COLUMNS & DTYPES ===")
    print(df.dtypes.sort_index())

    print("\n=== SAMPLE ROWS ===")
    print(df.head(5))

    print("\n=== MISSINGNESS (top 25) ===")
    miss = summarize_missingness(df)
    print(miss.head(25))

    # Numeric summary
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        print("\n=== NUMERIC SUMMARY ===")
        print(df[num_cols].describe().T)

    # Categorical-ish summary (object/string/category/bool)
    # ✅ updated: include "string" to silence Pandas warning and be future-proof
    cat_cols = df.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    if cat_cols:
        print("\n=== TOP VALUES (first 8 categorical columns) ===")
        for c in cat_cols[:8]:
            vc = df[c].value_counts(dropna=False).head(10)
            print(f"\n-- {c} (top 10) --")
            print(vc)

    # Detect likely date / text / source columns
    date_col = pick_first_existing(df, DATE_COL_CANDIDATES)
    text_col = pick_first_existing(df, TEXT_COL_CANDIDATES)
    source_col = pick_first_existing(df, SOURCE_COL_CANDIDATES)

    print("\n=== AUTO-DETECTED COLUMNS ===")
    print({"date_col": date_col, "text_col": text_col, "source_col": source_col})

    # If there is a date column, parse + time-series counts
    if date_col:
        df = df.copy()
        df["_dt"] = safe_to_datetime(df[date_col], date_col)
        valid = df["_dt"].notna().mean()
        print(f"\nParsed '{date_col}' -> valid datetime fraction: {valid:.2%}")

        # Create year-month for grouping
        df["_ym"] = df["_dt"].dt.to_period("M").astype(str)
        ts = df.groupby("_ym").size().sort_index()

        plt.figure()
        ts.plot()
        plt.title("Article count by month")
        plt.xlabel("Year-Month")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

        # Day-of-week distribution
        dow = df["_dt"].dt.day_name()
        dow_counts = dow.value_counts()
        plt.figure()
        dow_counts.plot(kind="bar")
        plt.title("Articles by day of week")
        plt.xlabel("Day")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

    # If there is a source column, top sources
    if source_col:
        top_sources = df[source_col].astype(str).value_counts().head(20)
        plt.figure()
        top_sources.sort_values().plot(kind="barh")
        plt.title("Top 20 sources")
        plt.xlabel("Count")
        plt.ylabel("Source")
        plt.tight_layout()
        plt.show()

    # If there is a text column, compute length stats + plot distribution
    if text_col:
        text = df[text_col].astype(str)
        lengths = text.str.len()
        words = text.str.split().str.len()

        print("\n=== TEXT LENGTH STATS ===")
        print(pd.DataFrame({
            "char_len": lengths.describe(),
            "word_count": words.describe()
        }))

        plt.figure()
        plt.hist(words.dropna(), bins=50)
        plt.title(f"Word-count distribution for '{text_col}'")
        plt.xlabel("Words")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

        # Most common terms (very simple tokenization; no NLP deps)
        tokens = (
            text.str.lower()
                .str.replace(r"[^a-z0-9\s]", " ", regex=True)
                .str.split()
        )

        # Flatten with a small cap for safety on huge datasets
        max_rows = min(len(tokens), 200_000)
        flat = []
        for row in tokens.iloc[:max_rows]:
            flat.extend([t for t in row if len(t) >= 3])

        if flat:
            vc = pd.Series(flat).value_counts().head(30)
            plt.figure()
            vc.sort_values().plot(kind="barh")
            plt.title(f"Top 30 tokens in '{text_col}' (len>=3, first {max_rows:,} rows)")
            plt.xlabel("Count")
            plt.ylabel("Token")
            plt.tight_layout()
            plt.show()

    # If year/month exist (from building), show coverage
    if "year" in df.columns and "month" in df.columns:
        cov = df.groupby(["year", "month"]).size().reset_index(name="count").sort_values(["year", "month"])
        print("\n=== COVERAGE (rows per year-month) ===")
        print(cov.to_string(index=False))


if __name__ == "__main__":
    df = load_or_build_combined()
    basic_eda(df)
