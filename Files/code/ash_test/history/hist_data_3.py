import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import json
from newspaper import Article
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from langdetect import detect
import os
import re
from urllib.parse import urlparse

# =============================================================================
# CONFIG
# =============================================================================
data_dir = "/Users/aswathsuresh/Documents/Projects/Campbell-B/ash_data/data_v3"
os.makedirs(data_dir, exist_ok=True)

# OPTIONAL: soft scoring via Tranco top sites list (download once, keep locally)
# If you don't have it, leave as "" and the pipeline will still run with hard filters + diversity caps.
TRANCO_CSV_PATH = "/Users/aswathsuresh/Documents/Projects/Campbell-B/ash_data/top-1m.csv"  # e.g. "/Users/aswathsuresh/Documents/Projects/Campbell-B/tranco.csv"

# Soft scoring threshold: keep only domains in Tranco top N (lower rank = more popular)
TRANCO_MAX_RANK = 200000  # tune: 100000 stricter, 500000 looser

# Diversity control: cap URLs per domain BEFORE scraping
MAX_URLS_PER_DOMAIN_PRE_SCRAPE = 200

# =============================================================================
# Hard filters (tune as needed)
# =============================================================================
LOW_QUALITY_DOMAIN_PATTERNS = [
    # PR / syndication
    r'prnewswire\.com$', r'globenewswire\.com$', r'businesswire\.com$',
    r'einnews\.com$', r'newswire\.ca$', r'openpr\.com$', r'1888pressrelease\.com$',

    # self-publishing / blogging platforms (optional but recommended for "reputable sources" skew)
    r'medium\.com$', r'substack\.com$', r'blogspot\.[a-z.]+$', r'wordpress\.com$',

    # common low-signal finance/politics blog patterns (example; tune to your dataset)
    r'zerohedge\.com$',
]

LOW_QUALITY_URL_PATTERNS = [
    r'/press-release', r'/pressrelease', r'/press_releases',
    r'/sponsored', r'/promoted', r'/advertorial',
]

# =============================================================================
# Date parsing + formatting
# =============================================================================
def parse_gdelt_seendate(seendate: str):
    """
    GDELT seendate format examples:
      - 20210115T010000Z
      - 20210115010000
    Returns a pandas.Timestamp (UTC) or NaT.
    """
    if not seendate:
        return pd.NaT
    s = str(seendate).strip()
    try:
        if "T" in s and s.endswith("Z"):
            return pd.to_datetime(s, format="%Y%m%dT%H%M%SZ", utc=True, errors="coerce")
        if len(s) == 14 and s.isdigit():
            return pd.to_datetime(s, format="%Y%m%d%H%M%S", utc=True, errors="coerce")
        return pd.to_datetime(s, utc=True, errors="coerce")
    except Exception:
        return pd.NaT


def human_readable_utc(ts):
    """Convert a UTC Timestamp to a readable string like '2021-01-15 01:00:00 UTC'."""
    if pd.isna(ts):
        return ""
    try:
        return ts.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return ""


def month_start_end(year: int, month: int):
    """Return (start_datetime, end_datetime_inclusive) for a given month."""
    start_date = datetime(year, month, 1)
    if month == 12:
        end_date = datetime(year, month, 31, 23, 59, 59)
    else:
        end_date = datetime(year, month + 1, 1) - timedelta(seconds=1)
    return start_date, end_date


def iterate_months(start_year: int, start_month: int, end_year: int, end_month: int):
    """
    Yield (year, month) from start (inclusive) to end (inclusive).
    """
    y, m = start_year, start_month
    while (y < end_year) or (y == end_year and m <= end_month):
        yield y, m
        m += 1
        if m == 13:
            m = 1
            y += 1


# =============================================================================
# Utilities: domain extraction
# =============================================================================
def get_domain_from_url(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
        host = host[4:] if host.startswith("www.") else host
        return host
    except Exception:
        return ""


def normalize_domain(d: str) -> str:
    d = (d or "").strip().lower()
    if d.startswith("www."):
        d = d[4:]
    return d


# =============================================================================
# 1) HARD FILTERS
# =============================================================================
def hard_filter_gdelt(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove obvious low-quality sources and press-release/advertorial URLs.
    """
    if df is None or len(df) == 0:
        return df

    out = df.copy()

    # Ensure domain exists (derive if missing)
    if 'domain' not in out.columns or out['domain'].isna().all():
        out['domain'] = out.get('url', '').apply(get_domain_from_url)
    out['domain'] = out['domain'].astype(str).apply(normalize_domain)

    # Domain pattern excludes
    if LOW_QUALITY_DOMAIN_PATTERNS:
        dom_re = re.compile("|".join(LOW_QUALITY_DOMAIN_PATTERNS), re.IGNORECASE)
        out = out[~out['domain'].str.contains(dom_re, na=False)]

    # URL pattern excludes
    if 'url' in out.columns and LOW_QUALITY_URL_PATTERNS:
        url_re = re.compile("|".join(LOW_QUALITY_URL_PATTERNS), re.IGNORECASE)
        out = out[~out['url'].astype(str).str.contains(url_re, na=False)]

    return out


# =============================================================================
# 2) SOFT SCORING (Tranco popularity proxy)
# =============================================================================
def load_tranco_ranks(tranco_csv_path: str) -> dict:
    """
    Optional: load Tranco list you downloaded locally (free).
    Expected format: rank,domain  OR domain only.
    Returns dict: domain -> rank (lower is better).
    """
    ranks = {}
    if not tranco_csv_path or not os.path.exists(tranco_csv_path):
        return ranks

    with open(tranco_csv_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 1:
                domain = parts[0].lower()
                rank = i
            else:
                try:
                    rank = int(parts[0])
                    domain = parts[1].lower()
                except Exception:
                    domain = parts[0].lower()
                    rank = i
            ranks[domain] = rank
    return ranks


def _best_tranco_rank(domain: str, ranks: dict) -> int | None:
    """
    Try exact match; if subdomain, progressively strip leftmost labels:
    e.g. finance.yahoo.com -> yahoo.com -> com (stop)
    """
    d = normalize_domain(domain)
    while d and "." in d:
        if d in ranks:
            return ranks[d]
        d = d.split(".", 1)[1]
    return ranks.get(d)


def apply_soft_scoring_tranco(
    df: pd.DataFrame,
    tranco_ranks: dict,
    max_rank: int
) -> pd.DataFrame:
    """
    Keep domains that appear in Tranco top `max_rank`.
    If tranco_ranks is empty, returns df unchanged.
    """
    if df is None or len(df) == 0:
        return df
    if not tranco_ranks or not max_rank:
        return df

    out = df.copy()
    out['domain'] = out['domain'].astype(str).apply(normalize_domain)

    out['_tranco_rank'] = out['domain'].apply(lambda d: _best_tranco_rank(d, tranco_ranks))
    out = out[out['_tranco_rank'].notna()]
    out = out[out['_tranco_rank'] <= max_rank]
    out = out.drop(columns=['_tranco_rank'], errors='ignore')
    return out


# =============================================================================
# 3) DIVERSITY CONTROLS (cap per domain)
# =============================================================================
def cap_per_domain(df: pd.DataFrame, max_per_domain: int = 200) -> pd.DataFrame:
    """
    Cap number of URLs per domain prior to scraping to prevent domination.
    Keeps earliest by seendate where possible.
    """
    if df is None or len(df) == 0 or max_per_domain is None:
        return df

    out = df.copy()
    out['domain'] = out['domain'].astype(str).apply(normalize_domain)

    if 'seendate' in out.columns:
        out['_seendate_dt'] = out['seendate'].apply(parse_gdelt_seendate)
    else:
        out['_seendate_dt'] = pd.NaT

    out = (
        out.sort_values(by=['domain', '_seendate_dt'], ascending=[True, True], na_position='last')
           .groupby('domain', as_index=False, group_keys=False)
           .head(max_per_domain)
    )
    return out.drop(columns=['_seendate_dt'], errors='ignore')


# =============================================================================
# GDELT fetching (BROAD: no domain list)
# =============================================================================
def fetch_gdelt_urls_broad(keywords, start_date, end_date, sourcelang="eng"):
    """
    Fetch from GDELT without a domain constraint (broad search).
    Returns a DataFrame of GDELT 'articles'.
    """
    all_data = []
    current = start_date

    # GDELT DOC 2.1 lower bound behavior in your original code
    if current < datetime(2017, 2, 19):
        current = datetime(2017, 2, 19)

    query = f'{keywords} sourcelang:{sourcelang}'

    while current < end_date:
        chunk_end = min(current + timedelta(days=7), end_date)  # 7-day chunks

        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params = {
            'query': query,
            'mode': 'artlist',
            'maxrecords': 250,
            'format': 'json',
            'startdatetime': current.strftime('%Y%m%d%H%M%S'),
            'enddatetime': chunk_end.strftime('%Y%m%d%H%M%S')
        }

        try:
            response = requests.get(url, params=params, timeout=30)

            if response.status_code == 200:
                try:
                    if response.text.strip().startswith('{'):
                        data = response.json()
                        articles = data.get('articles', [])
                        all_data.extend(articles)
                except json.JSONDecodeError:
                    pass

            elif response.status_code == 429:
                time.sleep(30)
                continue

        except Exception:
            pass

        current = chunk_end
        time.sleep(10)  # Slower for reliability

    return pd.DataFrame(all_data)


# =============================================================================
# Content extraction
# =============================================================================
def is_english(text):
    if not text or len(text) < 50:
        return False
    try:
        lang = detect(text[:500])
        return lang == 'en'
    except Exception:
        return False


def extract_article_content(url_data):
    url = url_data.get('url', '')
    if not url:
        return {'success': False, 'url': url}

    try:
        article = Article(url)
        article.download()
        article.parse()

        content = article.text

        if content and len(content) > 100 and is_english(content):
            raw_date = url_data.get('seendate', '')  # original GDELT format
            dt = parse_gdelt_seendate(raw_date)

            return {
                'date_raw': raw_date,
                'date_dt': dt,  # sortable datetime (UTC)
                'date_human': human_readable_utc(dt),
                'title': url_data.get('title', ''),
                'content': content,
                'source': url_data.get('domain', get_domain_from_url(url)),
                'url': url,
                'success': True
            }
        else:
            return {'success': False, 'url': url}

    except Exception:
        return {'success': False, 'url': url}


def scrape_parallel(gdelt_data, max_workers=30):
    complete_articles = []
    failed_count = 0

    url_data_list = gdelt_data.to_dict('records')

    print(f"Starting parallel scraping with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(extract_article_content, data): data for data in url_data_list}

        for future in tqdm(as_completed(future_to_url), total=len(url_data_list)):
            try:
                result = future.result()
            except Exception:
                failed_count += 1
                continue

            if result.get('success'):
                complete_articles.append(result)
            else:
                failed_count += 1

    print(f"\nScraping complete: {len(complete_articles)} successful, {failed_count} failed")
    return complete_articles


# =============================================================================
# Processing: month + range
# =============================================================================
def process_month(year, month, keywords, max_workers=30):
    start_date, end_date = month_start_end(year, month)

    # GDELT earliest bound
    if start_date < datetime(2017, 2, 19):
        start_date = datetime(2017, 2, 19)

    month_name = start_date.strftime('%Y_%m')
    output_file = f'{data_dir}/oil_news_{month_name}_filtered.csv'

    print("\n" + "=" * 70)
    print(f"PROCESSING: {start_date.strftime('%B %Y')}")
    print("Fetching broadly from GDELT (no domain list)")
    print("=" * 70)

    print(f"\nSTEP 1: Fetching URLs from GDELT (broad)")
    print("-" * 70)

    gdelt_data = fetch_gdelt_urls_broad(keywords, start_date, end_date)

    print(f"\n✓ Collected {len(gdelt_data)} article URLs (raw)")

    if len(gdelt_data) == 0:
        print("✗ No articles found!")
        return pd.DataFrame()

    # Ensure domain column exists
    if 'domain' not in gdelt_data.columns or gdelt_data['domain'].isna().all():
        gdelt_data['domain'] = gdelt_data.get('url', '').apply(get_domain_from_url)
    gdelt_data['domain'] = gdelt_data['domain'].astype(str).apply(normalize_domain)

    gdelt_data = gdelt_data.drop_duplicates(subset=['url'], keep='first')
    print(f"✓ After URL deduplication: {len(gdelt_data)} unique articles")

    # 1) Hard filters
    pre = len(gdelt_data)
    gdelt_data = hard_filter_gdelt(gdelt_data)
    print(f"✓ Hard filters removed {pre - len(gdelt_data)} rows; remaining {len(gdelt_data)}")

    # 2) Soft scoring (Tranco) - optional
    tranco_ranks = load_tranco_ranks(TRANCO_CSV_PATH)
    if tranco_ranks:
        pre = len(gdelt_data)
        gdelt_data = apply_soft_scoring_tranco(gdelt_data, tranco_ranks, TRANCO_MAX_RANK)
        print(f"✓ Soft scoring (Tranco top {TRANCO_MAX_RANK}) removed {pre - len(gdelt_data)} rows; remaining {len(gdelt_data)}")
    else:
        print("ℹ Soft scoring skipped (no Tranco CSV configured).")

    # 3) Diversity controls (cap per domain)
    pre = len(gdelt_data)
    gdelt_data = cap_per_domain(gdelt_data, MAX_URLS_PER_DOMAIN_PRE_SCRAPE)
    print(f"✓ Diversity cap (max {MAX_URLS_PER_DOMAIN_PRE_SCRAPE}/domain) removed {pre - len(gdelt_data)} rows; remaining {len(gdelt_data)}")

    # Show source breakdown (post filters)
    if len(gdelt_data) > 0:
        print(f"\nArticles per source (post-filters):")
        print(gdelt_data['domain'].value_counts().head(30))

    print(f"\nSTEP 2: Extracting full content")
    print("-" * 70)

    complete_articles = scrape_parallel(gdelt_data, max_workers=max_workers)

    if len(complete_articles) == 0:
        print("✗ No articles extracted!")
        return pd.DataFrame()

    final_df = pd.DataFrame(complete_articles)

    if 'success' in final_df.columns:
        final_df = final_df.drop('success', axis=1)

    # Ensure date_dt is datetime
    if 'date_dt' not in final_df.columns:
        final_df['date_dt'] = pd.NaT
    final_df['date_dt'] = pd.to_datetime(final_df['date_dt'], utc=True, errors='coerce')

    # Ensure the other date columns exist
    if 'date_raw' not in final_df.columns:
        final_df['date_raw'] = ""
    if 'date_human' not in final_df.columns:
        final_df['date_human'] = ""

    # Sort ascending by date, NaTs last
    final_df = final_df.sort_values(by='date_dt', ascending=True, na_position='last')

    # Final column order
    final_df = final_df[['date_raw', 'date_human', 'date_dt', 'title', 'content', 'source', 'url']]

    final_df.to_csv(output_file, index=False)

    print("\n" + "=" * 70)
    print(f"{start_date.strftime('%B %Y').upper()} COMPLETE!")
    print("=" * 70)
    print(f"Total articles: {len(final_df)}")
    print(f"Saved to: {output_file}")

    if len(final_df) > 0:
        print(f"\nContent statistics:")
        print(f"  Mean length: {final_df['content'].str.len().mean():.0f} characters")

        print(f"\nFinal articles per source (post-scrape):")
        print(final_df['source'].value_counts().head(30))

    return final_df


def process_range_to_one_csv(
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
    keywords: str,
    max_workers: int = 30,
    output_filename: str = "oil_news_RANGE_filtered.csv"
):
    """
    Process a month range (inclusive) and write ONE combined CSV at the end,
    sorted ascending by date_dt.
    """
    # Basic validation
    if (end_year, end_month) < (start_year, start_month):
        raise ValueError("End (year, month) must be >= Start (year, month).")

    all_month_dfs = []

    print("\n" + "=" * 70)
    print(f"PROCESSING RANGE: {start_year:04d}-{start_month:02d} to {end_year:04d}-{end_month:02d}")
    print("=" * 70)

    for y, m in iterate_months(start_year, start_month, end_year, end_month):
        df_m = process_month(y, m, keywords, max_workers=max_workers)
        if df_m is not None and len(df_m) > 0:
            all_month_dfs.append(df_m)

    if not all_month_dfs:
        print("\nNo articles found across the entire range.")
        return pd.DataFrame()

    combined = pd.concat(all_month_dfs, ignore_index=True)

    # Deduplicate again across months (same URL can appear in multiple months)
    if 'url' in combined.columns:
        combined = combined.drop_duplicates(subset=['url'], keep='first')

    # Make sure date_dt is parsed and sort
    combined['date_dt'] = pd.to_datetime(combined.get('date_dt', pd.NaT), utc=True, errors='coerce')
    combined['date_human'] = combined['date_dt'].apply(human_readable_utc)
    combined = combined.sort_values(by='date_dt', ascending=True, na_position='last')

    # Final columns
    combined = combined[['date_raw', 'date_human', 'date_dt', 'title', 'content', 'source', 'url']]

    output_path = os.path.join(data_dir, output_filename)
    combined.to_csv(output_path, index=False)

    print("\n" + "=" * 70)
    print("RANGE COMPLETE!")
    print("=" * 70)
    print(f"Total combined articles (deduped): {len(combined)}")
    print(f"Saved ONE combined CSV to: {output_path}")

    return combined


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("CRUDE OIL NEWS: BROAD GDELT + REPUTATION FILTERS + DIVERSITY CAPS")
    print("=" * 70)

    keywords = 'oil'
    max_workers = 30

    # ---- SET YOUR RANGE HERE ----
    START_YEAR = 2023
    START_MONTH = 1
    END_YEAR = 2023
    END_MONTH = 3

    combined_df = process_range_to_one_csv(
        start_year=START_YEAR,
        start_month=START_MONTH,
        end_year=END_YEAR,
        end_month=END_MONTH,
        keywords=keywords,
        max_workers=max_workers,
        output_filename=f"oil_news_{START_YEAR}{START_MONTH:02d}_to_{END_YEAR}{END_MONTH:02d}_filtered.csv"
    )

    print("\nDone.")