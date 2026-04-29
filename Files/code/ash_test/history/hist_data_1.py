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

data_dir = "/Users/aswathsuresh/Documents/Projects/Campbell-B/data"
os.makedirs(data_dir, exist_ok=True)


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


def fetch_gdelt_single_domain(keywords, domain, start_date, end_date):
    """
    Fetch from GDELT for ONE domain at a time
    """
    all_data = []

    current = start_date
    request_count = 0

    # Simple query for one domain
    query = f'{keywords} domain:{domain} sourcelang:eng'

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
            request_count += 1

            if response.status_code == 200:
                try:
                    if response.text.strip().startswith('{'):
                        data = response.json()
                        articles = data.get('articles', [])
                        all_data.extend(articles)
                    else:
                        pass  # Skip non-JSON responses

                except json.JSONDecodeError:
                    pass

            elif response.status_code == 429:
                time.sleep(30)
                continue

        except Exception:
            pass

        current = chunk_end
        time.sleep(10)  # Slower for reliability

    return all_data


def fetch_gdelt_urls_multi_domain(keywords, domains, start_date, end_date):
    """
    Fetch from GDELT by searching each domain separately
    """
    all_articles = []

    if start_date < datetime(2017, 2, 19):
        start_date = datetime(2017, 2, 19)

    print(f"Searching {len(domains)} domains separately...")
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n")

    for idx, domain in enumerate(domains, 1):
        print(f"[{idx}/{len(domains)}] Searching {domain}...", end=" ")

        articles = fetch_gdelt_single_domain(keywords, domain, start_date, end_date)

        print(f"✓ {len(articles)} articles")
        all_articles.extend(articles)

        # Wait between domains to avoid rate limiting
        if idx < len(domains):
            time.sleep(15)

    print(f"\n✓ Total: {len(all_articles)} articles from all domains")
    return pd.DataFrame(all_articles)


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
                'source': url_data.get('domain', ''),
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


def process_month(year, month, keywords, domains, max_workers=30):
    start_date = datetime(year, month, 1)

    if month == 12:
        end_date = datetime(year, month, 31, 23, 59, 59)
    else:
        end_date = datetime(year, month + 1, 1) - timedelta(seconds=1)

    if start_date < datetime(2017, 2, 19):
        start_date = datetime(2017, 2, 19)

    month_name = start_date.strftime('%Y_%m')
    output_file = f'{data_dir}/oil_news_{month_name}_filtered.csv'

    print("\n" + "=" * 70)
    print(f"PROCESSING: {start_date.strftime('%B %Y')}")
    print(f"Searching {len(domains)} energy news domains")
    print("=" * 70)

    print(f"\nSTEP 1: Fetching URLs from GDELT (one domain at a time)")
    print("-" * 70)

    gdelt_data = fetch_gdelt_urls_multi_domain(keywords, domains, start_date, end_date)

    print(f"\n✓ Collected {len(gdelt_data)} article URLs")

    if len(gdelt_data) == 0:
        print("✗ No articles found!")
        return pd.DataFrame()

    gdelt_data = gdelt_data.drop_duplicates(subset=['url'], keep='first')
    print(f"✓ After deduplication: {len(gdelt_data)} unique articles")

    # Show source breakdown
    if len(gdelt_data) > 0 and 'domain' in gdelt_data.columns:
        print(f"\nArticles per source:")
        print(gdelt_data['domain'].value_counts())

    print(f"\nSTEP 2: Extracting full content")
    print("-" * 70)

    complete_articles = scrape_parallel(gdelt_data, max_workers=max_workers)

    if len(complete_articles) == 0:
        print("✗ No articles extracted!")
        return pd.DataFrame()

    final_df = pd.DataFrame(complete_articles)

    if 'success' in final_df.columns:
        final_df = final_df.drop('success', axis=1)

    # Ensure date_dt is datetime (in case of any object dtype weirdness)
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

        print(f"\nFinal articles per source:")
        print(final_df['source'].value_counts())

    return final_df


if __name__ == "__main__":
    print("=" * 70)
    print("CRUDE OIL NEWS: TRUSTED SOURCES (ONE DOMAIN AT A TIME)")
    print("=" * 70)

    # Simple keyword
    keywords = 'oil'

    # Start with fewer, high-quality domains
    trusted_domains = [
        'oilprice.com',      # Best for crude oil
        'reuters.com',       # Major news
        'rigzone.com',       # Oil & gas
        'cnbc.com',          # Financial news
        'marketwatch.com',   # Markets
    ]

    print(f"\nSearching {len(trusted_domains)} domains:")
    for domain in trusted_domains:
        print(f"  ✓ {domain}")

    print("\nNote: Each domain searched separately to avoid GDELT rate limits")
    print("This will take longer but is more reliable")
    print("=" * 70)

    max_workers = 30

    # Process January 2021
    df_jan = process_month(
        year=2023,
        month=5,
        keywords=keywords,
        domains=trusted_domains,
        max_workers=max_workers
    )

    print("\n\n" + "=" * 70)
    print("TEST COMPLETE!")
    print("=" * 70)
    print(f"Total articles: {len(df_jan)}")
