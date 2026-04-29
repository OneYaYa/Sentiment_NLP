

import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import json
from newspaper import Article
import newspaper
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
# from langdetect import detect
import os

newspaper.settings.browser_user_agent = 'Mozilla/5.0'
newspaper.settings.request_timeout = 10
newspaper.settings.memoize_articles = False

os.makedirs('/content/sample_data/data', exist_ok=True)


def fetch_gdelt_single_domain(keywords, domain, start_date, end_date):
    session = requests.Session()
    all_data = []
    current = start_date
    query = f'{keywords} domain:{domain} sourcelang:eng'

    while current < end_date:
        chunk_end = min(current + timedelta(days=7), end_date)  # 7-day chunks
        params = {
            'query': query,
            'mode': 'artlist',
            'maxrecords': 50,  # Changed to 50
            'format': 'json',
            'startdatetime': current.strftime('%Y%m%d%H%M%S'),
            'enddatetime': chunk_end.strftime('%Y%m%d%H%M%S')
        }
        try:
            response = session.get(
                "https://api.gdeltproject.org/api/v2/doc/doc",
                params=params,
                timeout=15
            )
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                all_data.extend(articles)
            elif response.status_code == 429:
                time.sleep(10)
                continue
        except Exception:
            pass
        current = chunk_end
        time.sleep(1)

    return all_data


def rank_articles(gdelt_data, top_n=100):
    """Rank articles by quality/relevance"""
    if len(gdelt_data) == 0:
        return gdelt_data

    df = gdelt_data.copy()
    df['quality_score'] = 0

    # SOURCE QUALITY
    premium_sources = {
        'reuters.com': 10,
        'oilprice.com': 9,
        'rigzone.com': 8,
        'cnbc.com': 7,
        'marketwatch.com': 7,
    }
    df['quality_score'] += df['domain'].map(lambda x: premium_sources.get(x, 5))

    # KEYWORD RELEVANCE
    if 'title' in df.columns:
        oil_keywords = ['crude', 'wti', 'brent', 'opec', 'oil price',
                       'barrel', 'petroleum', 'production', 'supply']
        df['title_lower'] = df['title'].str.lower().fillna('')
        df['keyword_score'] = df['title_lower'].apply(
            lambda x: sum(3 if kw in x else 0 for kw in oil_keywords)
        )
        df['quality_score'] += df['keyword_score'].clip(0, 6)

    # TITLE LENGTH
    if 'title' in df.columns:
        df['title_len'] = df['title'].str.len()
        df['quality_score'] += (df['title_len'].clip(30, 100) - 30) / 70 * 3

    # AVOID PAYWALLS
    if 'url' in df.columns:
        paywall_indicators = ['premium', 'subscriber', 'membership']
        df['url_lower'] = df['url'].str.lower()
        df['has_paywall'] = df['url_lower'].apply(
            lambda x: any(p in x for p in paywall_indicators)
        )
        df['quality_score'] -= df['has_paywall'] * 3

    # Sort and keep top N
    df = df.sort_values('quality_score', ascending=False).head(top_n)

    # Cleanup
    drop_cols = ['quality_score', 'title_lower', 'keyword_score',
                 'title_len', 'url_lower', 'has_paywall']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


def fetch_gdelt_urls_multi_domain(keywords, domains, start_date, end_date,
                                   top_articles_per_domain=100):
    all_articles = []

    if start_date < datetime(2017, 2, 19):
        start_date = datetime(2017, 2, 19)

    print(f"Searching {len(domains)} domains...")
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n")

    for idx, domain in enumerate(domains, 1):
        print(f"[{idx}/{len(domains)}] {domain}...", end=" ")

        articles = fetch_gdelt_single_domain(keywords, domain, start_date, end_date)

        if len(articles) > 0:
            domain_df = pd.DataFrame(articles)
            domain_df = rank_articles(domain_df, top_n=top_articles_per_domain)
            articles = domain_df.to_dict('records')

        print(f"✓ {len(articles)} articles")
        all_articles.extend(articles)

    print(f"\n✓ Total: {len(all_articles)} ranked articles")
    return pd.DataFrame(all_articles)


def extract_article_content(url_data):
    """Removed langdetect, simple length check"""
    url = url_data['url']
    try:
        article = Article(url)
        article.download()
        article.parse()
        content = article.text

        if content and len(content) > 150:
            return {
                'date': url_data.get('seendate', ''),
                'title': url_data.get('title', ''),
                'content': content,
                'source': url_data.get('domain', ''),
                'url': url,
                'success': True
            }
    except Exception:
        pass

    return {'success': False, 'url': url}


def scrape_parallel(gdelt_data, max_workers=25):  # Increased to 25
    complete_articles = []
    failed_count = 0
    url_data_list = gdelt_data.to_dict('records')

    print(f"Starting parallel scraping with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(extract_article_content, data)
                   for data in url_data_list]

        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result.get('success'):
                complete_articles.append(result)
            else:
                failed_count += 1

    print(f"\nScraping complete: {len(complete_articles)} successful, {failed_count} failed")
    return complete_articles


def process_month(year, month, keywords, domains, max_workers=25,
                  top_articles_per_domain=100):
    start_date = datetime(year, month, 1)

    if month == 12:
        end_date = datetime(year, month, 31, 23, 59, 59)
    else:
        end_date = datetime(year, month + 1, 1) - timedelta(seconds=1)

    if start_date < datetime(2017, 2, 19):
        start_date = datetime(2017, 2, 19)

    month_name = start_date.strftime('%Y_%m')
    output_file = f'/content/sample_data/data/oil_news_{month_name}_filtered.csv'

    print("\n" + "="*70)
    print(f"PROCESSING: {start_date.strftime('%B %Y')}")
    print("="*70)

    print("\nSTEP 1: Fetching and ranking URLs")
    print("-"*70)

    gdelt_data = fetch_gdelt_urls_multi_domain(
        keywords, domains, start_date, end_date,
        top_articles_per_domain=top_articles_per_domain
    )

    print(f"\n✓ Collected {len(gdelt_data)} ranked URLs")

    if len(gdelt_data) == 0:
        return pd.DataFrame()

    gdelt_data = gdelt_data.drop_duplicates(subset=['url'])

    print("\nSTEP 2: Extracting full content")
    print("-"*70)

    complete_articles = scrape_parallel(gdelt_data, max_workers=max_workers)

    if len(complete_articles) == 0:
        return pd.DataFrame()

    final_df = pd.DataFrame(complete_articles)
    final_df = final_df.drop(columns=['success'], errors='ignore')
    final_df = final_df[['date', 'title', 'content', 'source', 'url']]

    final_df.to_csv(output_file, index=False)

    print(f"\nSaved {len(final_df)} articles \u2192 {output_file}")

    return final_df


if __name__ == "__main__":

    keywords = 'oil'

    trusted_domains = [
        'oilprice.com',
        'reuters.com',
        'rigzone.com',
        'cnbc.com',
        'marketwatch.com',
    ]

    max_workers = 25  # Increased to 25
    top_articles = 100

    all_dataframes = []

    for month in range(6, 13):
        df_month = process_month(
            year=2017,
            month=month,
            keywords=keywords,
            domains=trusted_domains,
            max_workers=max_workers,
            top_articles_per_domain=top_articles
        )

        if len(df_month) > 0:
            all_dataframes.append(df_month)

        print(f"\n\u2713 Month {month}/12 complete\n")

    if all_dataframes:
        final_df = pd.concat(all_dataframes, ignore_index=True)
        final_df.to_csv('/content/sample_data/data/oil_news_2017_all.csv', index=False)

        print(f"\n2017 COMPLETE \u2014 Total articles: {len(final_df)}")
        print(f"\nArticles per source:")
        print(final_df['source'].value_counts())
