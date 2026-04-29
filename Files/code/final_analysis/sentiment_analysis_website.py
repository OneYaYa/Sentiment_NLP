import csv
import os
import random
import time
from datetime import datetime
import torch
import numpy as np
import feedparser
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote, urlparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from googlenewsdecoder import gnewsdecoder
from dateutil import parser

# Model Configuration
MODEL_ID = "/Users/yuepan/Desktop/campbell-B/model/finbert-tone" #your path to the finbert-tone model
labels = ["Positive", "Negative", "Neutral"]

finbert_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
finbert_model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
finbert_model.eval()


# Article Content Fetching
def fetch_article_content(url: str, max_retries: int = 2) -> str:
    """Fetch full article text from publisher URL with multi-strategy fallback"""
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    ]
    
    for attempt in range(max_retries):
        try:
            headers = {
                "User-Agent": random.choice(USER_AGENTS),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Referer": f"https://www.google.com/",
                "Cache-Control": "max-age=0",
            }
            
            resp = requests.get(url, timeout=15, allow_redirects=True, headers=headers)
            
            if resp.status_code in [401, 403]:
                return "Content not retrieved."
            
            resp.raise_for_status()
            
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "meta"]):
                tag.decompose()
            
            text = ""
            article_containers = [
                ("article", {}),
                ("div", {"class": ["article", "article-body", "article-content", "post-content", "story-body", "entry-content", "content-body"]}),
                ("main", {}),
            ]
            
            for tag_name, attrs in article_containers:
                container = soup.find(tag_name, attrs) if attrs else soup.find(tag_name)
                if container:
                    paragraphs = container.find_all("p")
                    if paragraphs:
                        text = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
                        if len(text) > 100:
                            break
            
            if not text or len(text) < 50:
                paragraphs = soup.find_all("p")
                text = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
            
            if not text or len(text) < 50:
                divs = soup.find_all("div", class_=lambda x: x and any(kw in x.lower() for kw in ["text", "body", "content", "article"]))
                text = " ".join(div.get_text(" ", strip=True) for div in divs[:5])
            
            text = " ".join(text.split())
            return text if text and len(text) > 50 else "Content not retrieved."
            
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))
                continue
            return "Content not retrieved."
        except:
            return "Content not retrieved."
    
    return "Content not retrieved."


def decode_google_news_url(url: str) -> str:
    """Convert Google News RSS/article links to the original publisher URL"""
    try:
        decoded = gnewsdecoder(url)
        if isinstance(decoded, dict):
            for k in ("decoded_url", "url", "source_url", "original_url"):
                if k in decoded and decoded[k]:
                    return decoded[k]
        if isinstance(decoded, str) and decoded.startswith("http"):
            return decoded
        if isinstance(decoded, (list, tuple)) and decoded and isinstance(decoded[0], str):
            return decoded[0]
    except:
        pass
    return url


def build_query_with_dates(query, start_date=None, end_date=None):
    """Add date filters to Google News queries"""
    parts = [query]
    if start_date:
        parts.append(f"after:{start_date}")
    if end_date:
        parts.append(f"before:{end_date}")
    return " ".join(parts)

# News Sources

def fetch_google_news(query, num_articles=10, start_date=None, end_date=None):
    """Fetch articles from Google News RSS"""
    query_with_dates = build_query_with_dates(query, start_date, end_date)
    rss_url = f"https://news.google.com/rss/search?q={quote(query_with_dates)}"
    feed = feedparser.parse(rss_url)
    articles = []
    
    for item in feed.entries[:num_articles * 2]:
        gnews_link = getattr(item, "link", "")
        publisher_link = decode_google_news_url(gnews_link)
        content = fetch_article_content(publisher_link)
        
        if content and content != "Content not retrieved." and len(content) > 100:
            articles.append({
                "title": getattr(item, "title", ""),
                "link": publisher_link,
                "published": getattr(item, "published", ""),
                "content": content,
                "source": "Google News",
                "query": query  
            })
        
        if len(articles) >= num_articles:
            break
        time.sleep(random.uniform(0.5, 1.5))
    
    return articles


def fetch_bbc_news(query, num_articles=10):
    """Fetch news from BBC News RSS feeds"""
    rss_urls = [
        "http://feeds.bbci.co.uk/news/business/rss.xml",
        "http://feeds.bbci.co.uk/news/world/rss.xml",
        "http://feeds.bbci.co.uk/news/rss.xml",
    ]
    
    articles = []
    for rss_url in rss_urls:
        try:
            feed = feedparser.parse(rss_url)
            for item in feed.entries[:num_articles // len(rss_urls) + 3]:
                link = getattr(item, "link", "")
                if link:
                    content = fetch_article_content(link)
                    if content and content != "Content not retrieved." and len(content) > 100:
                        articles.append({
                            "title": getattr(item, "title", ""),
                            "link": link,
                            "published": getattr(item, "published", ""),
                            "content": content,
                            "source": "BBC News",
                            "query": query
                        })
                if len(articles) >= num_articles:
                    return articles
                time.sleep(random.uniform(0.3, 0.8))
        except:
            pass
    return articles


def fetch_reuters_news(query, num_articles=10):
    """Fetch Reuters articles via Google News (RSS feed deprecated)"""
    # Reuters RSS feeds are deprecated, use Google News to search Reuters articles
    search_query = f"{query} site:reuters.com"
    rss_url = f"https://news.google.com/rss/search?q={quote(search_query)}"
    feed = feedparser.parse(rss_url)
    articles = []
    
    for item in feed.entries[:num_articles * 2]:
        gnews_link = getattr(item, "link", "")
        publisher_link = decode_google_news_url(gnews_link)
        
        # Only include if it's actually from Reuters
        if "reuters.com" in publisher_link.lower():
            content = fetch_article_content(publisher_link)
            if content and content != "Content not retrieved." and len(content) > 100:
                articles.append({
                    "title": getattr(item, "title", ""),
                    "link": publisher_link,
                    "published": getattr(item, "published", ""),
                    "content": content,
                    "source": "Reuters",
                    "query": query
                })
            if len(articles) >= num_articles:
                break
            time.sleep(random.uniform(0.5, 1.0))
    
    return articles


def fetch_yahoo_finance_rss(query, num_articles=10):
    """Fetch news from Yahoo Finance RSS"""
    rss_url = "https://finance.yahoo.com/news/rss"
    
    try:
        feed = feedparser.parse(rss_url)
        articles = []
        
        for item in feed.entries[:num_articles * 2]:
            link = getattr(item, "link", "")
            if link:
                content = fetch_article_content(link)
                if content and content != "Content not retrieved." and len(content) > 100:
                    articles.append({
                        "title": getattr(item, "title", ""),
                        "link": link,
                        "published": getattr(item, "published", ""),
                        "content": content,
                        "source": "Yahoo Finance",
                        "query": query
                    })
                    if len(articles) >= num_articles:
                        break
                time.sleep(random.uniform(0.3, 0.8))
        return articles
    except:
        return []


def fetch_oilprice_news(num_articles=10):
    """Fetch articles from OilPrice.com RSS"""
    feed_url = "https://oilprice.com/rss/main"
    feed = feedparser.parse(feed_url)
    articles = []
    
    for item in feed.entries[:num_articles * 2]:
        link = getattr(item, "link", "")
        if link:
            content = fetch_article_content(link)
            if content and content != "Content not retrieved." and len(content) > 100:
                articles.append({
                    "title": getattr(item, "title", ""),
                    "link": link,
                    "published": getattr(item, "published", ""),
                    "content": content,
                    "source": "OilPrice.com",
                    "query": "crude oil general"
                })
            if len(articles) >= num_articles:
                break
            time.sleep(random.uniform(0.3, 0.8))
    return articles


def fetch_rigzone(num_articles=10):
    """Fetch Rigzone articles via Google News (RSS feed deprecated)"""
    search_query = "oil energy site:rigzone.com"
    rss_url = f"https://news.google.com/rss/search?q={quote(search_query)}"
    feed = feedparser.parse(rss_url)
    articles = []
    
    for item in feed.entries[:num_articles * 2]:
        gnews_link = getattr(item, "link", "")
        publisher_link = decode_google_news_url(gnews_link)
        
        if "rigzone.com" in publisher_link.lower():
            content = fetch_article_content(publisher_link)
            if content and content != "Content not retrieved." and len(content) > 100:
                articles.append({
                    "title": getattr(item, "title", ""),
                    "link": publisher_link,
                    "published": getattr(item, "published", ""),
                    "content": content,
                    "source": "Rigzone",
                    "query": "crude oil general"
                })
            if len(articles) >= num_articles:
                break
            time.sleep(random.uniform(0.5, 1.0))
    return articles


def fetch_world_oil(num_articles=10):
    """Fetch World Oil articles via Google News (RSS feed deprecated)"""
    search_query = "oil energy site:worldoil.com"
    rss_url = f"https://news.google.com/rss/search?q={quote(search_query)}"
    feed = feedparser.parse(rss_url)
    articles = []
    
    for item in feed.entries[:num_articles * 2]:
        gnews_link = getattr(item, "link", "")
        publisher_link = decode_google_news_url(gnews_link)
        
        if "worldoil.com" in publisher_link.lower():
            content = fetch_article_content(publisher_link)
            if content and content != "Content not retrieved." and len(content) > 100:
                articles.append({
                    "title": getattr(item, "title", ""),
                    "link": publisher_link,
                    "published": getattr(item, "published", ""),
                    "content": content,
                    "source": "World Oil",
                    "query": "oil general"
                })
            if len(articles) >= num_articles:
                break
            time.sleep(random.uniform(0.5, 1.0))
    return articles


# Sentiment Analysis - Title + Full Body
def _chunk_tokens(text: str, max_tokens=512, stride=64):
    """
    FinBERT max input length is 512 tokens.
    Chunk long text into overlapping windows.
    """
    enc = finbert_tokenizer(
        text,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=True,
    )
    input_ids = enc["input_ids"][0]
    attention_mask = enc["attention_mask"][0]
    
    chunks = []
    start = 0
    n = int(input_ids.size(0))
    
    while start < n:
        end = min(start + max_tokens, n)
        ids = input_ids[start:end].unsqueeze(0)
        mask = attention_mask[start:end].unsqueeze(0)
        
        # Pad to max_tokens
        pad_len = max_tokens - ids.size(1)
        if pad_len > 0:
            pad_id = finbert_tokenizer.pad_token_id
            ids = torch.cat([ids, torch.full((1, pad_len), pad_id, dtype=ids.dtype)], dim=1)
            mask = torch.cat([mask, torch.zeros((1, pad_len), dtype=mask.dtype)], dim=1)
        
        chunks.append({"input_ids": ids, "attention_mask": mask})
        
        if end == n:
            break
        start = end - stride
    
    return chunks

def analyze_sentiment_title_only(text: str):
    """Fallback: analyze sentiment using title only"""
    if not text.strip():
        return 0.0, "Neutral"
    
    inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        out = finbert_model(**inputs)
    
    probs = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return float(probs[idx]), labels[idx]

def analyze_sentiment_full_article(title: str, content: str):
    """
    Predict sentiment using title + entire article content.
    Chunk + average probabilities for long articles.
    Returns (confidence, sentiment_label).
    """
    if not title and not content:
        return 0.0, "Neutral"
    
    # Combine title and full content (no truncation)
    text = f"{title}\n\n{content}".strip()
    
    # If scraping fails, fall back to title-only
    if content.strip() == "Content not retrieved.":
        return analyze_sentiment_title_only(title)
    
    chunks = _chunk_tokens(text, max_tokens=512, stride=64)
    
    probs_all = []
    with torch.no_grad():
        for ch in chunks:
            out = finbert_model(input_ids=ch["input_ids"], attention_mask=ch["attention_mask"])
            probs = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
            probs_all.append(probs)
    
    avg_probs = np.mean(np.vstack(probs_all), axis=0)
    idx = int(np.argmax(avg_probs))
    return float(avg_probs[idx]), labels[idx]

# Data Processing and Storage
def delete_repeated_articles(all_articles):
    """Remove repeated articles based on title and content"""
    seen = set()
    unique_articles = []
    for article in all_articles:
        key = (article["title"], article["content"])
        if key not in seen:
            seen.add(key)
            unique_articles.append(article)
    return unique_articles


# Summary with Grouping
def summarize_sentiments(all_articles):
    """Summarize sentiment with grouping by date and query"""
    summary = {"Positive": 0, "Negative": 0, "Neutral": 0}
    grouped_summary = {}
    total = len(all_articles)
    
    for article in all_articles:
        if "sentiment" in article and "confidence" in article:
            sentiment = article["sentiment"]
        else:
            confidence, sentiment = analyze_sentiment_full_article(
                article["title"],
                article["content"],
            )
            article["sentiment"] = sentiment
            article["confidence"] = confidence

        summary[sentiment] += 1
        
        # Group by date
        published = article.get('published', '')
        try:
            date_key = parser.parse(published).date().isoformat()
        except Exception:
            date_key = "unknown"
        
        query_key = article.get('query', 'unknown')
        
        if date_key not in grouped_summary:
            grouped_summary[date_key] = {}
        if query_key not in grouped_summary[date_key]:
            grouped_summary[date_key][query_key] = {
                "Positive": 0,
                "Negative": 0,
                "Neutral": 0
            }
        grouped_summary[date_key][query_key][sentiment] += 1
    
    # Overall summary
    print("\n--- Overall (Full Article) ---")
    print(f"Total articles analyzed: {total}")
    for s, c in summary.items():
        pct = (c / total) * 100 if total else 0
        print(f"{s}: {c} ({pct:.2f}%)")
    
    # Grouped by date and query
    print("\n--- Sentiment Score Summary (Full Article) ---")
    for date_key in sorted(grouped_summary.keys()):
        print(f"\nDate: {date_key}")
        for query_key in sorted(grouped_summary[date_key].keys()):
            counts = grouped_summary[date_key][query_key]
            date_query_total = counts['Positive'] + counts['Negative'] + counts['Neutral']
            score = counts['Positive'] / date_query_total if date_query_total > 0 else 0
            line = (
                f"  Query: {query_key} | "
                f"Total Articles: {date_query_total} | "
                f"Sentiment Score: {score:.2f}"
            )
            print(line)
def save_to_csv(all_articles, base_name="sentiment_results", output_dir="website_results"):
    """Save results to CSV """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{timestamp}.csv"

    keys = [
        "title",
        "link",
        "published",
        "content",
        "source",
        "query",
        "sentiment",
        "confidence",
    ]

    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{filename}"
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for article in all_articles:
            writer.writerow({k: article.get(k, "") for k in keys})

    return output_path


def main():
    all_articles = []
    
    # Get date range from user
    start_input = input("Please enter start date (YYYY-MM-DD)/None: ").strip()
    end_input = input("Please enter end date (YYYY-MM-DD)/None: ").strip()
    start_date = start_input if start_input else None
    end_date = end_input if end_input else None
    
    # Fetch from Google News (multiple queries with date filtering)
    queries = [
        "WTI crude oil",
        "Brent crude oil",
        "crude oil prices",
        "crude oil market"
    ]
    
    num_articles_per_query = 200
    
    # Google News: fetch once per query with date filtering
    print("\n=== Fetching from Google News ===")
    for query in queries:
        print(f"  {query}...", end=" ", flush=True)
        articles = fetch_google_news(query, num_articles=num_articles_per_query, start_date=start_date, end_date=end_date)
        all_articles.extend(articles)
        print(f"({len(articles)} valid articles)")
    
    
    print("\n=== Fetching from Reuters News ===")
    for query in queries:
        print(f"  {query}...", end=" ", flush=True)
        articles = fetch_reuters_news(query, num_articles=num_articles_per_query)
        all_articles.extend(articles)
        print(f"({len(articles)} valid articles)")
    
    print("\n=== Fetching from BBC News ===")
    for query in queries:
        print(f"  {query}...", end=" ", flush=True)
        articles = fetch_bbc_news(query, num_articles=num_articles_per_query)
        all_articles.extend(articles)
        print(f"({len(articles)} valid articles)")
    
    print("\n=== Fetching from Yahoo Finance ===")
    for query in queries:
        print(f"  {query}...", end=" ", flush=True)
        articles = fetch_yahoo_finance_rss(query, num_articles=num_articles_per_query)
        all_articles.extend(articles)
        print(f"({len(articles)} valid articles)")
  
    print("\n=== Fetching from other news sources ===")
    print("  OilPrice.com...", end=" ", flush=True)
    articles = fetch_oilprice_news(num_articles=num_articles_per_query)
    all_articles.extend(articles)
    print(f"({len(articles)} valid articles)")
    
    print("  Rigzone...", end=" ", flush=True)
    articles = fetch_rigzone(num_articles=num_articles_per_query)
    all_articles.extend(articles)
    print(f"({len(articles)} valid articles)")
    
    print("  World Oil...", end=" ", flush=True)
    articles = fetch_world_oil(num_articles=num_articles_per_query)
    all_articles.extend(articles)
    print(f"({len(articles)} valid articles)")
    
    unique_articles = delete_repeated_articles(all_articles)
    print(f"\n✓ Finished fetching.")
    
    if not unique_articles:
        print("✗ No articles fetched! Check your network connection or try again later.")
        return
    
    # Analyze sentiment for each article
    print("=== Analyzing Sentiment ===")
    for i, article in enumerate(unique_articles, 1):
        sentiment, conf = analyze_sentiment_full_article(article["title"], article["content"])
        article["sentiment"] = sentiment
        article["confidence"] = conf
        print(f"[{i:3d}/{len(unique_articles)}] {article['title']:25} \n (-------{sentiment:10}) (confidence: {conf:.3f})")
    
    print(f"\nSummarizing Sentiments for all articles (Total: {len(unique_articles)})")
    summarize_sentiments(unique_articles)

    output_path = save_to_csv(unique_articles)
    print(f"\n✓ Finished, Total articles: {len(unique_articles)}")
    print(f"\n✓ Saved results to: {output_path}")

if __name__ == "__main__":
    main()
