import torch
import numpy as np
import feedparser
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from googlenewsdecoder import gnewsdecoder
from dateutil import parser

# ============================================================================
# Model Configuration
# ============================================================================
MODEL_ID = "yiyanghkust/finbert-tone"
labels = ["Positive", "Negative", "Neutral"]

finbert_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
finbert_model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
finbert_model.eval()

# ============================================================================
# URL Decoding & Content Fetching
# ============================================================================
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
    except Exception:
        pass
    return url

def fetch_article_content(url: str) -> str:
    """Fetch full article text from publisher URL"""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        }
        resp = requests.get(url, timeout=20, allow_redirects=True, headers=headers)
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Remove junk
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()
        
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
        text = " ".join(text.split())  # normalize whitespace
        
        return text if text else "Content not retrieved."
    except requests.RequestException:
        return "Content not retrieved."

# ============================================================================
# Date Filtering 
# ============================================================================
def build_query_with_dates(query, start_date=None, end_date=None):
    """Add date filters to Google News queries"""
    parts = [query]
    if start_date:
        parts.append(f"after:{start_date}")
    if end_date:
        parts.append(f"before:{end_date}")
    return " ".join(parts)

# ============================================================================
# News Sources 
# ============================================================================
def fetch_google_news(query, num_articles=10, start_date=None, end_date=None):
    """Fetch articles from Google News RSS (FREE)"""
    query_with_dates = build_query_with_dates(query, start_date, end_date)
    rss_url = f"https://news.google.com/rss/search?q={quote(query_with_dates)}"
    feed = feedparser.parse(rss_url)
    articles = []
    
    for item in feed.entries[:num_articles]:
        gnews_link = getattr(item, "link", "")
        publisher_link = decode_google_news_url(gnews_link)
        
        articles.append({
            "title": getattr(item, "title", ""),
            "link": publisher_link,
            "gnews_link": gnews_link,
            "published": getattr(item, "published", ""),
            "content": fetch_article_content(publisher_link),
            "source": "Google News",
            "query": query  # Track which query found this
        })
    
    return articles

def fetch_reuters_news(query, num_articles=10):
    """Fetch articles from Reuters RSS (FREE)"""
    feed_url = f"https://www.reuters.com/rss/search?q={quote(query)}"
    feed = feedparser.parse(feed_url)
    articles = []
    
    for item in feed.entries[:num_articles]:
        link = getattr(item, "link", "")
        articles.append({
            "title": getattr(item, "title", ""),
            "link": link,
            "published": getattr(item, "published", ""),
            "content": fetch_article_content(link),
            "source": "Reuters",
            "query": query
        })
    
    return articles

def fetch_oilprice_news(num_articles=10):
    """Fetch articles from OilPrice.com RSS (FREE)"""
    feed_url = "https://oilprice.com/rss/main"
    feed = feedparser.parse(feed_url)
    articles = []
    
    for item in feed.entries[:num_articles]:
        link = getattr(item, "link", "")
        articles.append({
            "title": getattr(item, "title", ""),
            "link": link,
            "published": getattr(item, "published", ""),
            "content": fetch_article_content(link),
            "source": "OilPrice.com",
            "query": "oil general"
        })
    
    return articles

def fetch_rigzone(num_articles=10):
    """Fetch articles from Rigzone (FREE)"""
    feed_url = "https://www.rigzone.com/news/rss.asp"
    feed = feedparser.parse(feed_url)
    articles = []
    
    for item in feed.entries[:num_articles]:
        link = getattr(item, "link", "")
        articles.append({
            "title": getattr(item, "title", ""),
            "link": link,
            "published": getattr(item, "published", ""),
            "content": fetch_article_content(link),
            "source": "Rigzone",
            "query": "oil general"
        })
    
    return articles

def fetch_world_oil(num_articles=10):
    """Fetch articles from World Oil (FREE)"""
    feed_url = "https://www.worldoil.com/rss"
    feed = feedparser.parse(feed_url)
    articles = []
    
    for item in feed.entries[:num_articles]:
        link = getattr(item, "link", "")
        articles.append({
            "title": getattr(item, "title", ""),
            "link": link,
            "published": getattr(item, "published", ""),
            "content": fetch_article_content(link),
            "source": "World Oil",
            "query": "oil general"
        })
    
    return articles

# ============================================================================
# Sentiment Analysis - Title + Full Body
# ============================================================================
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

# ============================================================================
# Summary with Grouping (NEW!)
# ============================================================================
def summarize_sentiments(articles):
    """Summarize sentiment with grouping by date and query"""
    summary = {"Positive": 0, "Negative": 0, "Neutral": 0}
    grouped_summary = {}
    total = len(articles)
    
    for article in articles:
        _, sentiment = analyze_sentiment_full_article(article["title"], article["content"])
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

# ============================================================================
# Main
# ============================================================================
def main():
    all_articles = []
    
    # Get date range from user
    start_input = input("Please enter start date (YYYY-MM-DD): ").strip()
    end_input = input("Please enter end date (YYYY-MM-DD): ").strip()
    start_date = start_input if start_input else None
    end_date = end_input if end_input else None
    
    # Fetch from Google News (multiple queries with date filtering)
    queries = [
        "WTI crude oil",
        "Brent crude oil",
        "oil prices"
    ]
    
    for query in queries:
        print(f"Fetching Google News for '{query}'...")
        all_articles.extend(fetch_google_news(query, num_articles=10, start_date=start_date, end_date=end_date))
    
    # Fetch from FREE RSS feeds (no API keys needed, no date filtering)
    print("Fetching Reuters...")
    for query in queries:
        print(f"Fetching Reuters for '{query}'...")
        all_articles.extend(fetch_reuters_news(query, num_articles=10))
    
    print("Fetching OilPrice.com...")
    all_articles.extend(fetch_oilprice_news(num_articles=10))
    
    print("Fetching Rigzone...")
    all_articles.extend(fetch_rigzone(num_articles=10))
    
    print("Fetching World Oil...")
    all_articles.extend(fetch_world_oil(num_articles=10))
    
    print(f"\nTotal articles fetched: {len(all_articles)}\n")
    
    # Analyze sentiment for each article
    for i, article in enumerate(all_articles, 1):
        print(f"Article {i}: {article['title']}")
        print(f"Source: {article['source']}")
        print(f"Publisher link: {article['link']}")
        print(f"Published: {article['published']}")
        print(f"Content chars: {len(article['content'])}")
        
        conf, sentiment = analyze_sentiment_full_article(article["title"], article["content"])
        print(f"Sentiment: {sentiment} (Confidence: {conf:.3f})\n")
    
    print(f"Summarizing Sentiments for all articles... Total amount: {len(all_articles)}")
    summarize_sentiments(all_articles)

if __name__ == "__main__":
    main()
