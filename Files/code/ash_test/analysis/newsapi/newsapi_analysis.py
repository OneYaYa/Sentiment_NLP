import torch
import numpy as np
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from datetime import datetime, timedelta

MODEL_ID = "yiyanghkust/finbert-tone"
labels = ["Positive", "Negative", "Neutral"]

finbert_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
finbert_model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
finbert_model.eval()


def fetch_article_content(url: str) -> str:
    """
    Fetch full article content from any news URL.
    """
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

        # Remove non-content elements
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()

        # Try common content selectors
        content_selectors = [
            "article",
            ".article-body",
            ".post-content",
            ".entry-content",
            ".content",
            ".story-body",
            ".article-content",
            "main"
        ]
        
        content = ""
        for selector in content_selectors:
            element = soup.select_one(selector)
            if element:
                paragraphs = element.find_all("p")
                content = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
                if len(content) > 200:  # Only use if substantial content
                    break
        
        # Fallback to all paragraphs
        if not content or len(content) < 200:
            paragraphs = soup.find_all("p")
            content = " ".join(p.get_text(" ", strip=True) for p in paragraphs)

        content = " ".join(content.split())  # normalize whitespace
        return content if content else "Content not retrieved."
    except requests.RequestException:
        return "Content not retrieved."


def fetch_newsapi_news(query, num_articles=10, api_key=None):
    """
    Fetch news using NewsAPI.org.
    Requires API key from https://newsapi.org/
    """
    if not api_key:
        # Try to get from environment variable
        api_key = os.getenv("NEWSAPI_KEY")
        if not api_key:
            print("Warning: No NewsAPI key provided. Set NEWSAPI_KEY environment variable or pass api_key parameter.")
            print("You can get a free API key from https://newsapi.org/")
            return []
    
    base_url = "https://newsapi.org/v2/everything"
    
    # Parameters for the API request
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": num_articles,
        "apiKey": api_key,
        "domains": "reuters.com,bloomberg.com,wsj.com,ft.com,cnbc.com"  # Focus on financial news sources
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        articles = []
        
        if data.get("status") == "ok" and "articles" in data:
            for article_data in data["articles"]:
                title = article_data.get("title", "")
                description = article_data.get("description", "")
                url = article_data.get("url", "")
                published_at = article_data.get("publishedAt", "")
                source_name = article_data.get("source", {}).get("name", "")
                
                # Only include articles with substantial content
                if title and url:
                    content = fetch_article_content(url)
                    
                    articles.append({
                        "title": title,
                        "link": url,
                        "published": published_at,
                        "content": content,
                        "description": description,
                        "source": source_name
                    })
        
        return articles
        
    except requests.RequestException as e:
        print(f"Error fetching from NewsAPI: {e}")
        return []
    except Exception as e:
        print(f"Error processing NewsAPI response: {e}")
        return []


def fetch_newsapi_business_news(query, num_articles=10, api_key=None):
    """
    Fetch business news specifically using NewsAPI.
    """
    if not api_key:
        api_key = os.getenv("NEWSAPI_KEY")
        if not api_key:
            return []
    
    base_url = "https://newsapi.org/v2/top-headlines"
    
    params = {
        "category": "business",
        "language": "en",
        "country": "us",  # US business news
        "pageSize": num_articles,
        "apiKey": api_key,
        "q": query if query else "business finance economy"  # Add query if provided
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        articles = []
        
        if data.get("status") == "ok" and "articles" in data:
            for article_data in data["articles"]:
                title = article_data.get("title", "")
                description = article_data.get("description", "")
                url = article_data.get("url", "")
                published_at = article_data.get("publishedAt", "")
                source_name = article_data.get("source", {}).get("name", "")
                
                if title and url:
                    content = fetch_article_content(url)
                    
                    articles.append({
                        "title": title,
                        "link": url,
                        "published": published_at,
                        "content": content,
                        "description": description,
                        "source": source_name
                    })
        
        return articles
        
    except requests.RequestException as e:
        print(f"Error fetching from NewsAPI business: {e}")
        return []
    except Exception as e:
        print(f"Error processing NewsAPI business response: {e}")
        return []


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

        # pad to max_tokens
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


def analyze_sentiment_full_article(title: str, content: str):
    """
    Predict sentiment using title + entire article content.
    Chunk + average probabilities for long articles.
    Returns (confidence, sentiment_label).
    """
    if not title and not content:
        return 0.0, "Neutral"

    text = f"{title}\n\n{content}".strip()

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


def analyze_sentiment_title_only(text: str):
    if not text.strip():
        return 0.0, "Neutral"
    inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = finbert_model(**inputs)
    probs = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return float(probs[idx]), labels[idx]


def summarize_sentiments(articles):
    summary = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for article in articles:
        _, sentiment = analyze_sentiment_full_article(article["title"], article["content"])
        summary[sentiment] += 1

    total = len(articles)
    print("\n--- NewsAPI Sentiment Summary (Full Article) ---")
    print(f"Total articles analyzed: {total}")
    for s, c in summary.items():
        pct = (c / total) * 100 if total else 0
        print(f"{s}: {c} ({pct:.2f}%)")


def main():
    # You need to set your NewsAPI key as environment variable or pass it directly
    api_key = os.getenv("NEWSAPI_KEY")
    
    if not api_key:
        print("Please set NEWSAPI_KEY environment variable with your NewsAPI.org API key")
        print("Get your free key at: https://newsapi.org/")
        return
    
    queries = [
        "gold market",
        "gold price",
        "precious metals",
        "commodities trading",
        "inflation hedge",
        "gold investment",
        "financial markets"
    ]
    num_articles_per_query = 3

    all_articles = []
    for query in queries:
        print(f"Fetching NewsAPI articles for '{query}'...\n")
        
        # Fetch from general search
        search_articles = fetch_newsapi_news(query, num_articles_per_query, api_key)
        all_articles.extend(search_articles)
        
        # Also fetch from business headlines
        business_articles = fetch_newsapi_business_news(query, max(1, num_articles_per_query // 2), api_key)
        all_articles.extend(business_articles)

    for i, article in enumerate(all_articles, 1):
        print(f"Article {i}: {article['title']}")
        print(f"Source: {article.get('source', 'Unknown')}")
        print(f"Link: {article['link']}")
        print(f"Published: {article['published']}")
        print(f"Content chars: {len(article['content'])}")

        conf, sentiment = analyze_sentiment_full_article(article["title"], article["content"])
        print(f"Sentiment: {sentiment} (Confidence: {conf:.3f})\n")

    summarize_sentiments(all_articles)


if __name__ == "__main__":
    main()
