import torch
import numpy as np
import feedparser
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# NEW: decode Google News redirect/encoded URLs to the real publisher URL
from googlenewsdecoder import gnewsdecoder  # pip install googlenewsdecoder


MODEL_ID = "yiyanghkust/finbert-tone"
labels = ["Positive", "Negative", "Neutral"]

finbert_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
finbert_model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
finbert_model.eval()


def decode_google_news_url(url: str) -> str:
    """
    Convert Google News RSS/article links to the original publisher URL.
    Falls back to the original URL if decoding fails.
    """
    try:
        # package returns a dict; common keys include "decoded_url" or similar
        decoded = gnewsdecoder(url)
        if isinstance(decoded, dict):
            for k in ("decoded_url", "url", "source_url", "original_url"):
                if k in decoded and decoded[k]:
                    return decoded[k]
        # some versions may return a tuple or string
        if isinstance(decoded, str) and decoded.startswith("http"):
            return decoded
        if isinstance(decoded, (list, tuple)) and decoded and isinstance(decoded[0], str):
            return decoded[0]
    except Exception:
        pass
    return url


def fetch_article_content(url: str) -> str:
    """
    Fetch full article text from publisher URL.
    Adds UA header and removes non-content tags.
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

        # remove obvious junk
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()

        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
        text = " ".join(text.split())  # normalize whitespace

        return text if text else "Content not retrieved."
    except requests.RequestException:
        return "Content not retrieved."


def fetch_news(query, num_articles=10):
    rss_url = f"https://news.google.com/rss/search?q={quote(query)}"
    feed = feedparser.parse(rss_url)

    news_items = feed.entries[:num_articles]
    articles = []

    for item in news_items:
        title = getattr(item, "title", "")
        gnews_link = getattr(item, "link", "")
        published = getattr(item, "published", "")

        # NEW: decode Google News link to publisher link
        publisher_link = decode_google_news_url(gnews_link)

        content = fetch_article_content(publisher_link)

        articles.append(
            {
                "title": title,
                "link": publisher_link,      # store publisher link (better for logging/scraping)
                "gnews_link": gnews_link,    # optional: keep original Google News link
                "published": published,
                "content": content,
            }
        )

    return articles


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

        # pad to max_tokens (optional)
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

    # combine title and content (title helps if content is partial/noisy)
    text = f"{title}\n\n{content}".strip()

    # If scraping fails, fall back to title-only rather than returning garbage
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
    print("\n--- Market Sentiment Summary (Full Article) ---")
    print(f"Total articles analyzed: {total}")
    for s, c in summary.items():
        pct = (c / total) * 100 if total else 0
        print(f"{s}: {c} ({pct:.2f}%)")


def main():
    queries = [
        "gold market",
        "gold price",
        "gold news",
        "gold trends",
        "gold analysis",
        "gold forecast",
        "gold investment",
    ]
    num_articles_per_query = 3

    all_articles = []
    for query in queries:
        print(f"Fetching news articles for '{query}'...\n")
        all_articles.extend(fetch_news(query, num_articles_per_query))
    

    for i, article in enumerate(all_articles, 1):
        print(f"Article {i}: {article['title']}")
        print(f"Publisher link: {article['link']}")
        print(f"Published: {article['published']}")
        print(f"Content chars: {len(article['content'])}")

        conf, sentiment = analyze_sentiment_full_article(article["title"], article["content"])
        print(f"Sentiment: {sentiment} (Confidence: {conf:.3f})\n")

    summarize_sentiments(all_articles)


if __name__ == "__main__":
    main()
