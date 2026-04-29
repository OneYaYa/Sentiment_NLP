import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from newsapi_analysis import fetch_newsapi_news, fetch_newsapi_business_news, analyze_sentiment_full_article


def collect_newsapi_sentiment(
    queries,
    num_articles_per_query=3,
    api_key=None
):
    """
    Fetch NewsAPI news, run full-article sentiment, return rows for CSV.
    """
    if not api_key:
        api_key = os.getenv("NEWSAPI_KEY")
        if not api_key:
            print("Warning: No NewsAPI key provided. Set NEWSAPI_KEY environment variable.")
            return []
    
    rows = []
    scored_at_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for query in queries:
        # Fetch from general search
        search_articles = fetch_newsapi_news(query, num_articles=num_articles_per_query, api_key=api_key)
        
        # Also fetch from business headlines
        business_articles = fetch_newsapi_business_news(query, max(1, num_articles_per_query // 2), api_key=api_key)
        
        all_articles = search_articles + business_articles

        for article in all_articles:
            confidence, sentiment = analyze_sentiment_full_article(
                article["title"],
                article["content"],
            )

            rows.append(
                {
                    "query": query,
                    "title": article["title"],
                    "content": article["content"],
                    "publisher_link": article["link"],
                    "published": article["published"],
                    "sentiment": sentiment,
                    "confidence": round(float(confidence), 6),
                    "content_length": len(article["content"]),
                    "scored_at_utc": scored_at_utc,
                    "source": article.get("source", "NewsAPI"),
                    "description": article.get("description", "")
                }
            )

    return rows


def save_to_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "query",
        "title",
        "content",
        "publisher_link",
        "published",
        "sentiment",
        "confidence",
        "content_length",
        "scored_at_utc",
        "source",
        "description"
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def main():
    # Check for API key
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

    rows = collect_newsapi_sentiment(
        queries=queries,
        num_articles_per_query=3,
        api_key=api_key
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = "/Users/aswathsuresh/Documents/Projects/Campbell-B/data"
    output_file = f"newsapi_sentiment_{timestamp}.csv"

    path = save_to_csv(rows, output_dir + "/" + output_file)
    print(f"Saved {len(rows)} rows to: {path.resolve()}")


if __name__ == "__main__":
    main()
