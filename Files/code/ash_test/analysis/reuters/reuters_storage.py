import csv
from datetime import datetime, timezone
from pathlib import Path
from reuters_analysis import fetch_reuters_news, fetch_reuters_world_news, analyze_sentiment_full_article


def collect_reuters_news_sentiment(
    queries,
    num_articles_per_query=3
):
    """
    Fetch Reuters news, run full-article sentiment, return rows for CSV.
    """
    rows = []
    scored_at_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for query in queries:
        # Fetch from business news
        business_articles = fetch_reuters_news(query, num_articles=num_articles_per_query)
        
        # Also fetch from world news for broader coverage
        world_articles = fetch_reuters_world_news(query, max(1, num_articles_per_query // 2))
        
        all_articles = business_articles + world_articles

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
                    "source": "Reuters"
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
        "source"
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def main():
    queries = [
        "gold market",
        "gold price",
        "commodities",
        "economy",
        "inflation",
        "federal reserve",
        "financial markets"
    ]

    rows = collect_reuters_news_sentiment(
        queries=queries,
        num_articles_per_query=3,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = "/Users/aswathsuresh/Documents/Projects/Campbell-B/data"
    output_file = f"reuters_sentiment_{timestamp}.csv"

    path = save_to_csv(rows, output_dir + "/" + output_file)
    print(f"Saved {len(rows)} rows to: {path.resolve()}")


if __name__ == "__main__":
    main()
