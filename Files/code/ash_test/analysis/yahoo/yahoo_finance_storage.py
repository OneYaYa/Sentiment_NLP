import csv
from datetime import datetime, timezone
from pathlib import Path
from yahoo_finance_analysis import fetch_yahoo_finance_rss, fetch_yahoo_finance_search, analyze_sentiment_full_article


def collect_yahoo_finance_sentiment(
    queries,
    num_articles_per_query=3
):
    """
    Fetch Yahoo Finance news, run full-article sentiment, return rows for CSV.
    """
    rows = []
    scored_at_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for query in queries:
        # Try RSS first
        rss_articles = fetch_yahoo_finance_rss(query, num_articles=num_articles_per_query)
        
        # If RSS doesn't give enough results, try search
        if len(rss_articles) < num_articles_per_query:
            search_articles = fetch_yahoo_finance_search(query, num_articles_per_query - len(rss_articles))
            all_articles = rss_articles + search_articles
        else:
            all_articles = rss_articles

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
                    "source": "Yahoo Finance"
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
        "gold futures",
        "precious metals",
        "commodities",
        "inflation hedge",
        "gold investment"
    ]

    rows = collect_yahoo_finance_sentiment(
        queries=queries,
        num_articles_per_query=3,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = "/Users/aswathsuresh/Documents/Projects/Campbell-B/data"
    output_file = f"yahoo_finance_sentiment_{timestamp}.csv"

    path = save_to_csv(rows, output_dir + "/" + output_file)
    print(f"Saved {len(rows)} rows to: {path.resolve()}")


if __name__ == "__main__":
    main()
