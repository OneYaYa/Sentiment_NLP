import csv
from datetime import datetime, timezone
from pathlib import Path
from title_analysis import fetch_news, analyze_sentiment


def fetch_and_score_to_rows(queries, num_articles_per_query=3):
    """
    Fetch news for each query and return rows ready for CSV writing.
    Each row includes: query, title, link, published, sentiment, confidence, scored_at_utc
    """
    rows = []
    scored_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for query in queries:
        articles = fetch_news(query, num_articles=num_articles_per_query)

        for a in articles:
            confidence, sentiment = analyze_sentiment(a.get("title", "") or "")

            rows.append(
                {
                    "query": query,
                    "title": a.get("title", ""),
                    "link": a.get("link", ""),
                    "published": a.get("published", ""),
                    "sentiment": sentiment,
                    "confidence": float(confidence),
                    "scored_at_utc": scored_at_utc,
                }
            )

    return rows


def write_rows_to_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "query",
        "title",
        "link",
        "published",
        "sentiment",
        "confidence",
        "scored_at_utc",
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
        "gold news",
        "gold trends",
        "gold analysis",
        "gold forecast",
        "gold investment",
    ]
    num_articles_per_query = 3

    rows = fetch_and_score_to_rows(queries, num_articles_per_query=num_articles_per_query)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    output_dir = "/Users/aswathsuresh/Documents/Projects/Campbell-B/data"
    out_file = f"title_sentiment_{timestamp}.csv"

    path = write_rows_to_csv(rows, output_dir + "/" + out_file)
    print(f"Saved {len(rows)} rows to: {path.resolve()}")


if __name__ == "__main__":
    main()
