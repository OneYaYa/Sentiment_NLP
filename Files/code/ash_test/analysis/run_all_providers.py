#!/usr/bin/env python3
"""
Unified script to run sentiment analysis across all news providers.
This script fetches and analyzes news from multiple sources and saves results.
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Import all provider modules
from body_storage import collect_news_sentiment as collect_google_sentiment
from bbc_storage import collect_bbc_news_sentiment
from reuters_storage import collect_reuters_news_sentiment
from yahoo_finance_storage import collect_yahoo_finance_sentiment
from newsapi_storage import collect_newsapi_sentiment


def run_all_providers(queries, num_articles_per_query=3, output_dir=None):
    """
    Run sentiment analysis across all available news providers.
    
    Args:
        queries: List of search queries
        num_articles_per_query: Number of articles per query per provider
        output_dir: Directory to save CSV files (default: ../data)
    
    Returns:
        Dictionary with provider names and their result file paths
    """
    if output_dir is None:
        output_dir = "/Users/aswathsuresh/Documents/Projects/Campbell-B/data"
    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results = {}
    
    print("=" * 60)
    print("Starting Multi-Provider News Sentiment Analysis")
    print("=" * 60)
    print(f"Queries: {queries}")
    print(f"Articles per query: {num_articles_per_query}")
    print(f"Output directory: {output_dir}")
    print(f"Timestamp: {timestamp}")
    print()
    
    # 1. Google News
    print("1. Running Google News analysis...")
    try:
        google_rows = collect_google_sentiment(queries, num_articles_per_query)
        if google_rows:
            from body_storage import save_to_csv as save_google_csv
            google_file = os.path.join(output_dir, f"google_sentiment_{timestamp}.csv")
            save_google_csv(google_rows, google_file)
            results["Google News"] = google_file
            print(f"   ✓ Saved {len(google_rows)} articles to {google_file}")
        else:
            print("   ✗ No articles retrieved from Google News")
    except Exception as e:
        print(f"   ✗ Error with Google News: {e}")
    
    # 2. BBC News
    print("\n2. Running BBC News analysis...")
    try:
        bbc_rows = collect_bbc_news_sentiment(queries, num_articles_per_query)
        if bbc_rows:
            from bbc_storage import save_to_csv as save_bbc_csv
            bbc_file = os.path.join(output_dir, f"bbc_sentiment_{timestamp}.csv")
            save_bbc_csv(bbc_rows, bbc_file)
            results["BBC News"] = bbc_file
            print(f"   ✓ Saved {len(bbc_rows)} articles to {bbc_file}")
        else:
            print("   ✗ No articles retrieved from BBC News")
    except Exception as e:
        print(f"   ✗ Error with BBC News: {e}")
    
    # 3. Reuters
    print("\n3. Running Reuters analysis...")
    try:
        reuters_rows = collect_reuters_news_sentiment(queries, num_articles_per_query)
        if reuters_rows:
            from reuters_storage import save_to_csv as save_reuters_csv
            reuters_file = os.path.join(output_dir, f"reuters_sentiment_{timestamp}.csv")
            save_reuters_csv(reuters_rows, reuters_file)
            results["Reuters"] = reuters_file
            print(f"   ✓ Saved {len(reuters_rows)} articles to {reuters_file}")
        else:
            print("   ✗ No articles retrieved from Reuters")
    except Exception as e:
        print(f"   ✗ Error with Reuters: {e}")
    
    # 4. Yahoo Finance
    print("\n4. Running Yahoo Finance analysis...")
    try:
        yahoo_rows = collect_yahoo_finance_sentiment(queries, num_articles_per_query)
        if yahoo_rows:
            from yahoo_finance_storage import save_to_csv as save_yahoo_csv
            yahoo_file = os.path.join(output_dir, f"yahoo_finance_sentiment_{timestamp}.csv")
            save_yahoo_csv(yahoo_rows, yahoo_file)
            results["Yahoo Finance"] = yahoo_file
            print(f"   ✓ Saved {len(yahoo_rows)} articles to {yahoo_file}")
        else:
            print("   ✗ No articles retrieved from Yahoo Finance")
    except Exception as e:
        print(f"   ✗ Error with Yahoo Finance: {e}")
    
    # 5. NewsAPI (if API key is available)
    print("\n5. Running NewsAPI analysis...")
    newsapi_key = os.getenv("NEWSAPI_KEY")
    if newsapi_key:
        try:
            newsapi_rows = collect_newsapi_sentiment(queries, num_articles_per_query, newsapi_key)
            if newsapi_rows:
                from newsapi_storage import save_to_csv as save_newsapi_csv
                newsapi_file = os.path.join(output_dir, f"newsapi_sentiment_{timestamp}.csv")
                save_newsapi_csv(newsapi_rows, newsapi_file)
                results["NewsAPI"] = newsapi_file
                print(f"   ✓ Saved {len(newsapi_rows)} articles to {newsapi_file}")
            else:
                print("   ✗ No articles retrieved from NewsAPI")
        except Exception as e:
            print(f"   ✗ Error with NewsAPI: {e}")
    else:
        print("   ⚠ Skipping NewsAPI - no API key found (set NEWSAPI_KEY environment variable)")
    
    # Summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    total_articles = sum(len([line for line in open(path, 'r').readlines() if line.strip()]) - 1 
                        for path in results.values() if os.path.exists(path))
    
    print(f"Providers processed: {len(results)}")
    print(f"Total articles analyzed: {total_articles}")
    print("\nGenerated files:")
    for provider, filepath in results.items():
        if os.path.exists(filepath):
            line_count = sum(1 for line in open(filepath, 'r') if line.strip()) - 1  # Subtract header
            print(f"  {provider}: {filepath} ({line_count} articles)")
    
    return results


def create_summary_report(results, output_dir):
    """
    Create a summary report combining all provider results.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(output_dir, f"all_providers_summary_{timestamp}.csv")
    
    all_rows = []
    
    for provider, filepath in results.items():
        if os.path.exists(filepath):
            try:
                import csv
                with open(filepath, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        row['provider'] = provider
                        all_rows.append(row)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
    
    if all_rows:
        # Save combined summary
        fieldnames = list(all_rows[0].keys())
        
        with open(summary_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        
        print(f"\nCombined summary saved to: {summary_file}")
        print(f"Total articles in summary: {len(all_rows)}")
        
        # Create sentiment summary by provider
        sentiment_summary = {}
        for row in all_rows:
            provider = row['provider']
            sentiment = row['sentiment']
            
            if provider not in sentiment_summary:
                sentiment_summary[provider] = {"Positive": 0, "Negative": 0, "Neutral": 0}
            
            sentiment_summary[provider][sentiment] += 1
        
        print("\nSentiment Summary by Provider:")
        print("-" * 40)
        for provider, sentiments in sentiment_summary.items():
            total = sum(sentiments.values())
            print(f"\n{provider}:")
            for sentiment, count in sentiments.items():
                pct = (count / total) * 100 if total > 0 else 0
                print(f"  {sentiment}: {count} ({pct:.1f}%)")
    
    return summary_file if all_rows else None


def main():
    """Main function to run all providers."""
    
    # Default queries focused on financial markets
    queries = [
        "gold market",
        "gold price",
        "commodities",
        "inflation",
        "federal reserve",
        "stock market",
        "financial news"
    ]
    
    # You can customize these
    num_articles_per_query = 3
    output_dir = "/Users/aswathsuresh/Documents/Projects/Campbell-B/data"
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run all providers
    results = run_all_providers(queries, num_articles_per_query, output_dir)
    
    # Create combined summary
    if results:
        create_summary_report(results, output_dir)
    
    print(f"\nAnalysis completed at: {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    main()
