import torch
import numpy as np
import feedparser
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.parse import quote
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_ID = "yiyanghkust/finbert-tone"
labels = ['Positive', 'Negative', 'Neutral']

finbert_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
finbert_model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)


def fetch_article_content(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text() for p in paragraphs])
        return content.strip()
    except requests.RequestException:
        return "Content not retrieved."


def fetch_news(query, num_articles=10):
    rss_url = f"https://news.google.com/rss/search?q={quote(query)}"
    feed = feedparser.parse(rss_url)

    news_items = feed.entries[:num_articles]

    articles = []
    for item in news_items:
        title = item.title
        link = item.link
        published = item.published
        content = fetch_article_content(link)
        
        articles.append({
            "title": title,
            "link": link,
            "published": published,
            "content": content
        })

    return articles


def analyze_sentiment(text):
    if not text.strip():
        return 0.0, 'Neutral'

    inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = finbert_model(**inputs)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).numpy()[0]
    max_index = np.argmax(probabilities)
    sentiment = labels[max_index]
    confidence = probabilities[max_index]

    return confidence, sentiment


def summarize_sentiments(articles):
    summary = {
        "Positive": 0,
        "Negative": 0,
        "Neutral": 0
    }

    for article in articles:
        _, sentiment = analyze_sentiment(article['title']) # + " " + article['content'])
        summary[sentiment] += 1

    total = len(articles)
    print("\n--- Market Sentiment Summary ---")
    print(f"Total articles analyzed: {total}")
    for sentiment, count in summary.items():
        percent = (count / total) * 100
        print(f"{sentiment}: {count} ({percent:.2f}%)")

def main():
    queries = [
        "gold market",
        "gold price",
        "gold news",
        "gold trends",
        "gold analysis",
        "gold forecast",
        "gold investment"
    ]
    num_articles_per_query = 3
    all_articles = []

    for query in queries:
        print(f"Fetching news articles for '{query}'...\n")
        articles = fetch_news(query, num_articles_per_query)
        all_articles.extend(articles)
    
    print(all_articles)

    for idx, article in enumerate(all_articles, 1):
        print(f"Article {idx}: {article['title']}")
        print(f"Link: {article['link']}")
        print(f"Published: {article['published']}")

        polarity, sentiment = analyze_sentiment(article['title'])  # or article['content']
        print(f"Sentiment: {sentiment} (Polarity: {polarity:.2f})\n")

    summarize_sentiments(all_articles)

if __name__ == "__main__":
    main()
