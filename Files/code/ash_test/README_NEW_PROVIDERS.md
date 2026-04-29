# Multi-Provider News Sentiment Analysis

This directory contains sentiment analysis modules for multiple news providers, extending beyond the original Google News implementation.

## Available Providers

### 1. Google News (Original)
- **Analysis**: `body_analysis.py`, `title_analysis.py`
- **Storage**: `body_storage.py`, `title_storage.py`
- **Features**: Full article content extraction, FinBERT sentiment analysis

### 2. BBC News
- **Analysis**: `bbc_analysis.py`
- **Storage**: `bbc_storage.py`
- **Features**: BBC Business RSS feed, content-specific extraction

### 3. Reuters
- **Analysis**: `reuters_analysis.py`
- **Storage**: `reuters_storage.py`
- **Features**: Business & World News RSS feeds, financial focus

### 4. Yahoo Finance
- **Analysis**: `yahoo_finance_analysis.py`
- **Storage**: `yahoo_finance_storage.py`
- **Features**: RSS + search API, financial news specialization

### 5. NewsAPI.org
- **Analysis**: `newsapi_analysis.py`
- **Storage**: `newsapi_storage.py`
- **Features**: Multiple sources, requires API key

## Usage

### Individual Providers

#### BBC News
```bash
python bbc_analysis.py  # Run analysis
python bbc_storage.py  # Save to CSV
```

#### Reuters
```bash
python reuters_analysis.py
python reuters_storage.py
```

#### Yahoo Finance
```bash
python yahoo_finance_analysis.py
python yahoo_finance_storage.py
```

#### NewsAPI (requires API key)
```bash
export NEWSAPI_KEY="your_api_key_here"
python newsapi_analysis.py
python newsapi_storage.py
```

### All Providers (Recommended)

Run the unified script to analyze across all providers:

```bash
python run_all_providers.py
```

This will:
- Fetch news from all available providers
- Run sentiment analysis using FinBERT
- Save individual CSV files per provider
- Create a combined summary report

## Setup

### Dependencies
```bash
pip install torch transformers feedparser requests beautifulsoup4 googlenewsdecoder vaderSentiment
```

### NewsAPI Setup (Optional)
1. Get free API key from https://newsapi.org/
2. Set environment variable:
   ```bash
   export NEWSAPI_KEY="your_api_key_here"
   ```

## Output Files

All CSV files are saved to `../data/` directory with timestamps:
- `google_sentiment_YYYYMMDD_HHMMSS.csv`
- `bbc_sentiment_YYYYMMDD_HHMMSS.csv`
- `reuters_sentiment_YYYYMMDD_HHMMSS.csv`
- `yahoo_finance_sentiment_YYYYMMDD_HHMMSS.csv`
- `newsapi_sentiment_YYYYMMDD_HHMMSS.csv`
- `all_providers_summary_YYYYMMDD_HHMMSS.csv`

## CSV Structure

Each CSV contains:
- `query`: Search query used
- `title`: Article title
- `content`: Full article content
- `publisher_link`: URL to article
- `published`: Publication date
- `sentiment`: Positive/Negative/Neutral
- `confidence`: Sentiment confidence score
- `content_length`: Character count of content
- `scored_at_utc`: Analysis timestamp
- `source`: News provider name

## Customization

### Modify Queries
Edit the `queries` list in any analysis script:

```python
queries = [
    "gold market",
    "gold price",
    "your_custom_query",
    # ... more queries
]
```

### Adjust Article Count
Change `num_articles_per_query` parameter:

```python
num_articles_per_query = 5  # Default is 3
```

### Add New Providers
1. Create `{provider}_analysis.py` following existing pattern
2. Create `{provider}_storage.py` for CSV export
3. Add to `run_all_providers.py`

## Model Information

All providers use the **FinBERT** model (`yiyanghkust/finbert-tone`) for financial sentiment analysis:
- **Positive**: Bullish/optimistic sentiment
- **Negative**: Bearish/pessimistic sentiment  
- **Neutral**: Objective/balanced sentiment

The model is specifically trained for financial text and provides domain-appropriate sentiment classification.

## Error Handling

- Content extraction failures fall back to title-only analysis
- Network timeouts are handled gracefully
- Missing API keys don't crash other providers
- Invalid URLs are skipped with warnings

## Performance Notes

- Full article analysis is more accurate but slower
- Title-only analysis is faster but less comprehensive
- Some providers may rate-limit requests
- Content extraction quality varies by site
