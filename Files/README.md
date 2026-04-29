# Campbell-B: Financial Sentiment Analysis

A comprehensive Python-based financial sentiment analysis tool that leverages FinBERT to analyze market sentiment from news articles with advanced features including date-range filtering and full article content analysis.

## Overview

This project fetches news articles related to financial markets and analyzes their sentiment using the FinBERT model from Hugging Face. The tool provides both title-based and full article content analysis with temporal filtering capabilities.

## Key Features

- **Enhanced News Aggregation**: Automatically fetches financial news with Google News URL decoding
- **Full Article Analysis**: Analyzes complete article content, not just titles
- **Date Range Filtering**: Filter news articles by specific date ranges
- **Advanced Content Processing**: Handles long articles with chunking and overlapping windows
- **Publisher URL Resolution**: Decodes Google News redirects to original publisher URLs
- **Comprehensive Sentiment Analysis**: Provides both overall and grouped sentiment summaries
- **Interactive Date Input**: User-friendly date range selection

## Recent Updates

### New Features in `body_analysis_withdate_260209.py`:
- **Full Article Content Analysis**: Processes entire article bodies instead of just titles
- **Date Range Queries**: Filter news by specific start and end dates
- **Google News URL Decoding**: Resolves Google News redirects to original publisher URLs
- **Advanced Text Chunking**: Handles long articles with 512-token chunks and 64-token stride
- **Grouped Sentiment Analysis**: Provides sentiment scores grouped by date and query
- **Enhanced Web Scraping**: Improved content extraction with better headers and junk removal
- **Interactive Input**: Prompts users for date ranges during execution

### Market Focus:
- **Original**: Gold market analysis
- **Updated**: Crude oil market analysis (configurable queries)

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Campbell-B
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install additional dependency for URL decoding:
```bash
pip install googlenewsdecoder python-dateutil
```

5. Download the FinBERT model:
The model will be automatically downloaded from Hugging Face when first run, or you can manually download it from:
https://huggingface.co/yiyanghkust/finbert-tone

## Usage

### Basic Analysis (Original)
```bash
python code/sentiment_analysis.py
```

### Advanced Analysis with Date Filtering (Recommended)
```bash
python code/body_analysis_withdate_260209.py
```

The advanced script will:
- Prompt for start and end dates (YYYY-MM-DD format)
- Fetch news articles for crude oil-related queries within the date range
- Analyze full article content using FinBERT
- Display individual article analysis with confidence scores
- Provide grouped sentiment analysis by date and query
- Show overall market sentiment summary

## Project Structure

```
Campbell-B/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── code/
│   ├── sentiment_analysis.py         # Original title-based analysis
│   ├── body_analysis_withdate_260209.py # Enhanced full article analysis
│   └── ash_test/                     # Test files
├── data/                             # Data storage (empty by default)
├── docs/                             # Documentation
│   ├── CampbellB_PPT.pdf
│   └── CampbellB_TradingStategy.pdf
└── venv/                             # Virtual environment
```

## Model Details

- **Model**: FinBERT-tone by yiyanghkust
- **Purpose**: Financial sentiment analysis
- **Classes**: Positive, Negative, Neutral
- **Max Input Length**: 512 tokens
- **Source**: https://huggingface.co/yiyanghkust/finbert-tone

## Dependencies

Core libraries include:
- `transformers` - For FinBERT model loading
- `torch` - PyTorch for model inference
- `feedparser` - RSS feed parsing
- `beautifulsoup4` - Web scraping
- `requests` - HTTP requests
- `numpy` - Numerical operations
- `googlenewsdecoder` - Google News URL decoding
- `python-dateutil` - Date parsing utilities

## Configuration

### Advanced Script Configuration:
- **Search Queries**: Modify the `queries` list in `main()` function
- **Article Count**: Adjust `num_articles_per_query` variable
- **Date Range**: Interactive input during runtime
- **Model Path**: Update `MODEL_ID` constant if using local model
- **Chunking Parameters**: Modify `max_tokens` and `stride` in `_chunk_tokens()`

### Original Script Configuration:
- **Search Queries**: Modify the `queries` list
- **Article Count**: Adjust `num_articles_per_query` variable

## Output

### Advanced Analysis Output:
- Individual article analysis with full content sentiment
- Confidence scores for each prediction
- Overall sentiment summary with percentages
- Grouped sentiment scores by date and query
- Sentiment score calculation (Positive/Total ratio)

### Original Analysis Output:
- Title-based sentiment analysis
- Basic sentiment summary

## Technical Improvements

### Content Processing:
- **Text Chunking**: Long articles are split into 512-token chunks with 64-token overlap
- **URL Decoding**: Google News URLs are resolved to original publisher URLs
- **Enhanced Scraping**: Better content extraction with proper headers and junk removal
- **Fallback Mechanism**: Falls back to title-only analysis if content extraction fails

### Analysis Features:
- **Temporal Analysis**: Date-based sentiment tracking
- **Query-based Grouping**: Separate sentiment analysis per search query
- **Confidence Scoring**: Detailed confidence metrics for predictions
- **Sentiment Score Calculation**: Ratio-based scoring system (0.0 to 1.0)

## Notes

- Internet connection required for news fetching and model download
- RSS feeds may have rate limits
- Full article analysis increases processing time but provides better accuracy
- Date filtering enables historical sentiment analysis
- Model path may need adjustment based on your system configuration
