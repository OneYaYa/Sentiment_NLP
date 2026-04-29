import csv
import os
import sys
import torch
import pandas as pd #pip install pandas
import numpy as np
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Model Configuration
MODEL_ID = "/Users/yuepan/Desktop/campbell-B/model/finbert-tone"
labels = ["Positive", "Negative", "Neutral"]

# Load model once
finbert_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
finbert_model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
finbert_model.eval()


# Sentiment Analysis - Title + Full Body
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
        
        # Pad to max_tokens
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

def analyze_sentiment_title_only(text: str):
    """Fallback: analyze sentiment using title only"""
    if not text.strip():
        return "Neutral", 0.0
    
    inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        out = finbert_model(**inputs)
    
    probs = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return labels[idx], float(probs[idx])

def analyze_sentiment_full_article(title: str, content: str):
    """
    Predict sentiment using title + entire article content.
    Chunk + average probabilities for long articles.
    Returns (sentiment_label, confidence).
    """
    if not title and not content:
        return "Neutral", 0.0
    
    # Combine title and full content (no truncation)
    text = f"{title}\n\n{content}".strip()
    
    # If scraping fails, fall back to title-only
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
    return labels[idx], float(avg_probs[idx])


def process_csv_file(input_csv: str, output_csv: str = None):
    """
    Process CSV file with sentiment analysis.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file (optional, auto-generated if not provided)
    
    Returns:
        Path to output CSV file
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV file not found: {input_csv}")
    
    # Read input CSV
    df = pd.read_csv(input_csv)
    
    # Check required columns
    required_cols = ["title", "content"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"Processing {len(df)} articles from {input_csv}...")
    
    # Add sentiment and confidence columns
    sentiments = []
    confidences = []
    
    for idx, row in df.iterrows():
        
        print(f"  Progress: {idx}/{len(df)}")
        
        title = str(row.get("title", ""))
        content = str(row.get("content", ""))
        
        # Combine title and content for analysis
    
        sentiment, confidence = analyze_sentiment_full_article(title, content)
        sentiments.append(sentiment)
        confidences.append(confidence)
    
    # Add new columns
    df["sentiment"] = sentiments
    df["confidence"] = confidences
    
    # Generate output filename if not provided
    if output_csv is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "csv_results"
        os.makedirs(output_dir, exist_ok=True)
        output_csv = os.path.join(output_dir, f"sentiment_results_{timestamp}.csv")
    
    # Save output CSV
    df.to_csv(output_csv, index=False, encoding="utf-8")
    
    # Print summary
    print("\nSentiment Distribution:")
    for sentiment in ["Positive", "Negative", "Neutral"]:
        count = (df["sentiment"] == sentiment).sum()
        pct = 100 * count / len(df)
        print(f"  {sentiment}: {count} ({pct:.1f}%)")
    print(f"\nAverage Confidence: {df['confidence'].mean():.4f}")
    
    return output_csv


def main():
    """Main entry point for command-line usage."""
    if len(sys.argv) < 2:
        print("Usage: python sentiment_analysis_csv.py <input_csv> [output_csv]")
        print("\nExample:")
        print("  python sentiment_analysis_csv.py articles.csv")
        print("  python sentiment_analysis_csv.py articles.csv results.csv")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        output_path = process_csv_file(input_csv, output_csv)
        print(f"\n✓ Successfully completed!")
        print(f"Saved output to: {output_path}")
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
