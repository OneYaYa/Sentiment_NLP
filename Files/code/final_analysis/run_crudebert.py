"""
Analyze crude oil news headlines from Desktop XLSX files using CrudeBERT.
Outputs results to Crudebert_results/ matching existing format.
"""
import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = "/Users/yuepan/Desktop/Campbell-B"
MODEL_PATH = os.path.join(ROOT, "model/crudebert/crude_bert_model.bin")
CONFIG_PATH = os.path.join(ROOT, "model/crudebert/crude_bert_config.json")
INPUT_DIR = "/Users/yuepan/Desktop/crude oil news"
OUTPUT_DIR = os.path.join(ROOT, "Files/code/final_analysis/Crudebert_results")

INPUT_FILES = [
    "2024-2026(1).xlsx",
    "2024-2026(2).XLSX",
    "2024-2026(3).XLSX",
    "2024-2026(4).XLSX",
]

labels = ["Positive", "Negative", "Neutral"]

# ── Load CrudeBERT ───────────────────────────────────────────────────────────
print("Loading CrudeBERT model...")
config = AutoConfig.from_pretrained(CONFIG_PATH)
model = AutoModelForSequenceClassification.from_config(config)
state_dict = torch.load(MODEL_PATH, map_location="cpu")
state_dict.pop("bert.embeddings.position_ids", None)
model.load_state_dict(state_dict, strict=False)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print("Model loaded.\n")


def analyze_title(text: str):
    """Predict sentiment from a single headline."""
    if not isinstance(text, str) or not text.strip():
        return "Neutral", 0.0
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        out = model(**inputs)
    probs = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return labels[idx], float(probs[idx])


# ── Read and combine all XLSX files ──────────────────────────────────────────
frames = []
for fname in INPUT_FILES:
    path = os.path.join(INPUT_DIR, fname)
    df = pd.read_excel(path)
    # Keep only the columns we need
    df = df[["Title", "Published date"]].copy()
    df.rename(columns={"Title": "title", "Published date": "date"}, inplace=True)
    frames.append(df)
    print(f"Loaded {fname}: {len(df)} rows")

all_news = pd.concat(frames, ignore_index=True)
all_news["date"] = pd.to_datetime(all_news["date"])
all_news = all_news.sort_values("date").reset_index(drop=True)
all_news = all_news.drop_duplicates(subset=["title", "date"]).reset_index(drop=True)
print(f"\nTotal unique articles: {len(all_news)}")
print(f"Date range: {all_news['date'].min()} to {all_news['date'].max()}\n")

# ── Run sentiment analysis ───────────────────────────────────────────────────
sentiments = []
confidences = []

for idx, row in all_news.iterrows():
    if idx % 200 == 0:
        print(f"  Progress: {idx}/{len(all_news)}")
    sentiment, confidence = analyze_title(row["title"])
    sentiments.append(sentiment)
    confidences.append(confidence)

all_news["sentiment"] = sentiments
all_news["confidence"] = confidences

# ── Print summary ────────────────────────────────────────────────────────────
print("\nSentiment Distribution:")
for s in labels:
    count = (all_news["sentiment"] == s).sum()
    pct = 100 * count / len(all_news)
    print(f"  {s}: {count} ({pct:.1f}%)")
print(f"Average Confidence: {all_news['confidence'].mean():.4f}")

# ── Save per-year CSVs matching existing format ──────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
all_news["year"] = all_news["date"].dt.year

for year, group in all_news.groupby("year"):
    out_path = os.path.join(OUTPUT_DIR, f"Crudebert_{year}_result.csv")
    out_df = group[["date", "title", "sentiment", "confidence"]].copy()
    out_df = out_df.sort_values("date").reset_index(drop=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved {out_path} ({len(out_df)} rows)")

print("\nDone!")
