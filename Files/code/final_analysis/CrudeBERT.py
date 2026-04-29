import os
import sys
import json
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
# ── Configuration ─────────────────────────────────────────────────────────────

# Input CSV path — update this to your file location
INPUT_CSV = "/content/oil_news_data.csv"

# Column name for entities dict in your CSV — update if different
ENTITIES_COL = "Dict"

# HuggingFace model (downloads on first run)

MODEL_PATH = hf_hub_download(repo_id="Captain-1337/CrudeBERT", filename="crude_bert_model.bin", local_dir="/content")
CONFIG_PATH = hf_hub_download(repo_id="Captain-1337/CrudeBERT", filename="crude_bert_config.json", local_dir="/content")

print(model_path, config_path)

labels = ["Positive", "Negative", "Neutral"]

# Priority tiers for gold label extraction (lower tier = higher priority)
PRIORITY_ENTITIES = {
    1: ["oil", "crude", "wti", "brent", "crude oil", "brent crude", "brent oil", "crude oil futures"],
    2: ["opec", "production", "supply", "demand", "inventory", "oil marketing companies", "omcs"],
    3: ["energy", "gas", "fuel", "diesel", "petroleum"],
}

# ── Load model ────────────────────────────────────────────────────────────────
config        = AutoConfig.from_pretrained(CONFIG_PATH)
finbert_model = AutoModelForSequenceClassification.from_config(config)
state_dict    = torch.load(MODEL_PATH, map_location="cpu")
state_dict.pop("bert.embeddings.position_ids", None)
finbert_model.load_state_dict(state_dict, strict=False)
finbert_model.eval()
finbert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print("Model loaded.\n")


# ── Gold Label Extraction ─────────────────────────────────────────────────────
def extract_gold_label(entities_json: str) -> str:
    """
    Extract a single gold sentiment label from the entities dict using
    a priority-based approach:
      Tier 1: oil/crude/WTI/Brent price entities
      Tier 2: supply/demand/OPEC entities
      Tier 3: energy/macro entities
      Fallback: majority vote across all entities
    Returns: Positive / Negative / Neutral, or None if unparseable
    """
    try:
        entities = json.loads(str(entities_json))
    except (json.JSONDecodeError, ValueError):
        return None

    entities_lower = {k.lower(): v.lower() for k, v in entities.items()}

    def normalize(label: str) -> str:
        label = label.strip().lower()
        if label in ("positive", "pos"):  return "Positive"
        if label in ("negative", "neg"):  return "Negative"
        if label in ("neutral",  "neu"):  return "Neutral"
        return None

    # Try priority tiers in order
    for tier in [1, 2, 3]:
        for entity in PRIORITY_ENTITIES[tier]:
            if entity in entities_lower:
                normalized = normalize(entities_lower[entity])
                if normalized:
                    return normalized

    # Fallback: majority vote across all entity sentiments
    all_labels = [normalize(v) for v in entities_lower.values()]
    all_labels = [l for l in all_labels if l is not None]
    if not all_labels:
        return None
    return Counter(all_labels).most_common(1)[0][0]


# ── Sentiment Analysis - Headline Only ────────────────────────────────────────
def analyze_sentiment_title_only(text: str):
    """Analyze sentiment using headline only."""
    if not text.strip():
        return "Neutral", 0.0

    inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=64)

    with torch.no_grad():
        out = finbert_model(**inputs)

    probs = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return labels[idx], float(probs[idx])


# ── Process CSV ───────────────────────────────────────────────────────────────
def process_csv_file(input_csv: str, output_csv: str = None):
    """
    Process CSV file with headline-only sentiment analysis.

    Args:
        input_csv: Path to input CSV file (must have a 'Title' column)
        output_csv: Path to output CSV file (optional, auto-generated if not provided)

    Returns:
        Path to output CSV file
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV file not found: {input_csv}")

    df = pd.read_csv(input_csv)

    if "Title" not in df.columns:
        raise ValueError("Missing required column: 'Title'")

    print(f"Processing {len(df)} headlines from {input_csv}...")

    sentiments  = []
    confidences = []
    gold_labels = []

    for idx, row in df.iterrows():

        print(f"  Progress: {idx + 1}/{len(df)}")

        Title = str(row.get("Title", ""))
        sentiment, confidence = analyze_sentiment_title_only(Title)
        sentiments.append(sentiment)
        confidences.append(confidence)

        # Extract gold label from entities dict if column exists
        if ENTITIES_COL in df.columns:
            gold = extract_gold_label(row.get(ENTITIES_COL, ""))
        else:
            gold = None
        gold_labels.append(gold)

    df["sentiment"]  = sentiments
    df["confidence"] = confidences
    df["gold_label"] = gold_labels

    if output_csv is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "csv_results"
        os.makedirs(output_dir, exist_ok=True)
        output_csv = os.path.join(output_dir, f"sentiment_results_{timestamp}.csv")

    df.to_csv(output_csv, index=False, encoding="utf-8")

    # ── Sentiment Distribution ────────────────────────────────────────────────
    print("\nSentiment Distribution:")
    for sentiment in ["Positive", "Negative", "Neutral"]:
        count = (df["sentiment"] == sentiment).sum()
        pct = 100 * count / len(df)
        print(f"  {sentiment}: {count} ({pct:.1f}%)")
    print(f"\nAverage Confidence: {df['confidence'].mean():.4f}")

    # ── Metrics vs Gold Labels ────────────────────────────────────────────────
    eval_df = df[df["gold_label"].notna()].copy()
    if len(eval_df) > 0:
        print(f"\n── Evaluation ({len(eval_df)} rows with valid gold labels) ──────────")
        y_true = eval_df["gold_label"].tolist()
        y_pred = eval_df["sentiment"].tolist()

        acc       = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {precision:.4f} (weighted)")
        print("\n── Per-Class Report ──────────────────────────────────────────────")
        print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    else:
        print(f"\n⚠ No gold labels found — check that '{ENTITIES_COL}' column exists in your CSV.")

    return output_csv


def main():
    # Use command-line arg if provided, otherwise fall back to INPUT_CSV variable above
    input_csv  = INPUT_CSV
    output_csv = "/content/CrudeBERTsentiment_results.csv"

    try:
        output_path = process_csv_file(input_csv, output_csv)
        print(f"\nSuccessfully completed!")
        print(f"Saved output to: {output_path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
