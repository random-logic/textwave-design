"""
task3.py

Compare generative performance of Mistral models (small/medium/large) without RAG.
Uses sentence chunking function for completeness, but the generative baseline is
"stand-alone" (no retrieval). Scores are computed via semantic similarity using
Mistral embeddings (mistral-embed).

Requirements:
- pip install mistralai pandas scikit-learn nltk
- export MISTRAL_API_KEY="your_key_here"

Run: python task3.py

Outputs: prints overall and difficulty-stratified mean similarity per model and
saves CSV to textwave/results/generative_mistral_sentence_baseline.csv
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from mistralai import Mistral
import nltk

nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

# ---------- Configuration ----------
QA_PATH = os.path.join("textwave", "qa_resources", "question.tsv")
STORAGE_DIR = os.path.join("textwave", "storage")
OUTPUT_DIR = os.path.join("textwave", "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = [
    # "mistral-small-latest",
    "mistral-medium-latest",
]

# semantic embed model name used by Mistral SDK
EMBED_MODEL = "mistral-embed"

# Generation settings
MAX_TOKENS = 256
SYSTEM_PROMPT = "You are a helpful factual question-answering assistant. Answer concisely."

# Optional sentence chunking params (provided for completeness)
CHUNK_SENTENCES = 3
CHUNK_OVERLAP = 1

# ---------- Helpers ----------

def make_sentence_chunks(text, chunk_size=CHUNK_SENTENCES, overlap=CHUNK_OVERLAP):
    """Split text into sentence-based chunks."""
    sents = sent_tokenize(text)
    if chunk_size <= 0:
        return [" ".join(sents)]
    chunks = []
    i = 0
    step = max(1, chunk_size - overlap)
    while i < len(sents):
        chunk = " ".join(sents[i:i+chunk_size])
        chunks.append(chunk)
        i += step
    return chunks


def semantic_similarity(client, a, b):
    """Compute cosine similarity between embeddings for strings a and b."""
    try:
        r1 = client.embeddings.create(model=EMBED_MODEL, inputs=[a])
        r2 = client.embeddings.create(model=EMBED_MODEL, inputs=[b])
        emb1 = r1.data[0].embedding
        emb2 = r2.data[0].embedding
        sim = cosine_similarity([emb1], [emb2])[0][0]
        return float(sim)
    except Exception as e:
        print(f"⚠️ Embedding error: {e}")
        return 0.0


def generate_answer(client, model, question):
    """Generate an answer using Mistral chat.complete (stand-alone, no RAG)."""
    max_retries = 3
    retries = 0
    while retries < max_retries:
        try:
            resp = client.chat.complete(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                max_tokens=MAX_TOKENS,
            )
            # message is an AssistantMessage object; use .content
            generated = resp.choices[0].message.content.strip()
            return generated
        except Exception as e:
            err_msg = str(e)
            # Check for 500-series error in exception message
            if "500" in err_msg:
                print(
                    f"⚠️ Server-side error (500) for model={model}, retrying in 60 seconds... (attempt {retries + 1}/{max_retries})")
                retries += 1
                time.sleep(60)
                continue
            else:
                print(f"❌ Generation error for model={model}: {e}")
                return ""
    print(f"❌ Exceeded max retries ({max_retries}) for model={model} on question: {question}")
    return ""


# ---------- Main evaluation ----------

def main():
    api_key = "LMbH1QizQMAyn5zaF6M1VUp61oPwL9Kb"
    if not api_key:
        print("Error: MISTRAL_API_KEY environment variable not set. Exiting.")
        sys.exit(1)

    client = Mistral(api_key=api_key)

    if not os.path.exists(QA_PATH):
        print(f"Error: QA file not found at {QA_PATH}")
        sys.exit(1)

    df = pd.read_csv(QA_PATH, sep="\t")
    rows = []

    for model in MODELS:
        print("\n==============================")
        print(f"Evaluating model: {model}")
        model_records = []

        count = 0
        for idx, row in df.iterrows():
            q = str(row.get("Question", "")).strip()
            gt = str(row.get("Answer", "")).strip()
            difficulty = str(row.get("DifficultyFromAnswerer", "unknown")).lower()

            # load article file (if exists) and create sentence chunks (not used for generation here)
            article_file = row.get("ArticleFile")
            article_text = ""
            if pd.notna(article_file):
                art_path = os.path.join(STORAGE_DIR, f"{article_file}.txt.clean")
                if os.path.exists(art_path):
                    try:
                        with open(art_path, "r", encoding="utf-8") as f:
                            article_text = f.read()
                    except Exception:
                        article_text = ""
            # generate chunks (for logging / possible analysis)
            chunks = make_sentence_chunks(article_text) if article_text else []

            generated = generate_answer(client, model, q)
            sim = semantic_similarity(client, generated, gt) if generated and gt else 0.0

            model_records.append({
                "model": model,
                "question": q,
                "ground_truth": gt,
                "generated": generated,
                "similarity": sim,
                "difficulty": difficulty,
                "num_chunks": len(chunks),
            })

            # small sleep to avoid rate limits
            time.sleep(0.05)

            print(count)
            count += 1

        mdf = pd.DataFrame(model_records)
        overall = mdf["similarity"].mean()
        strat = mdf.groupby("difficulty")["similarity"].mean().to_dict()

        print(f"Overall mean similarity: {overall:.4f}")
        for d in ["easy", "medium", "hard"]:
            print(f"  {d}: {strat.get(d, float('nan')):.4f}")

        # append for CSV
        rows.append({
            "model": model,
            "overall_mean_similarity": overall,
            "easy_mean_similarity": strat.get("easy", np.nan),
            "medium_mean_similarity": strat.get("medium", np.nan),
            "hard_mean_similarity": strat.get("hard", np.nan),
        })

        # save per-model detailed results
        per_model_path = os.path.join(OUTPUT_DIR, f"generative_{model.replace('/', '_')}_detailed.csv")
        mdf.to_csv(per_model_path, index=False)
        print(f"Saved detailed results to {per_model_path}")

    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(OUTPUT_DIR, "generative_mistral_sentence_with_rag.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()

'''
Overall mean similarity: 0.7059
  easy: 0.7077
  medium: 0.7498
  hard: 0.7393
Saved detailed results to textwave/results/generative_mistral-small-latest_detailed.csv

Overall mean similarity: 0.7164
  easy: 0.7078
  medium: 0.7666
  hard: 0.7594
Saved detailed results to textwave/results/generative_mistral-medium-latest_detailed.csv
'''

# %%
import matplotlib.pyplot as plt
import numpy as np

# Data
categories = ["Easy", "Medium", "Hard", "Overall"]
mistral_small_rag = [0.7077, 0.7498, 0.7393, 0.7059]
mistral_medium_rag = [0.7078, 0.7666, 0.7594, 0.7164]

x = np.arange(len(categories))
width = 0.35

# Create grouped bar plot
plt.figure(figsize=(9, 6))
bars1 = plt.bar(x - width/2, mistral_small_rag, width, label="Mistral-Small-Latest (RAG)", color="skyblue")
bars2 = plt.bar(x + width/2, mistral_medium_rag, width, label="Mistral-Medium-Latest (RAG)", color="lightgreen")

# Annotate bars with values
for bars in [bars1, bars2]:
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004, f"{bar.get_height():.4f}",
                 ha='center', fontsize=10, fontweight='bold')

# Formatting
plt.title("Retrieval-Augmented Generative Model Performance Comparison (No Reranker)",
          fontsize=14, fontweight='bold')
plt.ylabel("Mean Similarity", fontsize=12)
plt.xticks(x, categories)
plt.ylim(0.68, 0.78)
plt.legend()
plt.grid(axis='y', linestyle="--", alpha=0.6)
plt.show()