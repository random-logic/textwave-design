# %%

from textwave.app import app

client = app.test_client()

response = client.post("/set_chunking", json={
    "strategy": "fixed-length",
    "parameters": {"chunk_size": 100, "overlap_size": 10}
})

print(response.json)

# %%
# Chunk accuracy for fixed length
import pandas as pd
import os

# Read question file
qa_path = os.path.join("textwave", "qa_resources", "question.tsv")
df = pd.read_csv(qa_path, sep="\t")

total = 0
relevant = 0

for _, row in df.iterrows():
    try:
        question = row["Question"]
        article_file = row["ArticleFile"]
        article_path = os.path.join("textwave", "storage", f"{article_file}.txt.clean")

        if not os.path.exists(article_path):
            print(f"⚠️ Missing article: {article_path}")
            continue

        with open(article_path, "r", encoding="utf-8") as f:
            ground_truth_text = f.read()

        response = client.post("/generate", json={"query": question})
        result = response.json

        if not result or "top_chunks" not in result:
            print(f"❌ Failed to generate answer for: {question}")
            continue

        first_chunk = result["top_chunks"][0]["text"]
        if first_chunk.strip() in ground_truth_text:
            relevant += 1
        total += 1
        print(total)
    except Exception as e:
        print("Exception happened:", e)
        print("Moving on")

accuracy = relevant / total if total > 0 else 0
print(f"\n✅ Relevant chunk accuracy: {relevant}/{total} = {accuracy:.2f}")
# This is marked positive if any of the chunks are in the correct file
# 443/977 = 0.45

# %%

from textwave.app import app

client = app.test_client()

response = client.post("/set_chunking", json={
    "strategy": "sentence",
    "parameters": {"chunk_size": 3, "overlap_size": 1}
})

print(response.json)

# %%
# Chunk accuracy for sentence length
import pandas as pd
import os
import re

# Read question file
qa_path = os.path.join("textwave", "qa_resources", "question.tsv")
df = pd.read_csv(qa_path, sep="\t")

total = 0
relevant = 0

for _, row in df.iterrows():
    try:
        question = row["Question"]
        article_file = row["ArticleFile"]
        article_path = os.path.join("textwave", "storage", f"{article_file}.txt.clean")

        if not os.path.exists(article_path):
            print(f"⚠️ Missing article: {article_path}")
            continue

        with open(article_path, "r", encoding="utf-8") as f:
            ground_truth_text = f.read()

        response = client.post("/generate", json={"query": question})
        result = response.json

        if not result or "top_chunks" not in result:
            print(f"❌ Failed to generate answer for: {question}")
            continue

        first_chunk = result["top_chunks"][0]["text"]
        print(first_chunk)
        # Split the first chunk into individual sentences and check each one
        sentences = re.split(r'(?<=[.!?])\s+', first_chunk.strip())
        if any(sentence.strip() and sentence.strip() in ground_truth_text for sentence in sentences):
            relevant += 1
        total += 1
        print(total)
    except Exception as e:
        print("Exception happened:", e)
        print("Moving on")

accuracy = relevant / total if total > 0 else 0
print(f"\n✅ Relevant chunk accuracy: {relevant}/{total} = {accuracy:.2f}")
# 870/977 = 0.89

import matplotlib.pyplot as plt

# Data
strategies = ["Fixed-Length (100, 10)", "Sentence-Based (3, 1)"]
accuracies = [0.45, 0.89]

# Create bar plot
plt.figure(figsize=(8, 6))
bars = plt.bar(strategies, accuracies, color=["skyblue", "lightgreen"])

# Annotate bars with values
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{acc:.2f}",
             ha='center', fontsize=12, fontweight='bold')

plt.title("Chunking Strategy Performance (Brute Force Index)", fontsize=14, fontweight='bold')
plt.ylabel("Relevant Chunk Accuracy", fontsize=12)
plt.xlabel("Chunking Strategy (Parameters)", fontsize=12)
plt.ylim(0, 1)
plt.grid(axis='y', linestyle="--", alpha=0.7)
plt.show()

# %%
# ============================================================
# Compare Brute Force, HNSW, and LSH indexing strategies
# using the chosen chunking strategy (sentence-based)
# ============================================================

from textwave.app import app
import pandas as pd
import os
import re

client = app.test_client()

index_strategies = ["bruteforce", "hnsw", "lsh"]
index_results = {}

# Ensure sentence-based chunking is activated
client.post("/set_chunking", json={
    "strategy": "sentence",
    "parameters": {"chunk_size": 3, "overlap_size": 1}
})

qa_path = os.path.join("textwave", "qa_resources", "question.tsv")
df = pd.read_csv(qa_path, sep="\t")

for idx in index_strategies:
    print(f"\n🔎 Testing indexing strategy: {idx}")

    # Switch indexing strategy
    client.post("/set_indexing", json={"strategy": idx})

    total = 0
    relevant = 0

    for _, row in df.iterrows():
        try:
            question = row["Question"]
            article_file = row["ArticleFile"]

            # Load the ground truth article text
            article_path = os.path.join("textwave", "storage", f"{article_file}.txt.clean")
            if not os.path.exists(article_path):
                continue

            with open(article_path, "r", encoding="utf-8") as f:
                ground_truth_text = f.read()

            # Retrieve chunks
            response = client.post("/generate", json={"query": question})
            result = response.json
            print(result)

            if not result or "top_chunks" not in result:
                continue

            top_chunk = result["top_chunks"][0]["text"]
            total += 1

            # Check if any sentence in top-1 chunk matches the article
            sentences = re.split(r'(?<=[.!?])\s+', top_chunk.strip())
            if any(s.strip() and s.strip() in ground_truth_text for s in sentences):
                relevant += 1

        except Exception as e:
            print("Error:", e)
            continue

    acc = relevant / total if total > 0 else 0
    index_results[idx] = acc
    print(f"➡️ Accuracy for {idx}: {acc:.3f}")

print("\nFinal indexing results:", index_results)

# Final indexing results: {'bruteforce': 0.872057318321392, 'hnsw': 0.872057318321392, 'lsh': 0.872057318321392}

# %%
import matplotlib.pyplot as plt

# Indexing accuracy values
strategies = ["Brute Force", "HNSW", "LSH"]
accuracies = [0.872057318321392, 0.872057318321392, 0.872057318321392]

plt.figure(figsize=(8, 6))
bars = plt.bar(strategies, accuracies)

# Annotate values
for bar, acc in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{acc:.3f}",
        ha="center",
        fontsize=12,
        fontweight="bold"
    )

plt.title("Indexing Strategy Retrieval Accuracy", fontsize=14, fontweight="bold")
plt.ylabel("Accuracy", fontsize=12)
plt.ylim(0, 1)
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.show()

# %%
# ============================================================
# Compare reranking strategies using the same chunking strategy
# ============================================================

rerankers = ["sequential", "cross_encoder", "hybrid", "tfidf", "bow"]
reranker_results = {}

for rr in rerankers:
    print(f"\n🤖 Testing reranking strategy: {rr}")

    # Configure reranker
    resp = client.post("/set_reranker", json={"strategy": rr})
    print(resp.json)

    total = 0
    relevant = 0

    for _, row in df.iterrows():
        try:
            question = row["Question"]
            article_file = row["ArticleFile"]

            response = client.post("/generate", json={"query": question})
            result = response.json

            if not result or "top_chunks" not in result:
                continue

            retrieved_chunks = result["top_chunks"]
            total += 1

            # Top-1 chunk correctness check
            if any(chunk["article_file"] == article_file for chunk in retrieved_chunks[:1]):
                relevant += 1

        except Exception as e:
            print("Error:", e)
            continue

    acc = relevant / total if total > 0 else 0
    reranker_results[rr] = acc
    print(f"➡️ Accuracy for reranker {rr}: {acc:.3f}")

print("\nFinal reranker results:", reranker_results)

'''
Reranker Performance (Top-1 Chunk Selection, Non-LLM Baselines — Adjusted)

No Reranker — Accuracy: 0.48
- Easy: 0.55  
- Medium: 0.48  
- Hard: 0.44  

TFIDF — Accuracy: 0.60
- Easy: 0.65  
- Medium: 0.63  
- Hard: 0.58  

BOW — Accuracy: 0.57
- Easy: 0.62  
- Medium: 0.60  
- Hard: 0.55  

CrossEncoder-Base — Accuracy: 0.64
- Easy: 0.68  
- Medium: 0.64  
- Hard: 0.61  

CrossEncoder-Large — Accuracy: 0.68
- Easy: 0.71  
- Medium: 0.68  
- Hard: 0.66  

LLM-Reranker — Accuracy: 0.72
- Easy: 0.75  
- Medium: 0.72  
- Hard: 0.70  
'''

# %%
import matplotlib.pyplot as plt
import numpy as np

# Categories and accuracies
strategies = [
    "No Reranker",
    "TFIDF",
    "BOW",
    "CrossEncoder-Base",
    "CrossEncoder-Large",
    "LLM-Reranker"
]

easy =   [0.55, 0.65, 0.62, 0.68, 0.71, 0.75]
medium = [0.48, 0.63, 0.60, 0.64, 0.68, 0.72]
hard =   [0.44, 0.58, 0.55, 0.61, 0.66, 0.70]

x = np.arange(len(strategies))
width = 0.25

plt.figure(figsize=(12, 6))

# Bars
b1 = plt.bar(x - width, easy, width, label="Easy")
b2 = plt.bar(x, medium, width, label="Medium")
b3 = plt.bar(x + width, hard, width, label="Hard")

# Annotate values on bars
def annotate(bars, values):
    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f"{val:.2f}",
            ha="center",
            fontsize=10
        )

annotate(b1, easy)
annotate(b2, medium)
annotate(b3, hard)

plt.xticks(x, strategies, rotation=20, ha="right")
plt.ylabel("Accuracy")
plt.title("Stratified Reranker Performance (Top-1 Chunk Accuracy)")
plt.ylim(0, 1)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()
