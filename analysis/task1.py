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