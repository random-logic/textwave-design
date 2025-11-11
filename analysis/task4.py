# Task 4: Reranker Performance Comparison (without Mistral)
import pandas as pd
import os
import re
from textwave.app import app

client = app.test_client()

response = client.post("/set_chunking", json={
    "strategy": "sentence",
    "parameters": {"chunk_size": 3, "overlap_size": 1}
})

print(response.json)

RERANKERS = ["sequential", "cross_encoder", "hybrid", "tfidf", "bow"]

qa_path = os.path.join("textwave", "qa_resources", "question.tsv")
df = pd.read_csv(qa_path, sep="\t")

# ✅ Convert all columns to string to avoid float/NaN errors
df = df.astype(str)

# Optional stratified tracking
difficulty_levels = ["easy", "medium", "hard"]
results = {
    r: {
        "total": 0,
        "relevant": 0,
        "strata": {d: {"total": 0, "relevant": 0} for d in difficulty_levels},
    }
    for r in RERANKERS
}

for reranker in RERANKERS:
    print(f"\n🔹 Evaluating reranker: {reranker}")

    # Set reranker
    resp = client.post("/set_reranker", json={"strategy": reranker})
    if resp.status_code != 200:
        print(f"⚠️ Failed to set reranker {reranker}: {resp.json}")
        continue

    count = 0
    for _, row in df.iterrows():
        try:
            question = row.get("Question", "").strip()
            article_file = row.get("ArticleFile", "").strip()
            difficulty = row.get("DifficultyFromAnswerer", "").lower().strip()
            article_path = os.path.join("textwave", "storage", f"{article_file}.txt.clean")

            if not os.path.exists(article_path):
                print(f"⚠️ Missing article: {article_path}")
                continue

            with open(article_path, "r", encoding="utf-8", errors="ignore") as f:
                ground_truth_text = f.read()

            response = client.post("/generate", json={"query": question})
            result = response.json

            if not result or "top_chunks" not in result:
                print(f"❌ Failed to generate answer for: {question}")
                continue

            first_chunk = result["top_chunks"][0]["text"]
            sentences = re.split(r'(?<=[.!?])\s+', first_chunk.strip())
            match_found = any(
                sentence.strip() and sentence.strip() in ground_truth_text
                for sentence in sentences
            )

            # Update counters
            results[reranker]["total"] += 1
            if difficulty in difficulty_levels:
                results[reranker]["strata"][difficulty]["total"] += 1

            if match_found:
                results[reranker]["relevant"] += 1
                if difficulty in difficulty_levels:
                    results[reranker]["strata"][difficulty]["relevant"] += 1

        except Exception as e:
            print("Exception happened:", e)
            continue

        print(count)
        count += 1

# Summarize results
print("\n====================== RESULTS ======================")
for reranker, stats in results.items():
    total, relevant = stats["total"], stats["relevant"]
    acc = relevant / total if total else 0
    print(f"\n{reranker.upper()} — Accuracy: {relevant}/{total} = {acc:.2f}")

    for diff in difficulty_levels:
        d_total = stats["strata"][diff]["total"]
        d_rel = stats["strata"][diff]["relevant"]
        if d_total > 0:
            d_acc = d_rel / d_total
            print(f"  {diff.title()}: {d_rel}/{d_total} = {d_acc:.2f}")

print("\n✅ Done! Compare these results to Task 2 & 3 baselines to discuss performance changes.")

'''
Results:
No reranker accuracy: 0.723
  Easy: 0.695
  Hard: 0.742
  Medium: 0.741

TFIDF — Accuracy: 894/1049 = 0.85
  Easy: 297/322 = 0.92
  Medium: 273/304 = 0.90
  Hard: 158/184 = 0.86

BOW — Accuracy: 878/1049 = 0.84
  Easy: 289/322 = 0.90
  Medium: 267/304 = 0.88
  Hard: 159/184 = 0.86

CrossEncoder-Base — Accuracy: 917/1049 = 0.87
  Easy: 298/322 = 0.93
  Medium: 277/304 = 0.91
  Hard: 162/184 = 0.88

CrossEncoder-Large — Accuracy: 934/1049 = 0.89
  Easy: 300/322 = 0.93
  Medium: 280/304 = 0.92
  Hard: 164/184 = 0.89

LLM-Reranker — Accuracy: 947/1049 = 0.90
  Easy: 301/322 = 0.93
  Medium: 284/304 = 0.93
  Hard: 166/184 = 0.90
'''

# %%
import matplotlib.pyplot as plt
import numpy as np

# === Data (from your results) ===
rerankers = [
    "No Reranker",
    "TF-IDF",
    "BOW",
    "CrossEncoder-Base",
    "CrossEncoder-Large",
    "LLM-Reranker"
]

overall = [0.723, 0.85, 0.84, 0.87, 0.89, 0.90]
easy =    [0.695, 0.92, 0.90, 0.93, 0.93, 0.93]
medium =  [0.741, 0.90, 0.88, 0.91, 0.92, 0.93]
hard =    [0.742, 0.86, 0.86, 0.88, 0.89, 0.90]

x = np.arange(len(rerankers))
width = 0.2

# === Create grouped bar plot ===
plt.figure(figsize=(11, 6))
bars1 = plt.bar(x - width*1.5, easy, width, label="Easy", color="lightgreen")
bars2 = plt.bar(x - width/2, medium, width, label="Medium", color="skyblue")
bars3 = plt.bar(x + width/2, hard, width, label="Hard", color="salmon")
bars4 = plt.bar(x + width*1.5, overall, width, label="Overall", color="gray")

# === Annotate each bar ===
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{bar.get_height():.2f}", ha='center', fontsize=9, fontweight='bold')

# === Formatting ===
plt.title("Reranker Performance Comparison (Architecture Selection)", fontsize=14, fontweight='bold')
plt.ylabel("Accuracy", fontsize=12)
plt.xticks(x, rerankers, rotation=20, ha='right')
plt.ylim(0.65, 0.95)
plt.legend(title="Difficulty")
plt.grid(axis='y', linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()