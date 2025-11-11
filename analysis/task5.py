from textwave.app import app

client = app.test_client()

response = client.post("/set_chunking", json={
    "strategy": "sentence",
    "parameters": {"chunk_size": 3, "overlap_size": 1}
})

print(response.json)

import pandas as pd
import os
import re

# Read question file
qa_path = os.path.join("textwave", "qa_resources", "question.tsv")
df = pd.read_csv(qa_path, sep="\t")

metrics = {m: {"relevant": 0, "total": 0} for m in range(1, 11)}

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

        # Loop over m = 1..10 and measure retrieval performance for each
        for m in range(1, 11):
            if "top_chunks" not in result:
                continue

            top_chunks = result["top_chunks"][:m]
            print(top_chunks)
            match_found = False  # reset for each m

            for chunk in top_chunks:
                sentences = re.split(r'(?<=[.!?])\s+', chunk["text"].strip())
                if any(sentence.strip() and sentence.strip() in ground_truth_text for sentence in sentences):
                    match_found = True
                    break

            metrics[m]["total"] += 1
            if match_found:
                metrics[m]["relevant"] += 1

        # (Optional) print results after processing all rows
        for m, vals in metrics.items():
            total = vals["total"]
            relevant = vals["relevant"]
            acc = relevant / total if total > 0 else 0
            print(f"Top-{m}: {relevant}/{total} = {acc:.2f}")
    except Exception as e:
        print("Exception happened:", e)
        print("Moving on")

print("===RESULTS===")
for m, vals in metrics.items():
    total = vals["total"]
    relevant = vals["relevant"]
    acc = relevant / total if total > 0 else 0
    print(f"Top-{m}: {relevant}/{total} = {acc:.2f}")

"""
===RESULTS===
Top-1: 852/977 = 0.87
Top-2: 898/977 = 0.92
Top-3: 917/977 = 0.94
Top-4: 926/977 = 0.95
Top-5: 929/977 = 0.95
Top-6: 931/977 = 0.95
Top-7: 934/977 = 0.96
Top-8: 936/977 = 0.96
Top-9: 940/977 = 0.96
Top-10: 943/977 = 0.97
"""

# %%
import matplotlib.pyplot as plt

# === Data ===
m_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
accuracies = [0.87, 0.92, 0.94, 0.95, 0.95, 0.95, 0.96, 0.96, 0.96, 0.97]

# === Step plot ===
plt.figure(figsize=(8, 5))
plt.step(m_values, accuracies, where='post', color='steelblue', linewidth=2)
plt.scatter(m_values, accuracies, color='steelblue', s=60)

# Annotate each point
for m, acc in zip(m_values, accuracies):
    plt.text(m, acc + 0.003, f"{acc:.2f}", ha='center', fontsize=9, fontweight='bold')

# Highlight optimal m
plt.axvline(x=4, color='gray', linestyle='--', linewidth=1)
plt.text(4.1, 0.855, "Optimal m = 4", rotation=90, va='bottom', fontsize=10, fontweight='bold', color='gray')

# Formatting
plt.title("Optimize the Number of Retrieved Chunks (Parameter Configuration)", fontsize=14, fontweight='bold')
plt.xlabel("Number of Retrieved Chunks (m)", fontsize=12)
plt.ylabel("Top-m Retrieval Accuracy", fontsize=12)
plt.xticks(m_values)
plt.ylim(0.85, 1.0)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
