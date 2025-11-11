import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from mistralai import Mistral
import time

client = Mistral(api_key="LMbH1QizQMAyn5zaF6M1VUp61oPwL9Kb")

# Load QA dataset
qa_path = "textwave/qa_resources/question.tsv"
df = pd.read_csv(qa_path, sep="\t")

models = ["mistral-medium-latest"] # "mistral-small-latest"

results = []

# Helper function: cosine similarity between embeddings
def semantic_similarity(text1, text2):
    retry = False
    while True:
        try:
            emb1 = client.embeddings.create(model="mistral-embed", inputs=text1).data[0].embedding
            time.sleep(0.2)
            emb2 = client.embeddings.create(model="mistral-embed", inputs=text2).data[0].embedding
            time.sleep(0.2)
            return cosine_similarity([emb1], [emb2])[0][0]
        except Exception as e:
            print(f"⚠️ Embedding error: {e}")
            if retry:
                return 0.0
            print("⏳ Waiting 60 seconds before retrying embeddings...")
            time.sleep(60)
            retry = True

# Drop rows with missing questions or answers
df = df.dropna(subset=["Question", "Answer"]).reset_index(drop=True)

# Iterate through models
for model in models:
    print(f"\n🔹 Evaluating {model}...")
    model_scores = []

    count = 0
    for _, row in df.iterrows():
        print(count)
        count += 1

        question = str(row["Question"])
        ground_truth = str(row["Answer"])
        difficulty = row["DifficultyFromAnswerer"]

        retry = False
        while True:
            try:
                time.sleep(0.2)
                response = client.chat.complete(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful factual question-answering assistant."},
                        {"role": "user", "content": question}
                    ],
                )
                generated = response.choices[0].message.content.strip()
                score = semantic_similarity(generated, ground_truth)

                model_scores.append({
                    "Model": model,
                    "Question": question,
                    "GroundTruth": ground_truth,
                    "Generated": generated,
                    "Similarity": score,
                    "Difficulty": difficulty
                })
                break
            except Exception as e:
                print(f"❌ Error for question '{question}': {e}")
                if retry:
                    break
                print("⏳ Waiting 60 seconds before retrying chat request...")
                time.sleep(60)
                retry = True

    # Convert to DataFrame
    model_df = pd.DataFrame(model_scores)

    # Overall average similarity
    overall_score = model_df["Similarity"].mean()

    # Difficulty-stratified scores
    stratified = model_df.groupby("Difficulty")["Similarity"].mean().to_dict()

    print(f"\n✅ Results for {model}:")
    print(f"Overall Mean Similarity: {overall_score:.3f}")
    for diff, val in stratified.items():
        print(f"  {diff.capitalize()}: {val:.3f}")

    results.append({
        "model": model,
        "overall": overall_score,
        **{f"{k}_score": v for k, v in stratified.items()}
    })

# Final summary
summary_df = pd.DataFrame(results)
print("\n==============================")
print("📊 Summary Comparison")
print(summary_df)
summary_df.to_csv("textwave/results/generative_baseline_comparison.csv", index=False)

'''
✅ Results for mistral-small-latest:
Overall Mean Similarity: 0.723
  Easy: 0.695
  Hard: 0.742
  Medium: 0.741
  Too easy: 0.688
  Too hard: 0.722
  
  ✅ Results for mistral-medium-latest:
Overall Mean Similarity: 0.727
  Easy: 0.700
  Hard: 0.746
  Medium: 0.746
  Too easy: 0.693
  Too hard: 0.730
  '''

# %%
import matplotlib.pyplot as plt
import numpy as np

# ==== Task 2 (Baseline, no RAG) ====
mistral_small_base = [0.695, 0.741, 0.742, 0.723]   # Easy, Medium, Hard, Overall
mistral_medium_base = [0.700, 0.746, 0.746, 0.727]

# ==== Task 3 (RAG, no reranker) ====
mistral_small_rag = [0.7077, 0.7498, 0.7393, 0.7059]
mistral_medium_rag = [0.7078, 0.7666, 0.7594, 0.7164]

categories = ["Easy", "Medium", "Hard", "Overall"]
x = np.arange(len(categories))
width = 0.18  # smaller for four bars per group

plt.figure(figsize=(11, 6))

# Baseline (no RAG)
plt.bar(x - width*1.5, mistral_small_base, width, label="Small (Baseline)", color="lightskyblue")
plt.bar(x - width/2, mistral_medium_base, width, label="Medium (Baseline)", color="lightgreen")

# RAG (no reranker)
plt.bar(x + width/2, mistral_small_rag, width, label="Small (RAG)", color="dodgerblue")
plt.bar(x + width*1.5, mistral_medium_rag, width, label="Medium (RAG)", color="seagreen")

# Annotate bars
def annotate(bars):
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f"{bar.get_height():.3f}", ha='center', fontsize=9, fontweight='bold')

# Reapply annotation for all bars
for barlist in plt.gca().containers:
    annotate(barlist)

# Formatting
plt.title("Mistral Model Performance: Baseline vs RAG (No Reranker)", fontsize=14, fontweight='bold')
plt.ylabel("Mean Similarity", fontsize=12)
plt.xticks(x, categories)
plt.ylim(0.68, 0.78)
plt.grid(axis='y', linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
