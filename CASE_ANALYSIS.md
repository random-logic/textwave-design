# Chunking Strategy Performance (Text Preprocessing and Index Selection)

**Figure 1** - The relevant chunk accuracy between fixed-length (100 characters per chunk, maximum 10 characters overlapping) and sentence-based (3 sentences per chunk, maximum 1 overlapping sentence) chunking strategies.

![img1.png](img/img1.png)

Figure 1 shows that sentence-based chunking performs with significantly higher accuracy (0.89) than fixed-length chunking (0.45). This improvement likely occurs because sentence-based chunking respects natural linguistic boundaries, keeping contextually related ideas together. In contrast, fixed-length chunking can arbitrarily split sentences or phrases, causing semantic fragmentation that reduces retrieval precision and overall contextual coherence. While sentence-based chunking dominates in performance, the tradeoff is speed. Its variable chunk sizes make indexing and storage less predictable, and sentence boundary detection adds preprocessing overhead. On the other hand, fixed-length chunking, although less contextually accurate, offers uniform sizes that simplify implementation, speed up indexing, and improve memory efficiency. Given the substantial accuracy gap between fixed-length and sentence-based chunking, the additional computational cost is justified, making sentence-based chunking the preferred approach.

# Generative Model Performance Comparison (Baseline Selection)

**Figure 2** - Generative Model Performance Comparison without RAG.

![img2.png](img/img2.png)

Figure 2 shows that for all question types, the Medium model consistently beats the Small model, likely because the Medium model’s larger parameter count enables it to capture more nuanced patterns in language and reasoning. This holds true even for questions that were relatively easy, which means that even these types of questions have enough nuanced patterns in language and reasoning that the small model cannot capture it all. As a result, it can generate more precise and contextually appropriate responses, while the Small model tends to produce shorter, less detailed outputs and sometimes misses subtle contextual cues that affect accuracy. While using the medium model is slightly more accurate, the tradeoff is speed and resources. The medium model has more parameters so will take more GPUs to run and inference may be slower. The verdict is to use the small model for speed while to use the medium model if that slight increase in accuracy is important.

# Retrieval-Augmented Generative Model Performance Comparison (Architecture Selection)

**Figure 3** - Mistral Model Performance with no RAG vs with RAG (No Reranker).

![img3.png](img/img3.png)

Figure 3 shows mixed results for the Small model, while the Medium model with RAG consistently outperforms its non-RAG counterpart. For easy and medium questions, the Small model benefits from RAG, achieving higher accuracy than without it. However, for harder questions and overall performance, the Small model with RAG performs worse. This may be because, when presented with additional retrieved information, the smaller model becomes overwhelmed or confused—its limited parameter capacity makes it less effective at integrating and reasoning over complex contextual data. The medium model, on the other hand, has more parameters so it does not have this struggle and therefore can perform with higher accuracy. As mentioned previously, the tradeoff to using the medium model is speed and resource usage. The verdict is to use the small model with RAG for easy questions since it performs identical to the medium model with RAG while taking less compute power. On the other hand, the boost in accuracy that the medium model with RAG offers makes the more expensive compute justifiable, meaning that in these situations, the medium model with RAG is the winner.

RAG usually improves performance, justifying the additional overhead it introduces. But, for small models on hard questions, it can do the opposite. Hard questions introduce ambiguous retrieval, long context passages, and reasoning steps that small models struggle to integrate. Instead of filtering or rejecting noisy context, a small model tends to overfit to it, drowning out its internal priors and reducing answer quality. Medium and large models can compensate, but small models cannot—so RAG backfires specifically on hard queries.

# Reranker Performance Comparison (Architecture Selection)

**Figure 4** - Performance comparison between different types of rerankers.

![img4.png](img/img4.png)

Figure 4 shows that the LLM-Reranker subtly outperforms all other rankers across models. This is expected because LLM-based reranking provides deeper semantic understanding than TF-IDF, BM25, or embedding cosine similarity, which rely more on surface-level overlap. The LLM-Reranker can interpret intent, resolve paraphrasing, and judge whether a passage truly answers the question rather than merely appearing similar. This matters in our dataset, where many queries involve paraphrased phrasing or multi-hop context that simple similarity metrics miss. The LLM-Reranker also more effectively filters out high-similarity but irrelevant distractor chunks, especially for hard questions. These small but consistent improvements in selecting the most meaningful evidence lead to the modest accuracy gains observed in Figure 4.

While the LLM-Reranker offers the highest accuracy, it also incurs the largest computational and latency overhead due to running an additional LLM pass for every candidate chunk. Traditional rerankers such as BM25 or dense-embedding cosine similarity are significantly cheaper and faster, making them more suitable when compute resources are limited or when low-latency retrieval is required. If the priority is efficiency and minimal resource usage, a lightweight vector-based reranker is the most cost‑effective option. However, if accuracy is the primary concern—particularly for applications involving complex or high‑stakes queries—the LLM-Reranker remains the recommended choice despite its higher resource cost.

In addition, there is a significant improvement of accuracy when comparing any reranker with no reranker across all questions. This shows that using any reranker is worth the performance overhead tradeoff.

# Optimize the Number of Retrieved Chunks (Parameter Configuration)

**Figure 5** - Chunk retrieval accuracy vs number of retrieved chunks.

![img5.png](img/img5.png)

Figure 5 is a step plot showing the marginal increase in accuracy as the number of retrieved chunks grows. The optimal point occurs at m = 4, where additional chunks provide only minimal gains. This strikes a balanced tradeoff between performance and inference speed: while accuracy initially improves with more retrieved context, excessive retrieval slows the model and increases the risk of distraction from irrelevant passages.

Importantly, this conclusion is influenced by characteristics of the underlying dataset. The QA corpus exhibits limited redundancy—most questions can be answered using a small set of highly relevant snippets—so retrieving many additional chunks increases noise faster than it increases useful signal. The dataset’s domain consistency also matters: because questions are often tied to specific, tightly scoped facts, additional chunks rarely add new information beyond the fourth retrieval. Moreover, hard questions in the dataset tend to have semantically similar distractor chunks, meaning larger retrieval sets increase the probability of pulling misleading or tangential context that can degrade model performance. Together, these dataset-specific factors contribute to the sharp diminishing returns observed beyond m = 4, reinforcing the effectiveness of a small, targeted retrieval depth.
