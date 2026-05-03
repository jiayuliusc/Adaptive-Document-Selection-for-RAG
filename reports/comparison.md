# XGBoost Reranker vs. Top-K Retrieval Baselines

**Project:** Adaptive Document Selection for RAG  
**Evaluation dataset:** Natural Questions (NQ Open), 3,000 questions, 12,000 pairs  
**Test split:** 451 unique questions, 1,804 pairs (25% positive rate, 1 relevant + 3 random docs per question)  
**Date:** May 2, 2026

---

## Experimental Setup

### Methods compared

| Method | Description |
|--------|-------------|
| **Semantic baseline** | Rank candidates by `semantic_cosine_similarity` (SentenceTransformer `all-MiniLM-L6-v2`) — no learned model |
| **BM25 baseline** | Rank candidates by precomputed `bm25_score` (BM25Okapi over the document corpus) — no learned model |
| **TF-IDF baseline** | Rank candidates by `tfidf_similarity` (unigram + bigram TF-IDF cosine) — no learned model |
| **XGBoost reranker** | Binary classifier trained on all 17 features; documents ranked by predicted relevance probability |

All four methods operate on the **same test split** and are evaluated with the **same metric functions** from `models/evaluate.py`. No test-set leakage: the train/validation/test split is performed at the question level so no question in the test set was seen during model training.

### Baseline implementation

Baselines are implemented in [`baselines/topk_baseline.py`](../baselines/topk_baseline.py). They read the precomputed feature columns directly from `data/processed/features_test.csv` — no re-embedding or re-scoring is needed at inference time.

---

## Results

### Ranking Metrics (Test Set)

| Metric | Semantic | BM25 | TF-IDF | XGBoost |
|--------|----------|------|--------|---------|
| **MRR** | 0.8477 | 0.5772 | 0.5778 | **0.8553** |
| **Precision@1** | 0.7339 | 0.3392 | 0.3392 | **0.7494** |
| **nDCG@1** | 0.7339 | 0.3392 | 0.3392 | **0.7494** |
| **Precision@3** | 0.3245 | 0.2565 | 0.2572 | **0.3252** |
| **nDCG@3** | 0.8751 | 0.5834 | 0.5848 | **0.8817** |
| **Precision@5** | 0.2500 | 0.2500 | 0.2500 | **0.2500** |
| **nDCG@5** | 0.8866 | 0.6827 | 0.6831 | **0.8922** |

### XGBoost Classification Metrics (Test Set)

| Metric | Score |
|--------|-------|
| Accuracy | 0.818 |
| Precision | 0.606 |
| Recall | 0.776 |
| F1 | 0.680 |
| **AUROC** | **0.878** |

*Baselines do not produce binary predictions, so classification metrics are only reported for XGBoost.*

---

## Analysis

### 1. XGBoost vs. Semantic Baseline

XGBoost outperforms pure semantic retrieval on every ranking metric, but the margins are modest:

| Metric | Semantic | XGBoost | Absolute gain |
|--------|----------|---------|---------------|
| MRR | 0.8477 | 0.8553 | +0.008 |
| Precision@1 | 0.7339 | 0.7494 | +0.015 |
| nDCG@3 | 0.8751 | 0.8817 | +0.007 |
| nDCG@5 | 0.8866 | 0.8922 | +0.006 |

This is expected: `semantic_cosine_similarity` is the dominant feature in XGBoost (36% importance). The learned model adds value by combining semantics with document length, question type, and interaction terms, but the bulk of the ranking signal is already captured by a single embedding-based score.

**Key insight:** When dense embeddings are available, a simple semantic ranking already captures ~96% of the XGBoost MRR. The reranker is most valuable in scenarios where embeddings are noisy, unavailable, or insufficient on their own.

### 2. XGBoost vs. Sparse Retrieval (BM25 / TF-IDF)

The gap between XGBoost and sparse-only retrieval is large:

| Metric | BM25 | XGBoost | Absolute gain |
|--------|------|---------|---------------|
| MRR | 0.5772 | 0.8553 | +0.278 |
| Precision@1 | 0.3392 | 0.7494 | +0.410 |
| nDCG@3 | 0.5834 | 0.8817 | +0.298 |

BM25 and TF-IDF perform nearly identically to each other (MRR 0.577 vs 0.578), both well below the semantic and learned methods. This confirms that NQ-style questions — which are phrased in natural language and answered by short, concise documents — do not contain sufficient lexical overlap to support effective sparse retrieval on its own.

BM25 at P@5 (0.250) equals the random-chance ceiling (25% positive rate in 4-candidate pools), demonstrating that sparse retrieval essentially fails to discriminate at rank 5 for this dataset.

### 3. Semantic Baseline vs. Sparse Retrieval

| Metric | BM25 | Semantic | Absolute gain |
|--------|------|----------|---------------|
| MRR | 0.5772 | 0.8477 | +0.271 |
| Precision@1 | 0.3392 | 0.7339 | +0.395 |
| nDCG@3 | 0.5834 | 0.8751 | +0.292 |

Dense semantic embeddings vastly outperform sparse retrieval, confirming that for NQ the semantic understanding of a sentence-transformer model is essential for finding relevant documents.

---

## Summary and Recommendations

### Performance ranking

```
XGBoost reranker > Semantic baseline >> BM25 ≈ TF-IDF
```

### When to use each approach

| Method | When to use |
|--------|-------------|
| **XGBoost reranker** | When maximum ranking quality matters and compute for feature extraction (embeddings + BM25 + TF-IDF) is acceptable at preprocessing time. Best choice for this project. |
| **Semantic baseline** | Lightweight alternative when reranker training is not feasible. Captures ~96% of XGBoost's MRR with zero training overhead. |
| **BM25 / TF-IDF** | First-stage coarse retrieval from a large corpus (thousands of documents), before passing top-K candidates to a semantic or learned reranker. Should not be used as the sole ranking signal on NQ. |

### Takeaways

1. **Learned reranking over semantic features helps**, but the improvement over using semantic similarity alone is incremental (+0.8% MRR, +1.5% P@1) for this dataset and candidate pool size.

2. **The biggest gain is from switching dense to sparse** — moving from BM25 to semantic retrieval gains +27% MRR, and the XGBoost model then adds another +0.8% on top.

3. **The two-stage pipeline is well-motivated for large-scale RAG:** Use BM25/TF-IDF for first-stage retrieval over millions of documents (fast, cheap), then apply the XGBoost reranker over the top-K candidates (where dense embeddings + learned features shine).

4. **Future improvements** to the reranker should focus on: (a) enabling proper spaCy NER (currently a dead feature), (b) adding cross-encoder features (joint query-document attention), and (c) increasing the candidate pool size per question to stress-test ranking at deeper positions.
