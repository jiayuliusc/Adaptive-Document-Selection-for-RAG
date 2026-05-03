# XGBoost Reranker vs. Top-K Retrieval Baseline

**Project:** Adaptive Document Selection for RAG  
**Evaluation dataset:** Natural Questions (NQ Open), 3,000 questions, 12,000 pairs  
**Test split:** 451 unique questions, 1,804 pairs (25% positive rate, 1 relevant + 3 random docs per question)  
**Top-K:** K = 3  
**Date:** May 2, 2026

---

## Experimental Setup

### Methods compared

| Method | Description |
|--------|-------------|
| **Semantic baseline (RAG top-K)** | Vectorize query and documents with SentenceTransformer (`all-MiniLM-L6-v2`), rank by cosine similarity, select top-3. No learned model. |
| **XGBoost reranker** | Binary classifier trained on 17 features (semantic, lexical, BM25, TF-IDF, lengths, question type). Documents ranked by predicted relevance probability. |

The semantic baseline represents a standard RAG retrieval pipeline: embed → rank by cosine similarity → select top-K. The XGBoost reranker sits on top of that retrieval step and uses additional learned features to reorder the candidates.

Both methods operate on the **same test split** evaluated with the **same metric functions** from `models/evaluate.py`. No test-set leakage: the split is performed at the question level so no question in the test set appears during training.

### Baseline implementation

Implemented in [`baselines/topk_baseline.py`](../baselines/topk_baseline.py). For each question the script:
1. Scores all candidate documents by `semantic_cosine_similarity`
2. Keeps only the top-K documents (discards the rest)
3. Evaluates the selected set using P@1, MRR, and nDCG@K

The `semantic_cosine_similarity` values are precomputed by `experiment.py` using SentenceTransformer embeddings and stored in `data/processed/features_test.csv` — conceptually identical to querying a vector store at retrieval time.

BM25 and TF-IDF are also included as secondary reference points.

---

## Results

### Primary Comparison: Semantic Top-K vs. XGBoost (K = 3)

| Metric | Semantic top-3 | XGBoost | Gain |
|--------|---------------|---------|------|
| **MRR** | 0.841 | **0.855** | +0.014 |
| **Precision@1** | 0.734 | **0.749** | +0.015 |
| **nDCG@3** | 0.875 | **0.882** | +0.007 |
| **nDCG@5** | 0.875 | **0.892** | +0.017 |

### Full Ranking Metrics (Test Set, K = 3)

| Metric | Semantic | BM25 | TF-IDF | XGBoost |
|--------|----------|------|--------|---------|
| **MRR** | 0.841 | 0.476 | 0.477 | **0.855** |
| **Precision@1** | 0.734 | 0.297 | 0.299 | **0.749** |
| **nDCG@1** | 0.734 | 0.297 | 0.299 | **0.749** |
| **Precision@3** | 0.324 | 0.244 | 0.244 | **0.325** |
| **nDCG@3** | 0.875 | 0.541 | 0.542 | **0.882** |

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

### XGBoost vs. Semantic Baseline

XGBoost outperforms the semantic top-K baseline on every metric:

- **MRR:** +0.014 (0.841 → 0.855)
- **Precision@1:** +0.015 (0.734 → 0.749)
- **nDCG@3:** +0.007 (0.875 → 0.882)

The gains are modest but consistent. This is expected — `semantic_cosine_similarity` is XGBoost's dominant feature at 36% importance, so the learned model's main advantage is combining semantics with document length, question type, and interaction terms rather than replacing the semantic signal.

**Key insight:** The semantic top-K baseline already captures ~98% of XGBoost's MRR. The reranker is most valuable when embeddings alone are insufficient — e.g. when questions and relevant documents share little surface form, or when document length and question type carry additional discriminative signal.

### XGBoost vs. Sparse Retrieval (Reference)

BM25 and TF-IDF perform far below both the semantic baseline and XGBoost, confirming that NQ-style question-answer pairs have insufficient lexical overlap for sparse retrieval to work on its own.

| Metric | BM25 | XGBoost | Gain |
|--------|------|---------|------|
| MRR | 0.476 | 0.855 | +0.379 |
| Precision@1 | 0.297 | 0.749 | +0.452 |
| nDCG@3 | 0.541 | 0.882 | +0.341 |

---

## Summary

```
XGBoost reranker > Semantic top-K >> BM25 ≈ TF-IDF
```

The XGBoost reranker improves over the semantic RAG baseline by **+1.4% MRR** and **+1.5% P@1** by learning to combine multiple signals. The bulk of the retrieval quality comes from the dense embedding step, with the learned reranker providing a consistent incremental improvement on top.
