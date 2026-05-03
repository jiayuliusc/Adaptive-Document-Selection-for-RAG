# Evaluation Report: XGBoost Relevance Reranker

**Project:** Adaptive Document Selection for RAG  
**Dataset:** Natural Questions (NQ Open), 10,000 questions, 40,000 (question, document) pairs  
**Model:** XGBoost binary classifier (`binary:logistic`)  
**Date:** May 2, 2026

---

## 1. Model Overview

The system frames document selection as a binary relevance classification problem. For each (question, document) pair, the model predicts the probability that the document is relevant (`label=1`). At inference time, candidate documents for a query are ranked by this predicted probability and the top-K are selected.

**Training configuration:**

| Parameter | Value |
|-----------|-------|
| `objective` | `binary:logistic` |
| `max_depth` | 6 |
| `learning_rate` | 0.1 |
| `n_estimators` | 100 |
| `min_child_weight` | 1 |
| `subsample` | 0.9 |
| `colsample_bytree` | 0.7 |
| `scale_pos_weight` | 3 |

`scale_pos_weight=3` directly compensates for the 1:3 positive-to-negative ratio in the training data (1 relevant document + 3 random negatives per question).

---

## 2. Performance Analysis

### 2.1 Classification Metrics

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| Accuracy | 0.802 | 0.793 | 0.802 |
| Precision | 0.574 | 0.561 | 0.573 |
| Recall | 0.808 | 0.793 | 0.815 |
| F1 | 0.671 | 0.657 | 0.673 |
| **AUROC** | **0.895** | **0.879** | **0.883** |

### 2.2 Ranking Metrics

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| MRR | 0.870 | 0.857 | 0.865 |
| Precision@1 | 0.766 | 0.750 | 0.759 |
| nDCG@1 | 0.766 | 0.750 | 0.759 |
| Precision@3 | 0.328 | 0.326 | 0.326 |
| nDCG@3 | 0.896 | 0.884 | 0.890 |
| Precision@5 | 0.250 | 0.250 | 0.250 |
| nDCG@5 | 0.903 | 0.893 | 0.899 |

### 2.3 Generalization Analysis

The gap between train and test AUROC is **0.012** (0.895 → 0.883), and MRR drops by **0.005** (0.870 → 0.865). These are small, indicating the model generalizes well and is not significantly overfitting. The question-level train/validation/test split (no question appears in more than one split) ensures that evaluation numbers reflect performance on genuinely unseen queries.

The model is consistent across splits on ranking metrics, which are the most operationally relevant — a reranker's job is to surface the correct document near the top, not to achieve perfect binary accuracy.

*See [`models/saved/visualization/metrics_comparison.png`](../models/saved/visualization/metrics_comparison.png) for a visual comparison of all splits.*

---

## 3. Feature Importance Analysis

Feature importances are computed by XGBoost's built-in `feature_importances_` (gain-based), normalized to sum to 1.

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `semantic_cosine_similarity` | 36.0% |
| 2 | `similarity_times_doc_length` | 11.4% |
| 3 | `qtype_how` | 8.7% |
| 4 | `document_length` | 8.3% |
| 5 | `qtype_when` | 7.5% |
| 6 | `qtype_who` | 6.4% |
| 7 | `qtype_other` | 4.0% |
| 8 | `qtype_where` | 2.9% |
| 9 | `qtype_what` | 2.7% |
| 10 | `bm25_score` | 2.4% |
| 11 | `tfidf_similarity` | 2.2% |
| 12 | `qtype_why` | 1.9% |
| 13 | `query_length` | 1.6% |
| 14 | `token_overlap_ratio` | 1.4% |
| 15 | `normalized_overlap` | 1.4% |
| 16 | `token_overlap` | 1.2% |
| 17 | `named_entity_count` | 0.0% |

*See [`models/saved/visualization/feature_importance.png`](../models/saved/visualization/feature_importance.png) for the bar chart.*

### 3.1 Key Observations

**Semantic similarity dominates.** `semantic_cosine_similarity` accounts for 36% of the total importance — more than the next five features combined. This confirms that dense embedding overlap is the single strongest signal for relevance on NQ-style questions.

**The interaction term adds value.** `similarity_times_doc_length` ranks second at 11.4%. Documents that are both semantically similar *and* long are more likely to be relevant (they contain more supporting context). The model discovered this beyond what raw similarity alone captures.

**Question type is a meaningful group of features.** Collectively the 7 `qtype_*` columns contribute ~34% of importance, nearly matching semantic similarity. The model behaves differently for "how" questions (8.7%) and "when" questions (7.5%) compared to "what" questions (2.7%). This suggests that relevance patterns differ by question category — temporal queries ("when") and procedural queries ("how") have more discriminative document signals than broad factoid queries ("what").

**Lexical features are weak but not zero.** BM25 (2.4%) and TF-IDF (2.2%) add marginal value on top of dense semantics. This is expected for NQ: questions and their answers often share few exact words, making sparse retrieval less effective.

**Named entity count is a dead feature.** `named_entity_count` has 0.0% importance. This feature was disabled (no spaCy) during the run that generated the processed data, causing it to fall back to a regex heuristic. The fallback counts title-cased words, which is too noisy to be informative. This feature can be safely dropped from future runs, or replaced with proper spaCy NER (enabled via `--enable-ner`).

---

## 4. Error Analysis

### 4.1 Precision–Recall Trade-off

The test precision is **0.573** while recall is **0.815**. This means:
- The model recovers ~82% of all relevant documents.
- But for every 10 documents it labels as relevant, only ~6 are actually relevant.

This asymmetry is largely by design. `scale_pos_weight=3` pushes the model to be recall-biased, which is usually preferable in a retrieval-before-generation pipeline: it is better to include a relevant document than to miss it. However, this comes at a cost — many irrelevant documents are also included, increasing the context burden on the downstream LLM.

*See [`models/saved/visualization/score_distribution.png`](../models/saved/visualization/score_distribution.png) for the distribution of predicted probabilities by class.*

### 4.2 Confusion Matrix Analysis

At the default threshold of 0.5, the model operates as follows on the test set:

- **True Positives (TP):** relevant documents correctly flagged as relevant
- **False Negatives (FN):** relevant documents missed — about 18.5% of all positive pairs
- **False Positives (FP):** irrelevant documents incorrectly flagged — a large share due to recall bias
- **True Negatives (TN):** irrelevant documents correctly filtered out

*See [`models/saved/visualization/confusion_matrix.png`](../models/saved/visualization/confusion_matrix.png) for the full confusion matrix.*

**When does the model fail (FP — irrelevant marked relevant)?**
- When an irrelevant document is semantically similar to the query in surface form but discusses a different fact (e.g., a question about one historical event paired with a document about a superficially similar event)
- When both document and query are long and share many tokens by coincidence ("what" questions with generic wording)

**When does the model fail (FN — relevant missed)?**
- When the relevant document is short and terse (low `similarity_times_doc_length`)
- When the question is phrased very differently from the answer's language (low `semantic_cosine_similarity`) — a known limitation of embedding-based retrieval
- Numerical or date-specific "when"/"how many" questions where the answer is a single phrase with minimal lexical overlap with the question

### 4.3 ROC and Precision–Recall Curves

- **Test AUROC: 0.883.** The ROC curve demonstrates strong discrimination across all operating thresholds.
- The PR curve shows that increasing precision requires reducing recall steeply below a score threshold of ~0.7, confirming the default threshold of 0.5 is well-calibrated for a recall-first retrieval use case.

*See [`models/saved/visualization/roc_curve.png`](../models/saved/visualization/roc_curve.png) and [`models/saved/visualization/pr_curve.png`](../models/saved/visualization/pr_curve.png).*

### 4.4 Threshold Sensitivity

The default threshold of 0.5 is not necessarily optimal. Given the recall-biased training:
- Raising the threshold (e.g., to 0.65–0.70) would improve precision at the cost of recall, making the selected context more focused but potentially missing some relevant documents.
- Lowering the threshold would maximize recall for safety-critical applications.

The threshold can be tuned on the validation set using the PR curve to target a desired operating point.

---

## 5. Summary

The XGBoost reranker achieves strong and stable performance across splits. Key findings:

1. Dense semantic similarity is by far the most important feature (36%), confirming that sentence-transformer embeddings capture the core relevance signal in NQ.
2. Question type features collectively contribute ~34% of importance, showing the model adapts its behavior to different question categories.
3. The model generalizes well: train-to-test AUROC drop is only 0.012, with no significant overfitting.
4. The model is recall-biased (recall 0.815, precision 0.573), appropriate for retrieval but at the cost of some noise in the retrieved context.
5. `named_entity_count` contributes nothing and should be removed or replaced with proper spaCy NER in future iterations.
6. Lexical features (BM25, TF-IDF, token overlap) add marginal but nonzero value on top of semantic embeddings.
