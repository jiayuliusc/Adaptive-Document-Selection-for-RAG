# Model Documentation

## Overview

This module implements an XGBoost classifier to predict query-document relevance for RAG retrieval. Given a question and a candidate document, the model outputs a relevance score indicating how likely the document answers the question.

## Files

| File | Purpose |
|------|---------|
| `config.py` | Configuration constants, feature lists, and tuned hyperparameters |
| `train.py` | Main training pipeline: loads data, preprocesses features, trains XGBoost, evaluates on all splits |
| `evaluate.py` | Evaluation metrics: classification (AUROC, F1, Precision, Recall) and ranking (MRR, P@K, nDCG@K) |
| `tune.py` | Hyperparameter tuning via grid search over 2916 parameter combinations |
| `visualize.py` | Generates visualization plots (ROC curve, confusion matrix, feature importance, etc.) |

## Features Used

The model uses 17 features derived from question-document pairs:

- **Semantic**: `semantic_cosine_similarity` (Sentence-BERT embeddings)
- **Lexical**: `token_overlap`, `token_overlap_ratio`, `tfidf_similarity`, `bm25_score`
- **Document**: `document_length`
- **Query**: `query_length`, `named_entity_count`
- **Interaction**: `similarity_times_doc_length`, `normalized_overlap`
- **Question Type**: one-hot encoded (who, what, when, where, why, how, other)

## Tuned Hyperparameters

After grid search over 2916 combinations:

```python
{
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "min_child_weight": 1,
    "subsample": 0.9,
    "colsample_bytree": 0.7,
    "scale_pos_weight": 3  # handles 1:3 class imbalance
}
```

## Results

Evaluated on 10,000 questions (40,000 pairs: 1 positive + 3 negatives per question).

### Test Set Performance

| Metric | Score |
|--------|-------|
| AUROC | 0.883 |
| F1 | 0.673 |
| Precision | 0.573 |
| Recall | 0.815 |
| MRR | 0.865 |
| Precision@1 | 0.759 |
| nDCG@3 | 0.890 |

### Key Findings

1. **Semantic similarity is the most important feature** (31% importance), confirming that embedding quality matters.
2. **Question type features are highly predictive**, suggesting different question types have distinct relevance patterns.
3. **Setting `scale_pos_weight=3`** significantly improved recall (55% → 81%) by addressing class imbalance.
4. The model correctly identifies the best answer for ~76% of questions (P@1).

## Usage

```bash
# Train model
python train.py

# Hyperparameter tuning
python tune.py --quick  # fast (24 combinations)
python tune.py          # full (2916 combinations)

# Generate visualizations
python visualize.py
```

## Output

Models and results are saved to `models/saved/`:
- `xgboost_model.json` - trained model
- `metrics.json` - evaluation metrics
- `feature_importance.json` - feature importance scores
- `visualization/` - plots (ROC, PR curve, confusion matrix, etc.)
