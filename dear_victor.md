# Victor

## What's Already Done

### Data Pipeline (experiment.py)
- Loads Natural Questions dataset from HuggingFace
- Creates positive/negative pairs (1 correct + N random documents per question)
- Splits by question ID (70/15/15) to prevent leakage
- Computes 17 features: semantic similarity, lexical overlap, BM25, TF-IDF, etc.

### Model Training (models/)
- XGBoost classifier trained on 10K questions (40K pairs)
- Hyperparameters tuned via grid search (2916 combinations)
- Best config: `max_depth=6, scale_pos_weight=3, subsample=0.9`

### Current Results (Test Set)
| Metric | Score |
|--------|-------|
| AUROC | 0.883 |
| Recall | 0.815 |
| Precision@1 | 0.759 |
| MRR | 0.865 |
| nDCG@3 | 0.890 |

---

## How to Run

### 1. Setup
```bash
pip install -r requirements.txt
```

### 2. Generate Features
```bash
python experiment.py --sample-size 10000 --negatives-per-question 3
```

### 3. Train Model
```bash
cd models
python train.py
```

### 4. Hyperparameter Tuning (optional)
```bash
python tune.py --quick   # ~2 min
python tune.py           # ~15 min, full search
```

### 5. Generate Visualizations
```bash
python visualize.py
# outputs to models/saved/visualization/
```

---

## Your Tasks

### Task 1: Evaluation Report

Write a short evaluation report covering:
1. Model performance analysis (use metrics from `models/saved/metrics.json`)
2. Feature importance analysis (see `models/saved/feature_importance.json`)
3. Error analysis: when does the model fail?
4. Visualizations (from `models/saved/visualization/`)

### Task 2: Baseline Comparison

Implement a **regular RAG + top-K retrieval baseline** for comparison:

1. **Baseline approach**:
   - Use only semantic similarity (or BM25) to rank documents
   - Select top-K candidates
   - No learned reranking

2. **Implementation**:
   - Create `baselines/topk_baseline.py`
   - For each question, rank all candidate documents by similarity score
   - Compute the same metrics: P@1, MRR, nDCG@K

3. **Compare**:
   - XGBoost reranker vs. pure top-K retrieval
   - Show improvement from learned features

### Deliverables
- `reports/evaluation_report.md` - model evaluation
- `baselines/topk_baseline.py` - baseline implementation
- `reports/comparison.md` - XGBoost vs baseline results

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `experiment.py` | Data pipeline entry |
| `features.py` | Feature computation |
| `models/train.py` | Model training |
| `models/evaluate.py` | Metrics computation |
| `models/visualize.py` | Plot generation |
| `models/models.md` | Model documentation |
| `Features.md` | Feature definitions |
