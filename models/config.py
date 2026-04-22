"""Configuration constants for model training."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "saved"

# Feature columns for training
NUMERIC_FEATURES = [
    "semantic_cosine_similarity",
    "token_overlap",
    "token_overlap_ratio",
    "tfidf_similarity",
    "bm25_score",
    "document_length",
    "query_length",
    "named_entity_count",
    "similarity_times_doc_length",
    "normalized_overlap",
]

CATEGORICAL_FEATURES = ["question_type"]

# All feature columns
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Target and grouping columns
TARGET_COL = "label"
GROUP_COL = "question_id"

# Question type categories for encoding
QUESTION_TYPES = ["who", "what", "when", "where", "why", "how", "other"]

# Default XGBoost hyperparameters (tuned)
DEFAULT_XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "min_child_weight": 1,
    "subsample": 0.9,
    "colsample_bytree": 0.7,
    "scale_pos_weight": 3,
    "random_state": 42,
    "n_jobs": -1,
}

# Hyperparameter search space for tuning
TUNE_PARAM_GRID = {
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "n_estimators": [50, 100, 200],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
}
