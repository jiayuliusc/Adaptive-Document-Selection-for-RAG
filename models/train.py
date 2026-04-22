"""XGBoost training pipeline for relevance prediction."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from config import (
    DATA_DIR,
    MODEL_DIR,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
    GROUP_COL,
    QUESTION_TYPES,
    DEFAULT_XGBOOST_PARAMS,
)
from evaluate import compute_all_metrics, print_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


def load_data(split: str) -> pd.DataFrame:
    """Load feature data for a given split."""
    path = DATA_DIR / f"features_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {path}\n"
            "Run experiment.py first to generate features."
        )
    return pd.read_csv(path)


def preprocess_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess features for model training."""
    # Extract numeric features
    X_numeric = df[NUMERIC_FEATURES].copy()

    # Handle NaN in bm25_score (fill with 0)
    X_numeric["bm25_score"] = X_numeric["bm25_score"].fillna(0)

    # One-hot encode question_type
    for qtype in QUESTION_TYPES:
        X_numeric[f"qtype_{qtype}"] = (df["question_type"] == qtype).astype(int)

    X = X_numeric.values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.int32)

    return X, y


def get_feature_names() -> list:
    """Get feature names after preprocessing."""
    names = NUMERIC_FEATURES.copy()
    names.extend([f"qtype_{qtype}" for qtype in QUESTION_TYPES])
    return names


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict = None,
) -> XGBClassifier:
    """Train XGBoost classifier."""
    params = params or DEFAULT_XGBOOST_PARAMS.copy()

    model = XGBClassifier(**params)

    LOGGER.info("Training XGBoost model...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    return model


def evaluate_model(
    model: XGBClassifier,
    df: pd.DataFrame,
    X: np.ndarray,
    split_name: str,
) -> dict:
    """Evaluate model on a dataset split."""
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = compute_all_metrics(
        df=df,
        y_pred=y_pred,
        y_prob=y_prob,
        label_col=TARGET_COL,
        group_col=GROUP_COL,
    )

    print_metrics(metrics, split_name)
    return metrics


def save_model(
    model: XGBClassifier,
    metrics: dict,
    output_dir: Path,
) -> None:
    """Save model and metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "xgboost_model.json"
    model.save_model(model_path)
    LOGGER.info("Model saved to %s", model_path)

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    LOGGER.info("Metrics saved to %s", metrics_path)

    # Save feature importance
    feature_names = get_feature_names()
    importance = dict(zip(feature_names, model.feature_importances_.tolist()))
    importance_sorted = dict(sorted(importance.items(), key=lambda x: -x[1]))

    importance_path = output_dir / "feature_importance.json"
    with open(importance_path, "w") as f:
        json.dump(importance_sorted, f, indent=2)
    LOGGER.info("Feature importance saved to %s", importance_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost relevance model")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(MODEL_DIR),
        help="Directory to save model and results",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=DEFAULT_XGBOOST_PARAMS["max_depth"],
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_XGBOOST_PARAMS["learning_rate"],
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=DEFAULT_XGBOOST_PARAMS["n_estimators"],
    )
    args = parser.parse_args()

    # Load data
    LOGGER.info("Loading data...")
    train_df = load_data("train")
    val_df = load_data("validation")
    test_df = load_data("test")

    LOGGER.info("Train: %d rows, Val: %d rows, Test: %d rows",
                len(train_df), len(val_df), len(test_df))

    # Preprocess
    LOGGER.info("Preprocessing features...")
    X_train, y_train = preprocess_features(train_df)
    X_val, y_val = preprocess_features(val_df)
    X_test, y_test = preprocess_features(test_df)

    LOGGER.info("Feature matrix shape: %s", X_train.shape)

    # Train
    params = DEFAULT_XGBOOST_PARAMS.copy()
    params.update({
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "n_estimators": args.n_estimators,
    })

    model = train_model(X_train, y_train, X_val, y_val, params)

    # Evaluate
    LOGGER.info("Evaluating model...")
    train_metrics = evaluate_model(model, train_df, X_train, "Train")
    val_metrics = evaluate_model(model, val_df, X_val, "Validation")
    test_metrics = evaluate_model(model, test_df, X_test, "Test")

    all_metrics = {
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
        "params": params,
    }

    # Save
    output_dir = Path(args.output_dir)
    save_model(model, all_metrics, output_dir)

    LOGGER.info("Done!")


if __name__ == "__main__":
    main()
