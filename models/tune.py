"""Hyperparameter tuning for XGBoost model."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import ParameterGrid
from xgboost import XGBClassifier

from config import DATA_DIR, MODEL_DIR, DEFAULT_XGBOOST_PARAMS
from train import load_data, preprocess_features, get_feature_names
from evaluate import compute_all_metrics, print_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


def tune_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_df,
    param_grid: dict,
) -> tuple:
    """Grid search over hyperparameters."""
    best_auroc = 0
    best_params = None
    best_model = None
    results = []

    grid = list(ParameterGrid(param_grid))
    LOGGER.info("Tuning over %d parameter combinations...", len(grid))

    for i, params in enumerate(grid):
        full_params = DEFAULT_XGBOOST_PARAMS.copy()
        full_params.update(params)

        model = XGBClassifier(**full_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = compute_all_metrics(val_df, y_pred, y_prob)
        auroc = metrics["classification"]["auroc"]
        mrr = metrics["ranking"]["mrr"]

        results.append({
            "params": params,
            "auroc": auroc,
            "mrr": mrr,
            "f1": metrics["classification"]["f1"],
        })

        if auroc > best_auroc:
            best_auroc = auroc
            best_params = full_params
            best_model = model

        if (i + 1) % 10 == 0:
            LOGGER.info("Progress: %d/%d, best AUROC: %.4f", i + 1, len(grid), best_auroc)

    return best_model, best_params, results


def main():
    parser = argparse.ArgumentParser(description="Tune XGBoost hyperparameters")
    parser.add_argument("--quick", action="store_true", help="Quick search with fewer params")
    args = parser.parse_args()

    # Load data
    LOGGER.info("Loading data...")
    train_df = load_data("train")
    val_df = load_data("validation")
    test_df = load_data("test")

    X_train, y_train = preprocess_features(train_df)
    X_val, y_val = preprocess_features(val_df)
    X_test, y_test = preprocess_features(test_df)

    # Define search space
    if args.quick:
        param_grid = {
            "max_depth": [5, 7, 9],
            "learning_rate": [0.05, 0.1],
            "n_estimators": [100, 200],
            "scale_pos_weight": [1, 3],  # Handle class imbalance
        }
    else:
        param_grid = {
            "max_depth": [4, 6, 8, 10],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [100, 200, 300],
            "min_child_weight": [1, 3, 5],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
            "scale_pos_weight": [1, 2, 3],  # 1:3 negative ratio
        }

    # Tune
    best_model, best_params, results = tune_model(
        X_train, y_train, X_val, y_val, val_df, param_grid
    )

    LOGGER.info("Best parameters: %s", best_params)

    # Evaluate best model on test
    LOGGER.info("Evaluating best model on test set...")
    y_prob = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    test_metrics = compute_all_metrics(test_df, y_pred, y_prob)
    print_metrics(test_metrics, "Test (Best Model)")

    # Save
    output_dir = MODEL_DIR / "tuned"
    output_dir.mkdir(parents=True, exist_ok=True)

    best_model.save_model(output_dir / "xgboost_model.json")

    with open(output_dir / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    with open(output_dir / "tuning_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Feature importance
    feature_names = get_feature_names()
    importance = dict(zip(feature_names, best_model.feature_importances_.tolist()))
    importance_sorted = dict(sorted(importance.items(), key=lambda x: -x[1]))
    with open(output_dir / "feature_importance.json", "w") as f:
        json.dump(importance_sorted, f, indent=2)

    LOGGER.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
