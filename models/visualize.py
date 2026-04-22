"""Visualization script for model evaluation results."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from xgboost import XGBClassifier

from config import MODEL_DIR, TARGET_COL, GROUP_COL
from train import load_data, preprocess_features, get_feature_names


def plot_feature_importance(model_dir: Path, output_dir: Path, top_k: int = 15) -> None:
    """Plot feature importance bar chart."""
    importance_path = model_dir / "feature_importance.json"
    if not importance_path.exists():
        print(f"Feature importance file not found: {importance_path}")
        return

    with open(importance_path) as f:
        importance = json.load(f)

    # Get top K features
    features = list(importance.keys())[:top_k]
    values = [importance[f] for f in features]

    plt.figure(figsize=(10, 6))
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(features)))[::-1]
    bars = plt.barh(features[::-1], values[::-1], color=colors)
    plt.xlabel("Importance Score")
    plt.title("Top Feature Importance (XGBoost)")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'feature_importance.png'}")


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, output_dir: Path) -> None:
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'roc_curve.png'}")


def plot_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray, output_dir: Path) -> None:
    """Plot Precision-Recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curve.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'pr_curve.png'}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path) -> None:
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14,
                     color='white' if cm[i, j] > cm.max() / 2 else 'black')

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Negative", "Positive"])
    plt.yticks([0, 1], ["Negative", "Positive"])
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'confusion_matrix.png'}")


def plot_score_distribution(y_true: np.ndarray, y_prob: np.ndarray, output_dir: Path) -> None:
    """Plot score distribution for positive and negative samples."""
    plt.figure(figsize=(8, 6))

    pos_scores = y_prob[y_true == 1]
    neg_scores = y_prob[y_true == 0]

    plt.hist(neg_scores, bins=50, alpha=0.6, label=f'Negative (n={len(neg_scores)})', color='red')
    plt.hist(pos_scores, bins=50, alpha=0.6, label=f'Positive (n={len(pos_scores)})', color='green')
    plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold=0.5')

    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title("Score Distribution by Class")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "score_distribution.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'score_distribution.png'}")


def plot_metrics_comparison(model_dir: Path, output_dir: Path) -> None:
    """Plot train/val/test metrics comparison."""
    metrics_path = model_dir / "metrics.json"
    if not metrics_path.exists():
        print(f"Metrics file not found: {metrics_path}")
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    splits = ["train", "validation", "test"]
    classification_metrics = ["accuracy", "precision", "recall", "f1", "auroc"]
    ranking_metrics = ["mrr", "precision@1", "ndcg@3"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Classification metrics
    x = np.arange(len(classification_metrics))
    width = 0.25
    for i, split in enumerate(splits):
        values = [metrics[split]["classification"][m] for m in classification_metrics]
        axes[0].bar(x + i * width, values, width, label=split.capitalize())

    axes[0].set_ylabel("Score")
    axes[0].set_title("Classification Metrics by Split")
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels([m.upper() for m in classification_metrics], rotation=15)
    axes[0].legend()
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis='y', alpha=0.3)

    # Ranking metrics
    x = np.arange(len(ranking_metrics))
    for i, split in enumerate(splits):
        values = [metrics[split]["ranking"][m] for m in ranking_metrics]
        axes[1].bar(x + i * width, values, width, label=split.capitalize())

    axes[1].set_ylabel("Score")
    axes[1].set_title("Ranking Metrics by Split")
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels([m.upper() for m in ranking_metrics])
    axes[1].legend()
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "metrics_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'metrics_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(description="Visualize model results")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(MODEL_DIR),
        help="Directory containing model and metrics",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_path = model_dir / "xgboost_model.json"

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    # Create visualization output directory
    output_dir = model_dir / "visualization"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model and data...")
    model = XGBClassifier()
    model.load_model(model_path)

    test_df = load_data("test")
    X_test, y_test = preprocess_features(test_df)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    print("\nGenerating visualizations...")
    plot_feature_importance(model_dir, output_dir)
    plot_roc_curve(y_test, y_prob, output_dir)
    plot_precision_recall_curve(y_test, y_prob, output_dir)
    plot_confusion_matrix(y_test, y_pred, output_dir)
    plot_score_distribution(y_test, y_prob, output_dir)
    plot_metrics_comparison(model_dir, output_dir)

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
