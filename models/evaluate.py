"""Evaluation metrics for relevance prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from typing import Dict, List, Any


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """Compute standard classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auroc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
    }


def precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int = 1) -> float:
    """Compute Precision@K for a single query group."""
    if len(y_true) == 0:
        return 0.0
    k = min(k, len(y_true))
    top_k_indices = np.argsort(y_scores)[::-1][:k]
    return float(np.sum(y_true[top_k_indices])) / k


def reciprocal_rank(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute Reciprocal Rank for a single query group."""
    if len(y_true) == 0 or np.sum(y_true) == 0:
        return 0.0
    ranked_indices = np.argsort(y_scores)[::-1]
    for rank, idx in enumerate(ranked_indices, start=1):
        if y_true[idx] == 1:
            return 1.0 / rank
    return 0.0


def dcg_at_k(gains: np.ndarray, k: int) -> float:
    """Compute DCG@K given gains in rank order."""
    if len(gains) == 0:
        return 0.0
    k = min(k, len(gains))
    gains = gains[:k]
    discounts = np.log2(np.arange(2, k + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """Compute nDCG@K for a single query group."""
    if len(y_true) == 0 or np.sum(y_true) == 0:
        return 0.0
    k = min(k, len(y_true))
    # Get gains in predicted rank order
    ranked_indices = np.argsort(y_scores)[::-1][:k]
    predicted_gains = y_true[ranked_indices]
    dcg = dcg_at_k(predicted_gains, k)
    # Ideal gains: sorted descending
    ideal_gains = np.sort(y_true)[::-1][:k]
    idcg = dcg_at_k(ideal_gains, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_ranking_metrics(
    df: pd.DataFrame,
    score_col: str = "y_prob",
    label_col: str = "label",
    group_col: str = "question_id",
    k_values: List[int] = None,
) -> Dict[str, float]:
    """Compute ranking metrics aggregated over all query groups."""
    if k_values is None:
        k_values = [1, 3, 5]

    p_at_k = {k: [] for k in k_values}
    mrr_scores = []
    ndcg_at_k_scores = {k: [] for k in k_values}

    for _, group_df in df.groupby(group_col):
        y_true = group_df[label_col].values
        y_scores = group_df[score_col].values

        mrr_scores.append(reciprocal_rank(y_true, y_scores))

        for k in k_values:
            p_at_k[k].append(precision_at_k(y_true, y_scores, k))
            ndcg_at_k_scores[k].append(ndcg_at_k(y_true, y_scores, k))

    metrics = {"mrr": float(np.mean(mrr_scores))}
    for k in k_values:
        metrics[f"precision@{k}"] = float(np.mean(p_at_k[k]))
        metrics[f"ndcg@{k}"] = float(np.mean(ndcg_at_k_scores[k]))

    return metrics


def compute_all_metrics(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    label_col: str = "label",
    group_col: str = "question_id",
) -> Dict[str, Any]:
    """Compute both classification and ranking metrics."""
    y_true = df[label_col].values

    classification = compute_classification_metrics(y_true, y_pred, y_prob)

    eval_df = df[[group_col, label_col]].copy()
    eval_df["y_prob"] = y_prob
    ranking = compute_ranking_metrics(eval_df, label_col=label_col, group_col=group_col)

    return {
        "classification": classification,
        "ranking": ranking,
    }


def print_metrics(metrics: Dict[str, Any], split_name: str = "Evaluation") -> None:
    """Pretty print evaluation metrics."""
    print(f"\n{'=' * 50}")
    print(f"{split_name} Results")
    print("=" * 50)

    print("\nClassification Metrics:")
    print("-" * 30)
    for name, value in metrics["classification"].items():
        print(f"  {name:12s}: {value:.4f}")

    print("\nRanking Metrics:")
    print("-" * 30)
    for name, value in metrics["ranking"].items():
        print(f"  {name:12s}: {value:.4f}")
