"""
Top-K retrieval baselines for document relevance ranking.

Two non-learned baselines are implemented:
  - Semantic:  rank documents by precomputed semantic cosine similarity
  - BM25:      rank documents by precomputed BM25 score

Both use the same processed test features as the XGBoost model so results
are directly comparable.  Metric computation is delegated to the shared
models/evaluate.py module.

Usage
-----
    # From the project root:
    python baselines/topk_baseline.py

    # From inside baselines/:
    python topk_baseline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow imports from models/ regardless of working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "models"))

from evaluate import compute_ranking_metrics  # noqa: E402 — after sys.path update


DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "baselines"

LABEL_COL = "label"
GROUP_COL = "question_id"


def load_test_split() -> pd.DataFrame:
    """Load the held-out test feature CSV."""
    path = DATA_DIR / "features_test.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Test feature file not found: {path}\n"
            "Run experiment.py first to generate features."
        )
    return pd.read_csv(path)


def run_semantic_baseline(df: pd.DataFrame) -> dict:
    """Rank by semantic cosine similarity only."""
    score_col = "semantic_cosine_similarity"
    if score_col not in df.columns:
        raise ValueError(f"Column '{score_col}' not found in test data.")

    eval_df = df[[GROUP_COL, LABEL_COL, score_col]].copy()
    eval_df = eval_df.rename(columns={score_col: "score"})

    return compute_ranking_metrics(
        eval_df,
        score_col="score",
        label_col=LABEL_COL,
        group_col=GROUP_COL,
    )


def run_bm25_baseline(df: pd.DataFrame) -> dict:
    """Rank by BM25 score only (NaN filled with 0)."""
    score_col = "bm25_score"
    if score_col not in df.columns:
        raise ValueError(f"Column '{score_col}' not found in test data.")

    eval_df = df[[GROUP_COL, LABEL_COL, score_col]].copy()
    eval_df[score_col] = eval_df[score_col].fillna(0)
    eval_df = eval_df.rename(columns={score_col: "score"})

    return compute_ranking_metrics(
        eval_df,
        score_col="score",
        label_col=LABEL_COL,
        group_col=GROUP_COL,
    )


def run_tfidf_baseline(df: pd.DataFrame) -> dict:
    """Rank by TF-IDF cosine similarity only."""
    score_col = "tfidf_similarity"
    if score_col not in df.columns:
        raise ValueError(f"Column '{score_col}' not found in test data.")

    eval_df = df[[GROUP_COL, LABEL_COL, score_col]].copy()
    eval_df = eval_df.rename(columns={score_col: "score"})

    return compute_ranking_metrics(
        eval_df,
        score_col="score",
        label_col=LABEL_COL,
        group_col=GROUP_COL,
    )


def print_baseline_results(name: str, metrics: dict) -> None:
    print(f"\n{'=' * 50}")
    print(f"Baseline: {name}")
    print("=" * 50)
    print(f"  {'MRR':<14}: {metrics['mrr']:.4f}")
    for k in [1, 3, 5]:
        pk_key = f"precision@{k}"
        nk_key = f"ndcg@{k}"
        if pk_key in metrics:
            print(f"  {pk_key:<14}: {metrics[pk_key]:.4f}")
        if nk_key in metrics:
            print(f"  {nk_key:<14}: {metrics[nk_key]:.4f}")


def main() -> None:
    print("Loading test data...")
    test_df = load_test_split()
    print(f"  Rows: {len(test_df):,}  |  Unique questions: {test_df[GROUP_COL].nunique():,}")
    print(f"  Positive rate: {test_df[LABEL_COL].mean():.3f}")

    semantic_metrics = run_semantic_baseline(test_df)
    bm25_metrics = run_bm25_baseline(test_df)
    tfidf_metrics = run_tfidf_baseline(test_df)

    print_baseline_results("Semantic Cosine Similarity", semantic_metrics)
    print_baseline_results("BM25", bm25_metrics)
    print_baseline_results("TF-IDF Cosine Similarity", tfidf_metrics)

    results = {
        "semantic": semantic_metrics,
        "bm25": bm25_metrics,
        "tfidf": tfidf_metrics,
    }

    output_path = OUTPUT_DIR / "baseline_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
