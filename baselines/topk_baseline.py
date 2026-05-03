"""
Top-K retrieval baseline for document relevance ranking.

Simulates a regular RAG + top-K retrieval pipeline:
  - For each question, rank all candidate documents by a single similarity score
  - Keep only the top-K documents (discard the rest, as a real retriever would)
  - Evaluate the selected top-K set using P@1, MRR, and nDCG@K

The primary baseline uses semantic cosine similarity, which is the standard
retrieval signal in RAG systems.  BM25 and TF-IDF are included for reference.

All methods use precomputed feature columns from features_test.csv — no
re-embedding or re-scoring at inference time.  Metric computation is delegated
to the shared models/evaluate.py module so numbers are directly comparable to
the XGBoost reranker.

Usage
-----
    # From the project root:
    python baselines/topk_baseline.py [--top-k 3]

    # From inside baselines/:
    python topk_baseline.py [--top-k 3]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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


def select_top_k(df: pd.DataFrame, score_col: str, k: int) -> pd.DataFrame:
    """
    For each question group, keep only the top-K documents by score.

    This simulates a real retriever that returns exactly K candidates from a
    large corpus — documents outside the top-K are discarded before evaluation.
    """
    return (
        df.sort_values([GROUP_COL, score_col], ascending=[True, False])
        .groupby(GROUP_COL, group_keys=False)
        .head(k)
        .reset_index(drop=True)
    )


def run_baseline(
    df: pd.DataFrame,
    score_col: str,
    k: int,
    fill_na: float | None = None,
) -> dict:
    """
    Rank candidates by score_col, select top-K per question, compute metrics.
    """
    if score_col not in df.columns:
        raise ValueError(f"Column '{score_col}' not found in test data.")

    eval_df = df[[GROUP_COL, LABEL_COL, score_col]].copy()

    if fill_na is not None:
        eval_df[score_col] = eval_df[score_col].fillna(fill_na)

    # Select top-K per question — this is the retrieval step.
    eval_df = select_top_k(eval_df, score_col=score_col, k=k)
    eval_df = eval_df.rename(columns={score_col: "score"})

    return compute_ranking_metrics(
        eval_df,
        score_col="score",
        label_col=LABEL_COL,
        group_col=GROUP_COL,
    )


def print_baseline_results(name: str, k: int, metrics: dict) -> None:
    print(f"\n{'=' * 50}")
    print(f"Baseline: {name}  (top-K = {k})")
    print("=" * 50)
    print(f"  {'MRR':<14}: {metrics['mrr']:.4f}")
    for kv in [1, 3, 5]:
        pk_key = f"precision@{kv}"
        nk_key = f"ndcg@{kv}"
        if pk_key in metrics:
            print(f"  {pk_key:<14}: {metrics[pk_key]:.4f}")
        if nk_key in metrics:
            print(f"  {nk_key:<14}: {metrics[nk_key]:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run top-K retrieval baselines")
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top candidates to select per question (default: 3)",
    )
    args = parser.parse_args()
    k = args.top_k

    print("Loading test data...")
    test_df = load_test_split()
    print(f"  Rows: {len(test_df):,}  |  Unique questions: {test_df[GROUP_COL].nunique():,}")
    print(f"  Positive rate: {test_df[LABEL_COL].mean():.3f}")
    print(f"  Top-K: {k}")

    semantic_metrics = run_baseline(test_df, score_col="semantic_cosine_similarity", k=k)
    bm25_metrics     = run_baseline(test_df, score_col="bm25_score",                 k=k, fill_na=0.0)
    tfidf_metrics    = run_baseline(test_df, score_col="tfidf_similarity",            k=k)

    print_baseline_results("Semantic Cosine Similarity", k, semantic_metrics)
    print_baseline_results("BM25",                       k, bm25_metrics)
    print_baseline_results("TF-IDF Cosine Similarity",   k, tfidf_metrics)

    results = {
        "top_k": k,
        "semantic": semantic_metrics,
        "bm25":     bm25_metrics,
        "tfidf":    tfidf_metrics,
    }

    output_path = OUTPUT_DIR / "baseline_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
