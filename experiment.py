from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from features import FeatureConfig, compute_feature_table
from utils import (
    build_positive_negative_pairs,
    load_nq_subset,
    save_dataframe,
    save_json,
    split_pairs_by_question,
)


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a feature-ready relevance dataset from Natural Questions."
    )
    parser.add_argument("--source", choices=["hf", "csv", "jsonl"], default="hf")
    parser.add_argument("--input-path", type=str, default=None)
    parser.add_argument("--hf-dataset", type=str, default="nq_open")
    parser.add_argument("--hf-split", type=str, default="train")
    parser.add_argument("--sample-size", type=int, default=3000)
    parser.add_argument("--negatives-per-question", type=int, default=1)
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--disable-bm25", action="store_true")
    parser.add_argument("--enable-ner", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if args.source in {"csv", "jsonl"} and not args.input_path:
        raise ValueError("--input-path is required when --source is csv or jsonl")

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading NQ subset...")
    qa_df = load_nq_subset(
        sample_size=args.sample_size,
        seed=args.seed,
        source=args.source,
        hf_dataset=args.hf_dataset,
        hf_split=args.hf_split,
        input_path=args.input_path,
    )
    save_dataframe(qa_df, raw_dir / "nq_subset")
    LOGGER.info("Loaded %d QA rows.", len(qa_df))

    LOGGER.info("Creating positive and negative pairs...")
    pairs_df = build_positive_negative_pairs(
        qa_df,
        negatives_per_question=args.negatives_per_question,
        seed=args.seed,
    )
    save_dataframe(pairs_df, processed_dir / "pairs_all")
    LOGGER.info("Built %d total pairs.", len(pairs_df))

    LOGGER.info("Splitting pairs by question ID...")
    pair_splits = split_pairs_by_question(
        pairs_df,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    for split_name, split_df in pair_splits.items():
        save_dataframe(split_df, processed_dir / f"pairs_{split_name}")

    feature_config = FeatureConfig(
        embedding_model_name=args.embedding_model,
        use_bm25=not args.disable_bm25,
        use_named_entities=args.enable_ner,
    )

    # Use the full corpus as reference so split-level features are comparable.
    reference_texts = pd.concat([pairs_df["question"], pairs_df["document"]]).tolist()
    reference_documents = pairs_df["document"].tolist()

    LOGGER.info("Computing features for each split...")
    feature_splits = {}
    for split_name in ["train", "validation", "test"]:
        split_df = pair_splits[split_name]
        feature_df = compute_feature_table(
            split_df,
            config=feature_config,
            reference_texts=reference_texts,
            reference_documents=reference_documents,
        )
        feature_df.insert(0, "split", split_name)
        feature_splits[split_name] = feature_df
        save_dataframe(feature_df, processed_dir / f"features_{split_name}")

    features_all = pd.concat(
        [feature_splits["train"], feature_splits["validation"], feature_splits["test"]],
        ignore_index=True,
    )
    save_dataframe(features_all, processed_dir / "features_all")

    summary = {
        "source": args.source,
        "hf_dataset": args.hf_dataset if args.source == "hf" else None,
        "hf_split": args.hf_split if args.source == "hf" else None,
        "sample_size_requested": args.sample_size,
        "sample_size_loaded": int(len(qa_df)),
        "negatives_per_question": args.negatives_per_question,
        "total_pairs": int(len(pairs_df)),
        "splits": {
            split_name: {
                "rows": int(len(df)),
                "unique_questions": int(df["question_id"].nunique()),
                "positive_rate": float(df["label"].mean()),
            }
            for split_name, df in feature_splits.items()
        },
        "feature_columns": [
            col
            for col in features_all.columns
            if col
            not in {
                "split",
                "pair_id",
                "question_id",
                "question",
                "document",
                "label",
                "pair_type",
                "source_doc_question_id",
            }
        ],
    }
    save_json(summary, processed_dir / "dataset_summary.json")

    LOGGER.info("Saved processed datasets and feature tables to %s", output_dir)
    LOGGER.info("Done.")


if __name__ == "__main__":
    main()