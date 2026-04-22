# dgqw1

## What Is Already Done

This phase is complete for dataset construction and feature engineering.

Implemented components:

- NQ subset loading from Hugging Face or local CSV/JSONL
- Positive and negative pair creation with leakage-safe negative sampling
- Question-level train/validation/test split (no question overlap across splits)
- Feature computation for five groups:
  - semantic
  - lexical
  - document
  - query
  - interaction
- Unified output export to CSV and Parquet (CSV always succeeds even if Parquet dependencies are missing)
- Dataset summary export for quick inspection

Main files:

- experiment.py: full pipeline entrypoint
- utils.py: loading, pairing, splitting, saving helpers
- features.py: all feature computation logic
- Features.md: feature definitions
- requirements.txt: dependencies

## How To Run The Project

1. Install dependencies

   pip install -r requirements.txt

2. Run end-to-end pipeline (default NQ source)

   python experiment.py --sample-size 3000 --negatives-per-question 1

Optional local input run:

python experiment.py --source csv --input-path path/to/your_file.csv --sample-size 3000

Optional quick launcher:

bash scripts/run_pipeline.sh 3000 1

## Expected Outputs

Generated under data/raw and data/processed:

- nq_subset
- pairs_all, pairs_train, pairs_validation, pairs_test
- features_train, features_validation, features_test, features_all
- dataset_summary.json

Core training tables for next step:

- data/processed/features_train.csv
- data/processed/features_validation.csv
- data/processed/features_test.csv

## How To Use This For Model Training (Next Step)

1. Use features_train.csv to fit a relevance model for label prediction.
2. Use features_validation.csv for model selection and threshold tuning.
3. Use features_test.csv only once for final reporting.

Recommended initial setup:

- Start with a simple baseline: logistic regression, random forest, or gradient boosting
- Use numeric feature columns only (exclude raw text columns unless adding text models)
- Keep question_id for grouped evaluation

Suggested baseline feature set:

- semantic_cosine_similarity
- token_overlap
- token_overlap_ratio
- tfidf_similarity
- bm25_score
- document_length
- query_length
- named_entity_count
- similarity_times_doc_length
- normalized_overlap

Handle question_type as a categorical feature (one-hot encoding).

Recommended metrics:

- classification: AUROC, F1, precision, recall
- ranking per question_id: Precision@1, MRR, nDCG@k

## Suggested Immediate Follow-Up Tasks

- Add model.py for baseline training and evaluation
- Add experiment tracking for hyperparameters and metrics
- Add embedding caching to speed repeated feature runs
- Add unit tests for pair generation and split leakage checks
