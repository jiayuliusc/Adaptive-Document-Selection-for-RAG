# Beyond Top-K: Feature-Ready Dataset for RAG Retrieval Learning

This repository contains a lightweight prototype pipeline for building a supervised
relevance dataset from Natural Questions (NQ) for retrieval learning.

The current phase focuses on data and features, not model training.

## What This Pipeline Produces

For each (question, candidate document) pair, the pipeline creates:

- label (1 = relevant, 0 = irrelevant)
- semantic features
- lexical features
- document features
- query features
- interaction features

It saves train/validation/test splits with no question leakage across splits.

## Repository Layout

```
.
|-- data/
|   |-- raw/
|   `-- processed/
|-- notebooks/
|-- scripts/
|-- experiment.py
|-- features.py
|-- utils.py
|-- Features.md
|-- README.md
`-- requirements.txt
```

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the full pipeline:

```bash
python experiment.py --sample-size 3000 --negatives-per-question 1
```

This command:

1. Loads an NQ subset (default: Hugging Face `nq_open` train split)
2. Builds positive and negative pairs
3. Splits pairs by question into train/validation/test
4. Computes all feature groups
5. Saves all generated tables

## Key CLI Options

```bash
python experiment.py \
	--sample-size 5000 \
	--negatives-per-question 2 \
	--source hf \
	--hf-dataset nq_open \
	--hf-split train \
	--output-dir data
```

- `--source`: `hf`, `csv`, or `jsonl`
- `--input-path`: required for `csv` or `jsonl` input
- `--disable-bm25`: skip BM25 computation
- `--enable-ner`: enable named entity counting with spaCy
- `--embedding-model`: sentence-transformers model name for semantic similarity

## Output Files

Generated in `data/`:

- `data/raw/nq_subset.csv` and `.parquet`
- `data/processed/pairs_all.csv` and `.parquet`
- `data/processed/pairs_train.csv` and `.parquet`
- `data/processed/pairs_validation.csv` and `.parquet`
- `data/processed/pairs_test.csv` and `.parquet`
- `data/processed/features_train.csv` and `.parquet`
- `data/processed/features_validation.csv` and `.parquet`
- `data/processed/features_test.csv` and `.parquet`
- `data/processed/features_all.csv` and `.parquet`
- `data/processed/dataset_summary.json`

## Notes

- The pipeline ensures negative pairs never use the same question's answer.
- Splits are done at question level to prevent leakage.
- `Features.md` documents each feature and intuition.
