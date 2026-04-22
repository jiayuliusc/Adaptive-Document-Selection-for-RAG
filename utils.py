from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


QUESTION_TYPES = {"who", "what", "when", "where", "why", "how"}
LOGGER = logging.getLogger(__name__)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", normalize_whitespace(text).lower())


def detect_question_type(question: str) -> str:
    tokens = simple_tokenize(question)
    if not tokens:
        return "other"
    first_token = tokens[0]
    return first_token if first_token in QUESTION_TYPES else "other"


def _extract_first_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return normalize_whitespace(value)
    if isinstance(value, (list, tuple)):
        for item in value:
            extracted = _extract_first_text(item)
            if extracted:
                return extracted
        return ""
    if isinstance(value, dict):
        preferred_keys = [
            "text",
            "answer",
            "value",
            "span",
            "short_answers",
            "short_answer",
            "long_answer",
            "answers",
        ]
        for key in preferred_keys:
            if key in value:
                extracted = _extract_first_text(value.get(key))
                if extracted:
                    return extracted
        for dict_value in value.values():
            extracted = _extract_first_text(dict_value)
            if extracted:
                return extracted
        return ""
    return normalize_whitespace(str(value))


def _pick_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    for name in candidates:
        if name in columns:
            return name
    return None


def _load_records_from_source(
    source: str,
    sample_size: int,
    seed: int,
    hf_dataset: str,
    hf_split: str,
    input_path: Optional[str],
) -> List[Dict[str, Any]]:
    if source == "hf":
        from datasets import load_dataset

        dataset = load_dataset(hf_dataset, split=hf_split)
        if sample_size > 0:
            subset_size = min(sample_size, len(dataset))
            dataset = dataset.shuffle(seed=seed).select(range(subset_size))
        return dataset.to_list()

    if not input_path:
        raise ValueError("input_path is required when source is 'csv' or 'jsonl'.")

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if source == "csv":
        dataframe = pd.read_csv(path)
    elif source == "jsonl":
        dataframe = pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unsupported source: {source}")

    if sample_size > 0 and sample_size < len(dataframe):
        dataframe = dataframe.sample(n=sample_size, random_state=seed)

    return dataframe.to_dict(orient="records")


def load_nq_subset(
    sample_size: int = 3000,
    seed: int = 42,
    source: str = "hf",
    hf_dataset: str = "nq_open",
    hf_split: str = "train",
    input_path: Optional[str] = None,
) -> pd.DataFrame:
    records = _load_records_from_source(
        source=source,
        sample_size=sample_size,
        seed=seed,
        hf_dataset=hf_dataset,
        hf_split=hf_split,
        input_path=input_path,
    )
    if not records:
        raise ValueError("No records were loaded from the selected source.")

    columns = list(records[0].keys())
    question_col = _pick_column(columns, ["question", "query", "question_text"])
    answer_col = _pick_column(
        columns,
        [
            "answer",
            "answers",
            "short_answer",
            "short_answers",
            "long_answer",
            "document",
        ],
    )
    id_col = _pick_column(columns, ["id", "example_id", "question_id"])

    if not question_col:
        raise ValueError("Could not find a question column in the loaded data.")

    rows: List[Dict[str, str]] = []
    for idx, row in enumerate(records):
        question = _extract_first_text(row.get(question_col))
        answer = _extract_first_text(row.get(answer_col)) if answer_col else ""

        if not answer:
            for fallback_key in [
                "answers",
                "annotations",
                "short_answers",
                "long_answer",
            ]:
                answer = _extract_first_text(row.get(fallback_key))
                if answer:
                    break

        if not question or not answer:
            continue

        source_example_id = row.get(id_col) if id_col else idx
        rows.append(
            {
                "source_example_id": normalize_whitespace(str(source_example_id)),
                "question": question,
                "answer": answer,
            }
        )

    qa_df = pd.DataFrame(rows)
    if qa_df.empty:
        raise ValueError(
            "No valid question-answer rows were extracted. Try a different source format."
        )

    qa_df = qa_df.drop_duplicates(subset=["question"]).reset_index(drop=True)
    qa_df.insert(0, "question_id", [f"q_{i}" for i in range(len(qa_df))])
    return qa_df[["question_id", "source_example_id", "question", "answer"]]


def build_positive_negative_pairs(
    qa_df: pd.DataFrame,
    negatives_per_question: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    required_columns = {"question_id", "question", "answer"}
    missing_columns = required_columns - set(qa_df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    if negatives_per_question < 1:
        raise ValueError("negatives_per_question must be >= 1")
    if len(qa_df) < 2:
        raise ValueError("Need at least 2 questions to create negative pairs.")

    rng = np.random.default_rng(seed)
    questions = qa_df["question"].astype(str).to_numpy()
    answers = qa_df["answer"].astype(str).to_numpy()
    question_ids = qa_df["question_id"].astype(str).to_numpy()

    pair_rows: List[Dict[str, Any]] = []
    num_questions = len(qa_df)

    for idx in range(num_questions):
        pair_rows.append(
            {
                "question_id": question_ids[idx],
                "question": questions[idx],
                "document": answers[idx],
                "label": 1,
                "pair_type": "positive",
                "source_doc_question_id": question_ids[idx],
            }
        )

        for _ in range(negatives_per_question):
            negative_idx = rng.integers(0, num_questions - 1)
            if negative_idx >= idx:
                negative_idx += 1

            pair_rows.append(
                {
                    "question_id": question_ids[idx],
                    "question": questions[idx],
                    "document": answers[negative_idx],
                    "label": 0,
                    "pair_type": "negative",
                    "source_doc_question_id": question_ids[negative_idx],
                }
            )

    pairs_df = pd.DataFrame(pair_rows)
    pairs_df = pairs_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    pairs_df.insert(0, "pair_id", [f"p_{i}" for i in range(len(pairs_df))])
    return pairs_df


def split_pairs_by_question(
    pairs_df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1.0")
    if "question_id" not in pairs_df.columns:
        raise ValueError("pairs_df must contain 'question_id'.")

    unique_question_ids = pairs_df["question_id"].drop_duplicates().to_numpy()
    if len(unique_question_ids) < 3:
        raise ValueError("Need at least 3 unique questions for train/val/test split.")

    train_questions, temp_questions = train_test_split(
        unique_question_ids,
        test_size=(1.0 - train_size),
        random_state=seed,
        shuffle=True,
    )

    val_fraction_within_temp = val_size / (val_size + test_size)
    val_questions, test_questions = train_test_split(
        temp_questions,
        train_size=val_fraction_within_temp,
        random_state=seed,
        shuffle=True,
    )

    train_df = pairs_df[pairs_df["question_id"].isin(train_questions)].reset_index(drop=True)
    val_df = pairs_df[pairs_df["question_id"].isin(val_questions)].reset_index(drop=True)
    test_df = pairs_df[pairs_df["question_id"].isin(test_questions)].reset_index(drop=True)

    return {
        "train": train_df,
        "validation": val_df,
        "test": test_df,
    }


def save_dataframe(df: pd.DataFrame, output_base_path: Path | str) -> None:
    output_base_path = Path(output_base_path)
    output_base_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_base_path.with_suffix(".csv")
    parquet_path = output_base_path.with_suffix(".parquet")

    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception as exc:
        LOGGER.warning(
            "Parquet export skipped for %s (%s). CSV was written successfully.",
            parquet_path,
            exc,
        )


def save_json(payload: Dict[str, Any], output_path: Path | str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")