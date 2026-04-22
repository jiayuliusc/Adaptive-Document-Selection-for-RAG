from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import detect_question_type, normalize_whitespace, simple_tokenize


LOGGER = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    use_bm25: bool = True
    use_named_entities: bool = False
    batch_size: int = 64


def _compute_semantic_similarity(
    questions: Sequence[str],
    documents: Sequence[str],
    config: FeatureConfig,
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(config.embedding_model_name)
        question_embeddings = model.encode(
            list(questions),
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=config.batch_size,
        )
        document_embeddings = model.encode(
            list(documents),
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=config.batch_size,
        )
        return np.einsum("ij,ij->i", question_embeddings, document_embeddings)
    except Exception as exc:
        LOGGER.warning(
            "Semantic embedding failed (%s). Falling back to zeros for similarity.", exc
        )
        return np.zeros(len(questions), dtype=float)


def _compute_tfidf_similarity(
    questions: Sequence[str],
    documents: Sequence[str],
    reference_texts: Optional[Sequence[str]] = None,
) -> np.ndarray:
    fit_corpus = list(reference_texts) if reference_texts is not None else None
    if not fit_corpus:
        fit_corpus = list(questions) + list(documents)

    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
    vectorizer.fit(fit_corpus)
    question_matrix = vectorizer.transform(questions)
    document_matrix = vectorizer.transform(documents)

    # With default L2 normalization in TF-IDF vectors, row-wise dot product equals cosine.
    return question_matrix.multiply(document_matrix).sum(axis=1).A1


def _compute_bm25_scores(
    questions: Sequence[str],
    documents: Sequence[str],
    use_bm25: bool,
    reference_documents: Optional[Sequence[str]] = None,
) -> np.ndarray:
    if not use_bm25:
        return np.full(len(questions), np.nan, dtype=float)

    try:
        from rank_bm25 import BM25Okapi
    except Exception as exc:
        LOGGER.warning("BM25 package unavailable (%s). Setting BM25 scores to NaN.", exc)
        return np.full(len(questions), np.nan, dtype=float)

    corpus_documents = list(reference_documents) if reference_documents is not None else None
    if not corpus_documents:
        corpus_documents = list(documents)

    unique_documents = list(dict.fromkeys(corpus_documents))
    tokenized_corpus = [simple_tokenize(doc) for doc in unique_documents]
    if not tokenized_corpus:
        return np.full(len(questions), np.nan, dtype=float)

    bm25 = BM25Okapi(tokenized_corpus)
    document_index = {doc: idx for idx, doc in enumerate(unique_documents)}

    query_cache = {}
    scores = np.empty(len(questions), dtype=float)
    for idx, (question, document) in enumerate(zip(questions, documents)):
        if question not in query_cache:
            query_tokens = simple_tokenize(question)
            query_cache[question] = bm25.get_scores(query_tokens)

        matched_doc_idx = document_index.get(document)
        if matched_doc_idx is None:
            scores[idx] = np.nan
        else:
            scores[idx] = float(query_cache[question][matched_doc_idx])

    return scores


def _fallback_named_entity_count(question: str) -> int:
    # Fallback heuristic: count title-cased words as rough entity candidates.
    return len(re.findall(r"\b[A-Z][a-z]+\b", question))


def _compute_named_entity_counts(
    questions: Sequence[str],
    use_named_entities: bool,
    batch_size: int,
) -> np.ndarray:
    if not use_named_entities:
        return np.zeros(len(questions), dtype=int)

    try:
        import spacy

        nlp = spacy.load("en_core_web_sm")
        counts = [len(doc.ents) for doc in nlp.pipe(questions, batch_size=batch_size)]
        return np.array(counts, dtype=int)
    except Exception as exc:
        LOGGER.warning(
            "spaCy NER unavailable (%s). Falling back to a regex-based entity count.",
            exc,
        )
        return np.array([_fallback_named_entity_count(question) for question in questions])


def compute_feature_table(
    pairs_df: pd.DataFrame,
    config: Optional[FeatureConfig] = None,
    reference_texts: Optional[Sequence[str]] = None,
    reference_documents: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    config = config or FeatureConfig()
    required_columns = {"question_id", "question", "document", "label"}
    missing_columns = required_columns - set(pairs_df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns for feature computation: {missing_columns}")

    feature_df = pairs_df.copy()
    feature_df["question"] = feature_df["question"].astype(str).map(normalize_whitespace)
    feature_df["document"] = feature_df["document"].astype(str).map(normalize_whitespace)

    questions = feature_df["question"].tolist()
    documents = feature_df["document"].tolist()

    question_tokens = [simple_tokenize(text) for text in questions]
    document_tokens = [simple_tokenize(text) for text in documents]
    question_sets = [set(tokens) for tokens in question_tokens]
    document_sets = [set(tokens) for tokens in document_tokens]

    token_overlap = np.array(
        [len(qset.intersection(dset)) for qset, dset in zip(question_sets, document_sets)],
        dtype=float,
    )
    query_lengths = np.array([len(tokens) for tokens in question_tokens], dtype=float)
    document_lengths = np.array([len(tokens) for tokens in document_tokens], dtype=float)

    feature_df["semantic_cosine_similarity"] = _compute_semantic_similarity(
        questions,
        documents,
        config,
    )
    feature_df["token_overlap"] = token_overlap.astype(int)
    feature_df["token_overlap_ratio"] = token_overlap / np.maximum(query_lengths, 1.0)
    feature_df["tfidf_similarity"] = _compute_tfidf_similarity(
        questions,
        documents,
        reference_texts=reference_texts,
    )
    feature_df["bm25_score"] = _compute_bm25_scores(
        questions,
        documents,
        use_bm25=config.use_bm25,
        reference_documents=reference_documents,
    )

    feature_df["document_length"] = document_lengths.astype(int)
    feature_df["query_length"] = query_lengths.astype(int)
    feature_df["question_type"] = [detect_question_type(question) for question in questions]
    feature_df["named_entity_count"] = _compute_named_entity_counts(
        questions,
        use_named_entities=config.use_named_entities,
        batch_size=config.batch_size,
    ).astype(int)

    feature_df["similarity_times_doc_length"] = (
        feature_df["semantic_cosine_similarity"] * feature_df["document_length"]
    )
    feature_df["normalized_overlap"] = feature_df["token_overlap"] / np.maximum(
        feature_df["document_length"],
        1,
    )

    preferred_order = [
        "pair_id",
        "question_id",
        "question",
        "document",
        "label",
        "pair_type",
        "source_doc_question_id",
        "semantic_cosine_similarity",
        "token_overlap",
        "token_overlap_ratio",
        "tfidf_similarity",
        "bm25_score",
        "document_length",
        "query_length",
        "question_type",
        "named_entity_count",
        "similarity_times_doc_length",
        "normalized_overlap",
    ]
    existing_columns = [col for col in preferred_order if col in feature_df.columns]
    remaining_columns = [col for col in feature_df.columns if col not in existing_columns]
    return feature_df[existing_columns + remaining_columns]