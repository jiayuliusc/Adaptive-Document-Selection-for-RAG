# Feature Documentation

## Overview

This document describes the features computed for each **(question, document)** pair in the dataset.

Each row represents:

- a question
- a candidate document (answer text)
- a label (`1 = relevant`, `0 = not relevant`)
- a set of features used to predict relevance

---

## Feature Groups

### 1. Semantic Features

#### `semantic_cosine_similarity`

- **Definition**: Cosine similarity between the embedding of the question and the document.

- **How it's computed**:
  1. Encode question and document using a sentence embedding model (e.g., Sentence-BERT)
  2. Compute cosine similarity between the two vectors

- **Intuition**: Measures meaning-level similarity even if wording differs.

---

### 2. Lexical Features

#### `token_overlap`

- **Definition**: Number of shared tokens between question and document
- **How**:
  - tokenize both texts
  - count intersection of tokens

#### `token_overlap_ratio`

- **Definition**: Overlap normalized by question length
- **Formula**:
  overlap_count / len(question_tokens)

#### `tfidf_similarity`

- **Definition**: Cosine similarity between TF-IDF vectors of question and document
- **How**:
  - fit TF-IDF vectorizer on corpus
  - transform question and document
  - compute cosine similarity

#### `bm25_score` (optional)

- **Definition**: BM25 relevance score

- **How**:
  - use standard BM25 implementation over corpus

- **Intuition**: Captures keyword matching strength

---

### 3. Document Features

#### `document_length`

- **Definition**: Number of tokens (or words) in the document

- **How**:
  - count tokens after preprocessing

- **Intuition**: Longer documents may dilute relevance

---

### 4. Query Features

#### `query_length`

- **Definition**: Number of tokens in the question

#### `question_type`

- **Definition**: Type of question based on first word

- **Categories**:
  - who, what, when, where, why, how, other

- **How**:
  - extract first token
  - map to predefined categories

#### `named_entity_count` (optional)

- **Definition**: Number of named entities in the question

- **How**:
  - use NER model (e.g., spaCy)

- **Intuition**: Some question types rely more on entities

---

### 5. Interaction Features

#### `similarity_times_doc_length`

- **Definition**: semantic similarity × document length
- **Formula**:
  semantic_cosine_similarity \* document_length

#### `normalized_overlap`

- **Definition**: token overlap normalized by document length

- **Formula**:
  token_overlap / document_length

- **Intuition**: Combines relevance signal with document size

---

## Label Definition

#### `label`

- `1`: document is the correct answer for the question
- `0`: document is randomly sampled from another question

---

## Final Feature Table Schema

Each row contains:

- `question`
- `document`
- `label`
- `semantic_cosine_similarity`
- `token_overlap`
- `token_overlap_ratio`
- `tfidf_similarity`
- `bm25_score` (optional)
- `document_length`
- `query_length`
- `question_type`
- `named_entity_count` (optional)
- `similarity_times_doc_length`
- `normalized_overlap`

---

## Notes

- All features are computed **per (question, document) pair**
- Features are **independent of the model**
- No feature uses label information (no leakage)
- Embeddings should be cached to avoid recomputation

---

## Goal for Modeling

These features are designed to help a model learn:

- which documents are relevant
- how relevance depends on both meaning and surface form
- how query and document characteristics interact

The model will use these features to assign a **relevance score** to each candidate document.
