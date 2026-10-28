"""Evaluation metrics for RAG retrieval and generation (The RAG Triad + Correctness)."""

import re
from typing import Dict, List, Optional, Set

import numpy as np


def get_words(text: str) -> Set[str]:
    """Extracts non-stop lowercase words."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with",
        "of", "is", "are", "was", "were", "it", "this", "that", "by", "from",
        "what", "who", "where", "how", "does", "did", "their", "they",
    }
    tokens = re.findall(r"\b[a-zA-Z0-9_\-]+\b", text.lower())
    return {w for w in tokens if w not in stopwords and len(w) > 1}


# ============================================================================
# STAGE 1: RETRIEVAL METRICS
# ============================================================================
def precision_at_k(retrieved: List[Dict], expected_sources: List[str], k: int = 3) -> float:
    """Precision@k: (Relevant chunks in top-k) / k."""
    if not retrieved or k <= 0:
        return 0.0
    top = retrieved[:k]
    matches = sum(
        1 for c in top if any(str(src).lower() in str(c.get("source", "")).lower() for src in expected_sources)
    )
    return round(matches / min(k, len(top)), 4)


def recall_at_k(retrieved: List[Dict], expected_sources: List[str], k: int = 3) -> float:
    """Recall@k: (Relevant sources found in top-k) / (Total expected sources)."""
    if not retrieved or not expected_sources:
        return 0.0
    top = retrieved[:k]
    found = {
        src for src in expected_sources
        if any(str(src).lower() in str(c.get("source", "")).lower() for c in top)
    }
    return round(len(found) / len(expected_sources), 4)


def mean_reciprocal_rank(retrieved: List[Dict], expected_sources: List[str]) -> float:
    """MRR: 1 / rank of first relevant chunk."""
    for rank, c in enumerate(retrieved, start=1):
        if any(str(src).lower() in str(c.get("source", "")).lower() for src in expected_sources):
            return round(1.0 / rank, 4)
    return 0.0


# ============================================================================
# STAGE 2: GENERATION METRICS (THE RAG TRIAD + CORRECTNESS)
# ============================================================================
def context_relevance(query: str, context: str) -> float:
    """Triad Pillar 1: Context Relevance (Question -> Context).

    Measures whether the retrieved context contains relevant information for the question.
    """
    q_words = get_words(query)
    c_words = get_words(context)
    if not q_words:
        return 0.0
    return round(len(q_words.intersection(c_words)) / len(q_words), 4)


def faithfulness(answer: str, context: str) -> float:
    """Triad Pillar 2: Groundedness / Faithfulness (Context -> Answer).

    Measures whether statements in the answer are supported by context (Hallucination check).
    """
    a_words = get_words(answer)
    c_words = get_words(context)
    if not a_words:
        return 0.0
    return round(len(a_words.intersection(c_words)) / len(a_words), 4)


def answer_relevance(query: str, answer: str) -> float:
    """Triad Pillar 3: Answer Relevance (Question -> Answer).

    Measures whether the generated answer directly addresses the question asked.
    """
    q_words = get_words(query)
    a_words = get_words(answer)
    if not q_words or not a_words:
        return 0.0
    return round(min(1.0, len(q_words.intersection(a_words)) / len(q_words)), 4)


def factual_correctness(
    generated_answer: str,
    ground_truth_answer: str,
    embedder: Optional[object] = None,
) -> float:
    """Evaluates factual and semantic correctness of generated answer against reference ground truth.

    Combines token-level F1 overlap and embedding cosine similarity.
    """
    gen_words = get_words(generated_answer)
    truth_words = get_words(ground_truth_answer)

    if not gen_words or not truth_words:
        return 0.0

    common = gen_words.intersection(truth_words)
    if not common:
        token_f1 = 0.0
    else:
        precision = len(common) / len(gen_words)
        recall = len(common) / len(truth_words)
        token_f1 = 2 * (precision * recall) / (precision + recall)

    if embedder is not None:
        try:
            vec1 = embedder.encode([generated_answer], convert_to_numpy=True)[0]
            vec2 = embedder.encode([ground_truth_answer], convert_to_numpy=True)[0]
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 > 0 and norm2 > 0:
                sim = float(np.dot(vec1, vec2) / (norm1 * norm2))
                sim = max(0.0, min(1.0, sim))
                return round(0.4 * token_f1 + 0.6 * sim, 4)
        except Exception:
            pass

    return round(token_f1, 4)
