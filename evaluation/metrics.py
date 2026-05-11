"""Evaluation metrics for tailored resumes vs job descriptions."""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from typing import Iterable

import numpy as np

# A small built-in stopword list so we don't require nltk downloads.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "could", "did", "do", "does", "for", "from", "had", "has", "have", "he",
    "her", "his", "i", "if", "in", "into", "is", "it", "its", "of", "on",
    "or", "our", "she", "so", "such", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "those", "to", "was", "we",
    "were", "what", "when", "which", "who", "will", "with", "you", "your",
    "would", "should", "may", "must", "also", "about", "than", "via", "per",
    "etc", "any", "all", "not", "no", "yes", "ours", "us",
}

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z+/#.\-]*")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _content_tokens(text: str) -> list[str]:
    return [t for t in _tokenize(text) if len(t) > 2 and t not in _STOPWORDS]


def keyword_coverage(jd_text: str, resume_text: str, top_n: int = 20) -> float:
    """Percentage of the JD's top-N content keywords that appear in the resume.

    Args:
        jd_text: Job description text.
        resume_text: Resume text.
        top_n: How many of the JD's most-frequent content tokens to consider.

    Returns:
        Coverage as a percentage (0.0 – 100.0). 0.0 if the JD has no content tokens.
    """
    jd_tokens = _content_tokens(jd_text)
    if not jd_tokens:
        return 0.0

    most_common = [tok for tok, _ in Counter(jd_tokens).most_common(top_n)]
    resume_token_set = set(_content_tokens(resume_text))

    hits = sum(1 for kw in most_common if kw in resume_token_set)
    return 100.0 * hits / len(most_common)


@lru_cache(maxsize=1)
def _load_word2vec():
    """Lazy-load a pretrained Word2Vec / GloVe model via gensim downloader."""
    import gensim.downloader as api

    return api.load("glove-wiki-gigaword-100")


def _mean_vector(tokens: Iterable[str], kv) -> np.ndarray | None:
    vectors = [kv[t] for t in tokens if t in kv.key_to_index]
    if not vectors:
        return None
    return np.mean(np.stack(vectors), axis=0)


def semantic_similarity_word2vec(text1: str, text2: str) -> float:
    """Cosine similarity between mean-pooled Word2Vec embeddings of two texts.

    Args:
        text1, text2: Arbitrary strings (e.g. JD vs. tailored resume).

    Returns:
        Cosine similarity in [-1.0, 1.0]. Returns 0.0 if either text has no
        in-vocabulary tokens.
    """
    kv = _load_word2vec()
    tokens1 = _content_tokens(text1)
    tokens2 = _content_tokens(text2)

    v1 = _mean_vector(tokens1, kv)
    v2 = _mean_vector(tokens2, kv)
    if v1 is None or v2 is None:
        return 0.0

    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / denom)
