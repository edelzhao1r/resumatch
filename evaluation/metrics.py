"""Evaluation metrics for tailored resumes vs job descriptions."""

from __future__ import annotations

import re
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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


def semantic_similarity_tfidf(text1: str, text2: str) -> float:
    """
    Compute semantic similarity between two texts using TF-IDF
    and cosine similarity. Replaces Word2Vec for cloud compatibility.
    """
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    score = cosine_similarity(tfidf[0], tfidf[1])[0][0]
    return round(float(score), 4)
