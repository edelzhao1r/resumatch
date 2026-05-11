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


# ============================================================
# Fabrication / Faithfulness
# ============================================================
# Distinct stopword list + extraction logic that matches run_eval.py's
# keyword_coverage() so this metric stays consistent with what's reported
# in the rest of the eval table.
_EVAL_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "with", "on",
    "at", "by", "from", "is", "are", "be", "as", "we", "you", "that", "this",
    "it", "our", "your", "will", "have", "has", "their", "they", "who",
    "what", "all", "each", "can", "may", "more", "about", "which", "into",
    "its", "across", "using", "build", "built", "not", "strong", "new",
}


def _extract_top_jd_keywords(jd_text: str, top_n: int = 20) -> list[str]:
    """Top-N JD keywords using the same logic as run_eval.py's
    keyword_coverage(): lowercase words length ≥ 3, minus a small stoplist,
    then CountVectorizer.max_features."""
    from sklearn.feature_extraction.text import CountVectorizer

    words = re.findall(r"\b[a-z]{3,}\b", (jd_text or "").lower())
    words = [w for w in words if w not in _EVAL_STOPWORDS]
    if not words:
        return []
    vec = CountVectorizer(max_features=top_n)
    vec.fit([" ".join(words)])
    return list(vec.get_feature_names_out())


def fabrication_rate(
    original_text: str,
    tailored_text: str,
    jd_text: str,
    top_n: int = 20,
) -> dict:
    """Faithfulness metric: of the JD keywords the tailoring chose to include,
    what fraction lack any grounding in the candidate's original resume?

    Returns a dict with:
        new_keywords_count    — top-N JD keywords appearing in tailored
                                 (i.e., the keywords the tailoring added/kept)
        fabricated_count      — of those, how many do NOT appear in original
        fabrication_rate      — fabricated_count / new_keywords_count
                                 (0 if no JD keywords made it into tailored)
        fabricated_keywords   — the actual fabricated keyword list
    """
    keywords = _extract_top_jd_keywords(jd_text, top_n=top_n)
    if not keywords:
        return {
            "new_keywords_count": 0,
            "fabricated_count": 0,
            "fabrication_rate": 0.0,
            "fabricated_keywords": [],
        }

    original_lower = (original_text or "").lower()
    tailored_lower = (tailored_text or "").lower()

    # Top-N JD keywords that the tailored version surfaced (denominator)
    added = [kw for kw in keywords if kw in tailored_lower]
    # Of those, which are NOT grounded in the candidate's original resume?
    fabricated = [kw for kw in added if kw not in original_lower]

    rate = len(fabricated) / len(added) if added else 0.0

    return {
        "new_keywords_count": len(added),
        "fabricated_count": len(fabricated),
        "fabrication_rate": round(rate, 4),
        "fabricated_keywords": fabricated,
    }


def semantic_similarity_tfidf(text1: str, text2: str) -> float:
    """
    Compute semantic similarity between two texts using TF-IDF
    and cosine similarity. Replaces Word2Vec for cloud compatibility.
    """
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    score = cosine_similarity(tfidf[0], tfidf[1])[0][0]
    return round(float(score), 4)
