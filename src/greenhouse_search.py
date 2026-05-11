"""Greenhouse job-board client.

Greenhouse exposes a public, key-less JSON API for any company that hosts
their careers page on boards.greenhouse.io. We use it to pull live openings
straight from a target employer.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import requests

GREENHOUSE_BASE_URL = "https://boards-api.greenhouse.io/v1/boards/{slug}/jobs"

# Curated map of well-known tech companies → their Greenhouse board slugs.
# Slugs are the path segment used at boards.greenhouse.io/<slug>.
COMPANY_SLUGS: dict[str, str] = {
    "anthropic": "anthropic",
    "airbnb": "airbnb",
    "stripe": "stripe",
    "figma": "figma",
    "notion": "notion",
    "databricks": "databricks",
    "openai": "openai",
    "coinbase": "coinbase",
    "cloudflare": "cloudflare",
    "discord": "discord",
    "doordash": "doordash",
    "instacart": "instacart",
    "robinhood": "robinhood",
    "lyft": "lyft",
    "pinterest": "pinterest",
    "reddit": "reddit",
    "asana": "asana",
    "dropbox": "dropbox",
    "twitch": "twitch",
    "snowflake": "snowflakecomputing",
    "elastic": "elastic",
    "mongodb": "mongodb",
    "hashicorp": "hashicorp",
    "gitlab": "gitlab",
    "datadog": "datadog",
    "plaid": "plaid",
    "brex": "brex",
    "ramp": "ramp",
    "scale ai": "scaleai",
    "scale": "scaleai",
    "samsara": "samsara",
    "rippling": "rippling",
    "gusto": "gusto",
    "zapier": "zapier",
    "webflow": "webflow",
    "vercel": "vercel",
    "retool": "retool",
    "linear": "linear",
    "perplexity": "perplexity",
    "mistral": "mistralai",
    "mistral ai": "mistralai",
    "cohere": "cohere",
    "huggingface": "huggingface",
    "hugging face": "huggingface",
}


def _normalize(name: str) -> str:
    """Lowercase + collapse whitespace for fuzzy matching."""
    return re.sub(r"\s+", " ", name.strip().lower())


def find_company_slug(company_name: str) -> Optional[str]:
    """Resolve a free-text company name to a Greenhouse slug.

    Matching is case-insensitive and tolerant of extra whitespace. Falls back
    to a substring match against the known names.
    """
    if not company_name:
        return None

    norm = _normalize(company_name)

    # Exact match
    if norm in COMPANY_SLUGS:
        return COMPANY_SLUGS[norm]

    # Strip common suffixes ("inc", "labs", "ai") and retry
    stripped = re.sub(r"\b(inc|llc|labs|ai)\b", "", norm).strip()
    if stripped and stripped in COMPANY_SLUGS:
        return COMPANY_SLUGS[stripped]

    # Substring match
    for known_name, slug in COMPANY_SLUGS.items():
        if norm in known_name or known_name in norm:
            return slug

    return None


def _strip_html(html: str) -> str:
    """Very lightweight HTML→text. Greenhouse content is HTML, sometimes
    with entities pre-encoded — so we decode entities BEFORE stripping tags."""
    import html as html_mod

    text = html_mod.unescape(html or "")
    # Now `text` should contain real `<` `>` characters.
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"</p>", "\n\n", text)
    text = re.sub(r"</li>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def search_jobs_greenhouse(
    company_slug: str,
    keyword_filter: str = "",
) -> list[dict[str, Any]]:
    """Fetch live job listings from a single Greenhouse-hosted careers board.

    Args:
        company_slug: The board slug (e.g. "anthropic", "stripe").
        keyword_filter: If provided, only jobs whose title OR description
            contain this keyword (case-insensitive) are returned.

    Returns:
        A list of dicts with keys: title, company, location, description, url.
        Returns an empty list (not an exception) if the company is not found
        or the API is unreachable.
    """
    if not company_slug:
        return []

    url = GREENHOUSE_BASE_URL.format(slug=company_slug)
    try:
        resp = requests.get(url, params={"content": "true"}, timeout=20)
    except requests.RequestException:
        return []

    if resp.status_code == 404:
        return []
    if not resp.ok:
        return []

    try:
        payload = resp.json()
    except ValueError:
        return []

    raw_jobs = payload.get("jobs", []) or []
    needle = keyword_filter.strip().lower()

    results: list[dict[str, Any]] = []
    for job in raw_jobs:
        title = (job.get("title") or "").strip()
        content_html = job.get("content") or ""
        description = _strip_html(content_html)

        if needle:
            haystack = f"{title}\n{description}".lower()
            if needle not in haystack:
                continue

        location = ((job.get("location") or {}).get("name") or "").strip()
        results.append(
            {
                "title": title,
                "company": company_slug,
                "location": location,
                "description": description,
                "url": (job.get("absolute_url") or "").strip(),
                "salary_min": None,
                "salary_max": None,
            }
        )

    return results


if __name__ == "__main__":
    test_company = "anthropic"
    slug = find_company_slug(test_company)
    print(f"Resolved {test_company!r} -> slug={slug!r}")

    jobs = search_jobs_greenhouse(slug or test_company, keyword_filter="research")
    print(f"Found {len(jobs)} matching jobs at {test_company}.\n")
    for i, job in enumerate(jobs[:5], 1):
        print(f"[{i}] {job['title']} — {job['location']}")
        print(f"    {job['url']}")
        snippet = (job["description"] or "")[:160].replace("\n", " ")
        print(f"    {snippet}...\n")

    # Also exercise fuzzy matching
    print("Fuzzy match tests:")
    for q in ["Anthropic", "ANTHROPIC", "Hugging Face", "Mistral AI", "unknown corp"]:
        print(f"  {q!r:>20} -> {find_company_slug(q)!r}")
