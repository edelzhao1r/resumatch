"""Adzuna job-search client and pasted-JD passthrough."""

from __future__ import annotations

import os
from typing import Any

import requests

# Load .env for local development. On Streamlit Cloud, secrets come from
# st.secrets instead — see get_secret() below.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def get_secret(key: str) -> str:
    """Fetch a secret from Streamlit secrets if available, else env var."""
    try:
        import streamlit as st
        return st.secrets[key]
    except Exception:
        return os.getenv(key, "")

ADZUNA_BASE_URL = "https://api.adzuna.com/v1/api/jobs/us/search/1"


def search_jobs_adzuna(
    keyword: str,
    location: str,
    num_results: int = 10,
) -> list[dict[str, Any]]:
    """Search the Adzuna US job index.

    Args:
        keyword: Free-text query (e.g. "data scientist").
        location: Where to search (e.g. "New York", "Remote").
        num_results: Max number of jobs to return (Adzuna caps at 50).

    Returns:
        A list of dicts with keys: title, company, location, description,
        url, salary_min, salary_max.
    """
    app_id = get_secret("ADZUNA_APP_ID")
    app_key = get_secret("ADZUNA_APP_KEY")
    if not app_id or not app_key:
        raise RuntimeError(
            "ADZUNA_APP_ID and ADZUNA_APP_KEY must be set in your environment "
            "(.env locally) or in .streamlit/secrets.toml (Streamlit Cloud)."
        )

    params = {
        "app_id": app_id,
        "app_key": app_key,
        "what": keyword,
        "where": location,
        "results_per_page": max(1, min(num_results, 50)),
        "content-type": "application/json",
    }

    response = requests.get(ADZUNA_BASE_URL, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()

    jobs: list[dict[str, Any]] = []
    for item in payload.get("results", [])[:num_results]:
        jobs.append(
            {
                "title": item.get("title", "").strip(),
                "company": (item.get("company") or {}).get("display_name", "").strip(),
                "location": (item.get("location") or {}).get("display_name", "").strip(),
                "description": item.get("description", "").strip(),
                "url": item.get("redirect_url", ""),
                "salary_min": item.get("salary_min"),
                "salary_max": item.get("salary_max"),
            }
        )
    return jobs


def process_pasted_jd(
    jd_text: str,
    job_title: str = "Unknown",
    company_name: str = "Unknown",
) -> list[dict[str, Any]]:
    """Wrap a user-pasted job description into the standard job-dict shape.

    Returned as a single-element list so callers can treat it the same way
    they treat a search result set.
    """
    cleaned = (jd_text or "").strip()
    if not cleaned:
        return []
    return [
        {
            "title": (job_title or "Unknown").strip() or "Unknown",
            "company": (company_name or "Unknown").strip() or "Unknown",
            "location": "",
            "description": cleaned,
            "url": "",
            "salary_min": None,
            "salary_max": None,
        }
    ]


if __name__ == "__main__":
    sample = search_jobs_adzuna("data scientist", "New York", num_results=3)
    print(f"Found {len(sample)} jobs.\n")
    for i, job in enumerate(sample, 1):
        print(f"[{i}] {job['title']} @ {job['company']} — {job['location']}")
        print(f"    {job['url']}")
        salary_min = job.get("salary_min")
        salary_max = job.get("salary_max")
        if salary_min or salary_max:
            print(f"    Salary: {salary_min} – {salary_max}")
        snippet = (job.get("description") or "")[:160].replace("\n", " ")
        print(f"    {snippet}...\n")
