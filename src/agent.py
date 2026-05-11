"""ReAct-style agent that searches jobs (3 sources) and tailors resumes."""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Optional

from anthropic import Anthropic

from src.greenhouse_search import (
    COMPANY_SLUGS,
    find_company_slug,
    search_jobs_greenhouse,
)
from src.job_search import process_pasted_jd, search_jobs_adzuna
from src.tailoring import (
    analyze_match,
    extract_jd_insights,
    fallback_analysis,
    fallback_jd_insights,
    tailor_resume,
    track_resume_changes,
)

# Load .env for local dev; on Streamlit Cloud, secrets come from st.secrets.
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

MODEL = "claude-sonnet-4-20250514"
MAX_ITERATIONS = 3

AGENT_SYSTEM_PROMPT = """You are ResuMatch, a job-search agent helping a
candidate find the most relevant job listings for their background.

You have THREE tools, each suited to a different sourcing strategy:

1. search_jobs_adzuna(keyword, location, num_results)
   Use when the user wants a broad keyword search across many companies.
   Best for: "find me data analyst roles in Boston".

2. search_jobs_greenhouse(company_slug, keyword_filter)
   Use when the user has named specific target companies. Pulls live openings
   straight from a company's official careers board (no API key needed).
   Call this once PER company. Pass keyword_filter to narrow the role.
   Best for: "show me openings at Anthropic / Stripe / Figma".

3. process_pasted_jd(jd_text, job_title, company_name)
   Use when the user has already pasted a full job description. No search
   needed — just wrap their text so the system can tailor against it.

Loop:
- Pick the tool that matches the user's mode.
- Inspect results. If they look weak, refine and retry — but stop after at
  most 3 total tool calls.
- When satisfied, stop calling tools and write a brief final note. The
  system already has the job results; you do not need to re-list them.

Be decisive. Don't loop endlessly."""

TOOLS = [
    {
        "name": "search_jobs_adzuna",
        "description": (
            "Search jobs by keyword and location using Adzuna API. Best for "
            "broad keyword-based searches across many companies."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "Free-text job search query, e.g. 'data scientist nlp'.",
                },
                "location": {
                    "type": "string",
                    "description": "Where to search, e.g. 'New York' or 'Remote'.",
                },
                "num_results": {
                    "type": "integer",
                    "description": "How many jobs to return (1–20).",
                    "default": 10,
                },
            },
            "required": ["keyword", "location"],
        },
    },
    {
        "name": "search_jobs_greenhouse",
        "description": (
            "Get job listings directly from a specific company's official "
            "careers page via Greenhouse API. Best when user has target "
            "companies in mind. Returns real-time openings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "company_slug": {
                    "type": "string",
                    "description": (
                        "Greenhouse board slug, e.g. 'anthropic', 'stripe', "
                        "'figma'. Must match a known slug."
                    ),
                },
                "keyword_filter": {
                    "type": "string",
                    "description": (
                        "Optional substring to filter jobs by title or "
                        "description, e.g. 'data analyst'."
                    ),
                    "default": "",
                },
            },
            "required": ["company_slug"],
        },
    },
    {
        "name": "process_pasted_jd",
        "description": (
            "Use when the user has already provided a job description by "
            "pasting it directly. No search needed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "jd_text": {
                    "type": "string",
                    "description": "The full pasted job description text.",
                },
                "job_title": {
                    "type": "string",
                    "description": "Job title, if known.",
                    "default": "Unknown",
                },
                "company_name": {
                    "type": "string",
                    "description": "Company name, if known.",
                    "default": "Unknown",
                },
            },
            "required": ["jd_text"],
        },
    },
]


def _log(callback: Optional[Callable[[str], None]], message: str) -> None:
    if callback is not None:
        try:
            callback(message)
        except Exception:
            pass


def _summarize_jobs_for_model(jobs: list[dict[str, Any]]) -> str:
    """Compact representation of search results to feed back to Claude."""
    rows = []
    for i, job in enumerate(jobs, 1):
        snippet = (job.get("description") or "")[:300].replace("\n", " ")
        rows.append(
            f"[{i}] {job.get('title', '')} @ {job.get('company', '')} "
            f"({job.get('location', '')})\n    {snippet}..."
        )
    return "\n".join(rows) if rows else "(no results)"


def _build_user_message(mode: str, resume_text: str, kwargs: dict[str, Any]) -> str:
    """Compose the initial user turn so Claude knows which tool to favor."""
    resume_snippet = resume_text[:6000]
    header = "Help me find the best job listings for this candidate.\n\n"

    if mode == "adzuna":
        body = (
            f"MODE: keyword search (use search_jobs_adzuna).\n"
            f"Keyword: {kwargs.get('keyword', '')}\n"
            f"Location: {kwargs.get('location', '')}\n"
        )
    elif mode == "greenhouse":
        companies = kwargs.get("company_names") or []
        keyword_filter = kwargs.get("keyword_filter", "")
        # Resolve names → slugs so the model has them ready.
        resolved = []
        for name in companies:
            slug = find_company_slug(name)
            resolved.append(f"  - {name!r} -> slug={slug!r}")
        body = (
            "MODE: company search (use search_jobs_greenhouse, once per company).\n"
            f"Target companies and resolved slugs:\n" + "\n".join(resolved) + "\n"
            f"Optional keyword filter: {keyword_filter!r}\n"
            "If a slug is None, skip that company.\n"
        )
    elif mode == "paste":
        body = (
            "MODE: pasted JD (use process_pasted_jd, no search needed).\n"
            f"Job title: {kwargs.get('job_title', 'Unknown')}\n"
            f"Company: {kwargs.get('company_name', 'Unknown')}\n"
            "JD text is supplied via the tool argument.\n"
        )
    else:
        body = f"MODE: {mode} (unknown — pick the best tool yourself).\n"

    return (
        header
        + body
        + "\n=== CANDIDATE RESUME ===\n"
        + resume_snippet
        + "\n=== END RESUME ===\n\n"
        + "Call the appropriate tool. When results look good, stop and "
        + "write a brief final note."
    )


def _dispatch_tool(
    tool_name: str,
    args: dict[str, Any],
    log_callback: Optional[Callable[[str], None]],
) -> tuple[list[dict[str, Any]], str, bool]:
    """Run a tool call. Returns (jobs, tool_result_content, is_error)."""
    if tool_name == "search_jobs_adzuna":
        kw = args.get("keyword", "")
        loc = args.get("location", "") or "us"
        n = int(args.get("num_results", 10))
        _log(log_callback, f"  → search_jobs_adzuna(keyword={kw!r}, location={loc!r}, n={n})")
        try:
            jobs = search_jobs_adzuna(kw, loc, num_results=n)
            _log(log_callback, f"  ← {len(jobs)} jobs returned")
            return jobs, _summarize_jobs_for_model(jobs), False
        except Exception as exc:
            _log(log_callback, f"  ← search_jobs_adzuna failed: {exc}")
            return [], f"Error: {exc}", True

    if tool_name == "search_jobs_greenhouse":
        slug = args.get("company_slug", "")
        kf = args.get("keyword_filter", "")
        _log(log_callback, f"  → search_jobs_greenhouse(company_slug={slug!r}, filter={kf!r})")
        try:
            jobs = search_jobs_greenhouse(slug, keyword_filter=kf)
            _log(log_callback, f"  ← {len(jobs)} jobs returned")
            content = _summarize_jobs_for_model(jobs) if jobs else (
                f"No jobs found for slug={slug!r}. "
                "Either the slug is wrong, the company doesn't use Greenhouse, "
                "or the keyword_filter excluded everything."
            )
            return jobs, content, False
        except Exception as exc:
            _log(log_callback, f"  ← search_jobs_greenhouse failed: {exc}")
            return [], f"Error: {exc}", True

    if tool_name == "process_pasted_jd":
        jd = args.get("jd_text", "")
        title = args.get("job_title", "Unknown")
        comp = args.get("company_name", "Unknown")
        _log(
            log_callback,
            f"  → process_pasted_jd(title={title!r}, company={comp!r}, "
            f"len={len(jd)} chars)",
        )
        jobs = process_pasted_jd(jd, job_title=title, company_name=comp)
        _log(log_callback, f"  ← wrapped {len(jobs)} JD entry")
        return jobs, _summarize_jobs_for_model(jobs), False

    _log(log_callback, f"  ← unknown tool: {tool_name}")
    return [], f"Unknown tool: {tool_name}", True


def run_agent(
    resume_text: str,
    mode: str,
    log_callback: Optional[Callable[[str], None]] = None,
    tailoring_mode: str = "Balanced",
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the ResuMatch agent end-to-end.

    Args:
        resume_text: Parsed resume text.
        mode: One of "adzuna", "greenhouse", "paste". Selects the sourcing
            strategy and which kwargs are required:
              - mode="adzuna":     keyword, location
              - mode="greenhouse": company_names (list[str]), keyword_filter,
                                   location (optional substring filter)
              - mode="paste":      jd_text, job_title, company_name
        log_callback: Optional fn(str) to stream progress updates.

    Returns:
        {"jobs": [...], "tailored_resumes": [...]} where each entry in
        tailored_resumes is {"job": <job-dict>, "tailored": <str>}.
    """
    api_key = get_secret("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY must be set in your environment (.env locally) "
            "or in .streamlit/secrets.toml (Streamlit Cloud)."
        )

    # Adzuna requires a location; if the user opted out, default to US-wide.
    if mode == "adzuna" and not kwargs.get("location"):
        kwargs["location"] = "us"

    client = Anthropic(api_key=api_key)
    user_message = _build_user_message(mode, resume_text, kwargs)
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
    accumulated_jobs: list[dict[str, Any]] = []

    _log(log_callback, f"Starting agent in mode={mode!r}")

    for iteration in range(1, MAX_ITERATIONS + 1):
        _log(log_callback, f"Agent iteration {iteration}: calling Claude...")
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=AGENT_SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]

        if not tool_uses:
            text_blocks = [b.text for b in response.content if getattr(b, "type", None) == "text"]
            if text_blocks:
                _log(log_callback, f"Agent: {' '.join(text_blocks).strip()[:300]}")
            break

        tool_results_content = []
        for tool_use in tool_uses:
            args = tool_use.input or {}
            jobs, content, is_error = _dispatch_tool(tool_use.name, args, log_callback)
            if jobs:
                accumulated_jobs.extend(jobs)
            tool_results_content.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": content,
                    **({"is_error": True} if is_error else {}),
                }
            )

        messages.append({"role": "user", "content": tool_results_content})

        if response.stop_reason != "tool_use":
            break

    # Fallback if the model never produced anything usable.
    if not accumulated_jobs:
        _log(log_callback, "No jobs found via agent — running direct fallback.")
        try:
            if mode == "adzuna":
                accumulated_jobs = search_jobs_adzuna(
                    kwargs.get("keyword", ""),
                    kwargs.get("location") or "us",
                    num_results=10,
                )
            elif mode == "greenhouse":
                kf = kwargs.get("keyword_filter", "")
                for name in kwargs.get("company_names") or []:
                    slug = find_company_slug(name) or name.strip().lower()
                    accumulated_jobs.extend(search_jobs_greenhouse(slug, keyword_filter=kf))
            elif mode == "paste":
                accumulated_jobs = process_pasted_jd(
                    kwargs.get("jd_text", ""),
                    job_title=kwargs.get("job_title", "Unknown"),
                    company_name=kwargs.get("company_name", "Unknown"),
                )
        except Exception as exc:
            _log(log_callback, f"Fallback failed: {exc}")

    # De-dup by URL (Greenhouse calls per-company can produce dupes if user
    # lists the same company twice).
    seen_urls = set()
    deduped: list[dict[str, Any]] = []
    for job in accumulated_jobs:
        key = job.get("url") or f"{job.get('title')}|{job.get('company')}"
        if key in seen_urls:
            continue
        seen_urls.add(key)
        deduped.append(job)

    # Greenhouse-only: post-fetch location filter. Strict (case-insensitive)
    # substring match against the job's location field. If location is None
    # (user opted out), skip filtering entirely. If the filter wipes out
    # everything, fall back to the unfiltered list with a warning.
    if mode == "greenhouse" and kwargs.get("location") is not None:
        loc_filter = (kwargs.get("location") or "").strip().lower()
        if loc_filter:
            filtered = [
                j for j in deduped
                if loc_filter in (j.get("location") or "").lower()
            ]
            if filtered:
                _log(
                    log_callback,
                    f"Location filter {loc_filter!r}: kept "
                    f"{len(filtered)}/{len(deduped)} jobs.",
                )
                deduped = filtered
            else:
                _log(
                    log_callback,
                    f"Warning: no jobs matched location {loc_filter!r}; "
                    f"returning all {len(deduped)} unfiltered results.",
                )

    top_jobs = deduped[:10]
    tailored_resumes: list[dict[str, Any]] = []
    match_analyses: list[dict[str, Any]] = []
    jd_insights: list[dict[str, Any]] = []
    change_tracking: list[list[dict[str, Any]]] = []

    for i, job in enumerate(top_jobs, 1):
        title = job.get("title", "")
        company = job.get("company", "")

        _log(log_callback, f"Tailoring resume {i}/{len(top_jobs)} for {title} @ {company} (mode={tailoring_mode})...")
        time.sleep(4)
        try:
            tailored = tailor_resume(resume_text, job, tailoring_mode=tailoring_mode)
        except Exception as exc:
            _log(log_callback, f"  Tailoring failed: {exc}")
            tailored = f"(Tailoring failed: {exc})"
        tailored_resumes.append({"job": job, "tailored": tailored})

        _log(log_callback, f"Analyzing match {i}/{len(top_jobs)} for {title} @ {company}...")
        time.sleep(4)
        try:
            analysis = analyze_match(resume_text, job)
        except Exception as exc:
            _log(log_callback, f"  Match analysis failed: {exc}")
            analysis = fallback_analysis()
        match_analyses.append(analysis)

        _log(log_callback, f"Extracting JD insights {i}/{len(top_jobs)} for {title} @ {company}...")
        time.sleep(3)
        try:
            insights = extract_jd_insights(job)
        except Exception as exc:
            print(
                f"extract_jd_insights error for job {i}: "
                f"{type(exc).__name__}: {exc}"
            )
            _log(
                log_callback,
                f"JD insights error for job {i}: {type(exc).__name__}: {exc}",
            )
            insights = fallback_jd_insights()
        jd_insights.append(insights)

        _log(log_callback, f"Tracking resume changes {i}/{len(top_jobs)} for {title} @ {company}...")
        time.sleep(3)
        print(f"Calling track_resume_changes for job {i}: {job.get('title', '')}")
        try:
            change_tracking_result = track_resume_changes(
                resume_text, tailored, job, tailoring_mode=tailoring_mode
            )
        except Exception as exc:
            print(
                f"track_resume_changes error for job {i}: "
                f"{type(exc).__name__}: {exc}"
            )
            _log(
                log_callback,
                f"Change tracking error for job {i}: {type(exc).__name__}: {exc}",
            )
            change_tracking_result = []
        print(f"track_resume_changes result length: {len(change_tracking_result)}")
        change_tracking.append(change_tracking_result)

    _log(log_callback, "Done.")
    return {
        "jobs": top_jobs,
        "tailored_resumes": tailored_resumes,
        "match_analyses": match_analyses,
        "jd_insights": jd_insights,
        "change_tracking": change_tracking,
        "tailoring_mode": tailoring_mode,
    }


__all__ = ["run_agent", "TOOLS", "AGENT_SYSTEM_PROMPT", "COMPANY_SLUGS"]
