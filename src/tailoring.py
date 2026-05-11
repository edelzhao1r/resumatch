"""Resume tailoring + match analysis with Claude."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from anthropic import Anthropic

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

SYSTEM_PROMPT = """You are ResuMatch, an expert resume-tailoring agent.

Your job: rewrite a candidate's existing resume so it speaks directly to a
specific job description. You sharpen language, surface relevant
accomplishments, and align terminology with the JD.

STRICT RULE — DO NOT FABRICATE:
You must NEVER add any skill, tool, certification, employer, project, or
experience that is not already present (explicitly or clearly implied) in the
original resume. Rewording is allowed; inventing is not. If the JD asks for
something the candidate does not have, leave it out — do not pretend they have
it. Honesty matters more than keyword density.

STYLE:
- Keep the candidate's overall structure (sections, ordering, dates).
- Prefer strong action verbs and quantified outcomes already in the resume.
- Mirror keywords from the JD only when they truly describe existing experience.
- Output the full tailored resume as plain text, ready to paste back."""

CONSERVATIVE_SYSTEM_PROMPT = """You are a professional resume editor. Your job is to make MINIMAL changes to tailor a candidate's resume to a specific job description.

STRICT RULES:
1. Do NOT add any skills, tools, experiences, or qualifications not present in the original resume.
2. Do NOT change sentence structure or writing style.
3. ONLY make these limited changes:
   - Replace generic words with JD-specific keywords where meaning is identical
   - Add a relevant JD keyword to an existing bullet if it fits naturally in 1-2 words
   - Reorder bullets within a section to prioritize most relevant experience
4. If a bullet already matches the JD well, leave it completely unchanged.
5. Output the full resume preserving all original sections and structure.

Here are examples of CONSERVATIVE changes:
Original: 'Worked with databases to pull data for reports'
Tailored: 'Worked with SQL databases to pull data for stakeholder reports'
(only added 'SQL' and 'stakeholder' — no structural change)

Original: 'Helped team build a prediction model using Python'
Tailored: 'Helped team build a machine learning model using Python'
(only replaced 'prediction' with 'machine learning' — no structural change)"""


AGGRESSIVE_SYSTEM_PROMPT = """You are a professional resume editor. Your job is to SUBSTANTIALLY rewrite a candidate's resume to maximize alignment with a specific job description.

STRICT RULES:
1. Do NOT add any skills, tools, experiences, or qualifications not present in the original resume.
2. You MAY make these substantial changes:
   - Completely rewrite bullet points using the JD's language and framing
   - Reorder and restructure sections to lead with most relevant experience
   - Expand brief bullets into detailed achievement-focused statements
   - Reframe the candidate's experience using industry-standard terminology from the JD
   - Combine related bullets if it strengthens the narrative
3. Every rewrite must remain truthful to the candidate's actual experience.
4. Prioritize keywords and phrases that appear in the JD's requirements section.
5. Output the full tailored resume.

Here are examples of AGGRESSIVE changes:
Original: 'Worked with databases to pull data for reports'
Tailored: 'Engineered SQL-based data pipelines to extract, transform, and deliver actionable insights to cross-functional stakeholders, reducing reporting cycle time'
(substantially expanded and reframed using JD language)

Original: 'Helped team build a prediction model using Python'
Tailored: 'Collaborated with engineering team to architect and deploy a Python-based machine learning model, contributing to model design, feature engineering, and production integration'
(fully rewritten to maximize impact and keyword coverage)"""


def get_system_prompt(tailoring_mode: str) -> str:
    """Return the system prompt for the requested tailoring intensity."""
    if tailoring_mode == "Conservative":
        return CONSERVATIVE_SYSTEM_PROMPT
    if tailoring_mode == "Aggressive":
        return AGGRESSIVE_SYSTEM_PROMPT
    # Default / "Balanced" — use the existing prompt.
    return SYSTEM_PROMPT


FEW_SHOT_EXAMPLES = """Here are examples of good original-bullet → tailored-bullet rewrites.
Notice that no new tools or experience are introduced — only emphasis,
phrasing, and JD-aligned terminology change.

Example 1:
JD emphasis: "experimentation, A/B testing, causal inference"
Original:   "Built dashboards in Looker to track product KPIs."
Tailored:   "Built Looker dashboards that surfaced product KPIs and powered weekly A/B test readouts for the growth team."

Example 2:
JD emphasis: "production ML, MLOps, model monitoring"
Original:   "Trained an XGBoost churn model and deployed it via a Flask API."
Tailored:   "Shipped an XGBoost churn model to production behind a Flask service, including request logging and weekly drift checks to monitor live performance."

Example 3:
JD emphasis: "stakeholder communication, executive reporting"
Original:   "Presented quarterly findings to the analytics team."
Tailored:   "Presented quarterly analytics findings to cross-functional partners and translated results into clear takeaways for non-technical stakeholders."
"""

USER_TEMPLATE = """Tailor the following resume to the job description below.

=== JOB DESCRIPTION ===
Title: {title}
Company: {company}
Location: {location}

{description}

=== ORIGINAL RESUME ===
{resume}

=== INSTRUCTIONS ===
Return ONLY the full tailored resume as plain text. Do not include
explanations, preambles, or markdown fences. Remember: do not add any skill,
tool, or experience that is not already in the original resume."""


def tailor_resume(
    resume_text: str,
    job: dict[str, Any],
    tailoring_mode: str = "Balanced",
) -> str:
    """Produce a JD-tailored version of the resume using Claude.

    Args:
        resume_text: Plain-text resume.
        job: Dict with at least keys title, company, location, description.
        tailoring_mode: One of "Conservative", "Balanced", "Aggressive".
            Selects how aggressively the resume is rewritten.

    Returns:
        The tailored resume as a single plain-text string.
    """
    api_key = get_secret("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY must be set in your environment (.env locally) "
            "or in .streamlit/secrets.toml (Streamlit Cloud)."
        )

    client = Anthropic(api_key=api_key)

    description = job.get("description", "") or ""
    if len(description) > 1500:
        description = description[:1500] + "..."

    mode_line = (
        f"Tailoring Mode: {tailoring_mode} — apply the corresponding level of "
        f"changes as instructed in the system prompt."
    )

    base_user = USER_TEMPLATE.format(
        title=job.get("title", ""),
        company=job.get("company", ""),
        location=job.get("location", ""),
        description=description,
        resume=resume_text,
    )

    # Only "Balanced" uses the original FEW_SHOT_EXAMPLES prefix; the
    # Conservative and Aggressive system prompts already include their own
    # inline examples.
    if tailoring_mode == "Balanced":
        user_message = mode_line + "\n\n" + FEW_SHOT_EXAMPLES + "\n\n" + base_user
    else:
        user_message = mode_line + "\n\n" + base_user

    response = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        system=get_system_prompt(tailoring_mode),
        messages=[{"role": "user", "content": user_message}],
    )

    parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
    return "\n".join(parts).strip()


# ============================================================
# Match analysis
# ============================================================

ANALYSIS_SYSTEM_PROMPT = (
    "You are a professional resume analyst. Analyze how well a candidate's "
    "resume matches a job description. Be honest and specific. Return ONLY "
    "valid JSON with no preamble, no markdown, no backticks."
)


def fallback_analysis() -> dict[str, Any]:
    """The analysis dict returned when Claude's output can't be parsed."""
    return {
        "match_score": 0,
        "confidence": "Low",
        "confidence_reason": "Analysis unavailable",
        "strong_matches": [],
        "weak_areas": ["Match analysis could not be completed"],
        "why_selected": "Please review this job manually.",
    }


def _parse_analysis_json(text: str) -> dict[str, Any]:
    """Parse Claude's JSON output, tolerating fences / stray text."""
    if not text:
        return fallback_analysis()

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # First try: parse the whole thing.
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Second try: snip out the first {...} block.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(cleaned[start : end + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    return fallback_analysis()


def analyze_match(resume_text: str, job: dict[str, Any]) -> dict[str, Any]:
    """Use Claude to score how well a resume matches a JD.

    Returns a dict with keys: match_score, confidence, confidence_reason,
    strong_matches, weak_areas, why_selected. On any parse failure, returns
    the ``fallback_analysis()`` dict.
    """
    api_key = get_secret("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY must be set in your environment (.env locally) "
            "or in .streamlit/secrets.toml (Streamlit Cloud)."
        )

    client = Anthropic(api_key=api_key)

    description = job.get("description", "") or ""
    if len(description) > 1500:
        description = description[:1500] + "..."

    user_msg = (
        "Analyze the match between this resume and job description.\n\n"
        f"Job Title: {job.get('title', '')}\n"
        f"Company: {job.get('company', '')}\n"
        f"Job Description: {description}\n\n"
        "Resume:\n"
        f"{resume_text}\n\n"
        "Return a JSON object with exactly these fields:\n"
        "{\n"
        "  'match_score': integer 0-100,\n"
        "  'confidence': one of 'High', 'Medium', 'Low',\n"
        "  'confidence_reason': one sentence explaining why confidence is High/Medium/Low,\n"
        "  'strong_matches': list of 2-4 strings, each describing a specific alignment between resume and JD,\n"
        "  'weak_areas': list of 1-3 strings, each describing a gap or missing requirement,\n"
        "  'why_selected': 2-3 sentence summary of why this job is a good match overall despite any gaps\n"
        "}\n\n"
        "Confidence level guidance:\n"
        "- High: resume clearly meets most requirements, specific evidence available\n"
        "- Medium: partial match, some requirements met but notable gaps exist\n"
        "- Low: limited overlap, match is based on general transferable skills\n\n"
        "Be specific and reference actual skills, tools, or experiences from "
        "the resume and JD. Do not invent skills not present in the resume."
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=600,
        system=ANALYSIS_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    text = "\n".join(
        b.text for b in response.content if getattr(b, "type", None) == "text"
    ).strip()
    return _parse_analysis_json(text)


# ============================================================
# JD insights (keyword extraction + visa sponsorship)
# ============================================================

JD_INSIGHTS_SYSTEM_PROMPT = (
    "You are a job description analyst. Extract structured information from "
    "job descriptions. Return ONLY valid JSON with no preamble, no markdown, "
    "no backticks."
)


def fallback_jd_insights() -> dict[str, Any]:
    """Returned when JD-insight parsing fails."""
    return {
        "skill_keywords": [],
        "attribute_keywords": [],
        "visa_sponsorship": "Unknown",
        "visa_note": "Could not analyze job description",
    }


def _strip_code_fences(raw_text: str) -> str:
    """Strip ```json / ``` markdown fences and trim whitespace.

    Uses pure string ops (no regex) so it's bulletproof against any casing,
    extra spaces, or trailing whitespace inside the fence line itself.
    Handles: ``` ```json\\n...\\n``` ```, ``` ```JSON\\n...\\n``` ```,
    bare ``` ```\\n...\\n``` ```, single-line ``` ```json[...] ``` ```,
    and content with no fences at all.
    """
    text = (raw_text or "").strip()

    # Remove opening code fence
    if text.startswith("```"):
        nl = text.find("\n")
        if nl != -1:
            text = text[nl:].strip()
        else:
            # No newline after fence — strip ``` and optional language label.
            text = text[3:].strip()
            lower = text.lower()
            for lang in ("json",):
                if lower.startswith(lang):
                    text = text[len(lang):].strip()
                    break

    # Remove closing code fence
    if text.endswith("```"):
        text = text[: text.rfind("```")].strip()

    return text


def _parse_dict_json(text: str, fallback) -> dict[str, Any]:
    """Tolerant JSON-object parser. ``fallback`` is a zero-arg factory."""
    if not text:
        return fallback()

    cleaned = _strip_code_fences(text)

    # Extract just the JSON object between the first { and last }.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]

    print(f"Cleaned JSON text (first 200 chars): {cleaned[:200]}")

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    return fallback()


def _recover_truncated_array(text: str) -> str | None:
    """Salvage a truncated JSON array by trimming to the last complete object.

    Given e.g. ``[{"a":"b"},{"c":"d"},{"e":"<cut off here``, returns
    ``[{"a":"b"},{"c":"d"}]``. Returns None if no complete top-level object
    could be recovered, or if the input isn't an array.
    """
    if not text.startswith("["):
        return None

    depth = 0
    in_string = False
    escape = False
    last_complete_end = -1  # exclusive index just past a top-level `}`

    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if in_string:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 1 and ch == "}":
                # Just closed an object at the top level of the outer array.
                last_complete_end = i + 1
            elif depth == 0 and ch == "]":
                # The outer array closed cleanly — no recovery needed.
                return text[: i + 1]

    if last_complete_end > 0:
        return text[:last_complete_end] + "]"
    return None


def _coerce_to_dict_list(data: Any) -> list[dict[str, Any]] | None:
    """Accept either a JSON array of dicts, or a dict whose first list value
    is the array we want (e.g. ``{"changes": [...]}``). Returns None if
    neither shape applies."""
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return None


def _parse_list_json(text: str) -> list[dict[str, Any]]:
    """Tolerant JSON-array parser. Handles bare arrays AND dict-wrapped arrays
    like ``{"changes": [...]}``. Returns [] on any failure.

    Uses pure string ops to clean fences (see _strip_code_fences) and then
    slices to the first [...] block, so any preamble or trailing chatter is
    safely discarded.
    """
    if not text:
        return []

    cleaned = _strip_code_fences(text)

    # First preference: extract the [...] array slice. This naturally handles
    # both `[...]` and dict-wrapped `{"changes": [...]}` because the slice
    # captures everything between the first `[` and the last `]`.
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start : end + 1]
        print(f"Cleaned JSON text (first 200 chars): {candidate[:200]}")
        try:
            data = json.loads(candidate)
            coerced = _coerce_to_dict_list(data)
            if coerced is not None:
                return coerced
        except json.JSONDecodeError as exc:
            # Likely a max_tokens cutoff — try to recover the prefix.
            print(f"json.loads failed on bracket slice: {exc}; attempting recovery...")

    # Recovery path: if we never closed `]` (Claude was cut off by max_tokens),
    # take the candidate that begins with `[` and truncate to the last complete
    # `{...}` object, then close the array.
    if start != -1:
        # `cleaned[start:]` covers the case where rfind("]") returned -1 OR
        # returned a position inside a still-incomplete object.
        truncated_source = cleaned[start:]
        recovered = _recover_truncated_array(truncated_source)
        if recovered is not None:
            print(f"Recovered JSON prefix (first 200 chars): {recovered[:200]}")
            try:
                data = json.loads(recovered)
                coerced = _coerce_to_dict_list(data)
                if coerced is not None:
                    print(f"Recovered {len(coerced)} complete object(s) from truncated response.")
                    return coerced
            except json.JSONDecodeError as exc:
                print(f"Recovered JSON also failed to parse: {exc}")

    # Fallback: maybe Claude returned just an object with a list value and
    # no brackets at the outer level we expected (unlikely after the slice
    # above, but kept as a safety net).
    print(f"Cleaned JSON text (first 200 chars): {cleaned[:200]}")
    try:
        data = json.loads(cleaned)
        coerced = _coerce_to_dict_list(data)
        if coerced is not None:
            return coerced
    except json.JSONDecodeError:
        pass

    o_start = cleaned.find("{")
    o_end = cleaned.rfind("}")
    if o_start != -1 and o_end != -1 and o_end > o_start:
        try:
            data = json.loads(cleaned[o_start : o_end + 1])
            coerced = _coerce_to_dict_list(data)
            if coerced is not None:
                return coerced
        except json.JSONDecodeError:
            pass

    return []


def extract_jd_insights(job: dict[str, Any]) -> dict[str, Any]:
    """Extract skill/attribute keywords and visa-sponsorship signal from a JD."""
    api_key = get_secret("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY must be set in your environment (.env locally) "
            "or in .streamlit/secrets.toml (Streamlit Cloud)."
        )

    client = Anthropic(api_key=api_key)

    description = job.get("description", "") or ""
    if len(description) > 1500:
        description = description[:1500] + "..."

    user_msg = (
        "Analyze this job description and return a JSON object with exactly "
        "these fields:\n"
        "{\n"
        "  'skill_keywords': list of 4-8 specific technical skills or tools "
        "mentioned (e.g. Python, SQL, Tableau, AWS),\n"
        "  'attribute_keywords': list of 2-4 job attribute tags (e.g. "
        "Entry-level, Senior, Remote, Full-time, Part-time, Internship),\n"
        "  'visa_sponsorship': one of 'Yes', 'No', 'Unknown',\n"
        "  'visa_note': one short sentence explaining why (e.g. 'JD states "
        "visa sponsorship is not available' or 'No mention of visa "
        "sponsorship found')\n"
        "}\n\n"
        "Visa sponsorship rules:\n"
        "- Yes: JD contains phrases like 'visa sponsorship available', 'will "
        "sponsor', 'H-1B sponsored', 'sponsorship provided'\n"
        "- No: JD contains phrases like 'no sponsorship', 'must be authorized "
        "to work', 'US citizens only', 'no visa support', 'cannot sponsor'\n"
        "- Unknown: no relevant mention found\n\n"
        f"Job Description:\n{description}"
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        system=JD_INSIGHTS_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = "\n".join(
        b.text for b in response.content if getattr(b, "type", None) == "text"
    ).strip()
    return _parse_dict_json(text, fallback_jd_insights)


# ============================================================
# Resume change tracking
# ============================================================

CHANGE_TRACKING_SYSTEM_PROMPT = (
    "You are a resume editing analyst. Compare original and tailored resumes "
    "and classify each change. Return ONLY valid JSON with no preamble, no "
    "markdown, no backticks."
)


def track_resume_changes(
    original_text: str,
    tailored_text: str,
    job: dict[str, Any],
    tailoring_mode: str = "Balanced",
) -> list[dict[str, Any]]:
    """Identify and classify the most significant edits made during tailoring.

    Returns a list of dicts, each with keys:
        original, tailored, change_type, explanation.
    Empty list on parse error.
    """
    api_key = get_secret("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY must be set in your environment (.env locally) "
            "or in .streamlit/secrets.toml (Streamlit Cloud)."
        )

    client = Anthropic(api_key=api_key)

    description = job.get("description", "") or ""
    if len(description) > 800:
        description = description[:800] + "..."

    user_msg = (
        f"Note: This resume was tailored in {tailoring_mode} mode.\n\n"
        "Compare the original and tailored resume below. Identify the most "
        "significant changes (maximum 8).\n\n"
        f"Job Title: {job.get('title', '')}\n"
        f"Job Description (excerpt): {description}\n\n"
        "Original Resume:\n"
        f"{original_text}\n\n"
        "Tailored Resume:\n"
        f"{tailored_text}\n\n"
        "Return a JSON array where each element has exactly these fields:\n"
        "{\n"
        "  'original': the original sentence or bullet point (or empty string "
        "if this is a pure addition),\n"
        "  'tailored': the tailored sentence or bullet point,\n"
        "  'change_type': one of 'Added keyword', 'Rewritten bullet', "
        "'Potentially exaggerated',\n"
        "  'explanation': one short sentence explaining why this change was "
        "made or flagged\n"
        "}\n\n"
        "Change type rules:\n"
        "- 'Added keyword': a JD keyword was naturally incorporated into an "
        "existing bullet\n"
        "- 'Rewritten bullet': the bullet was substantially reworded to "
        "better match the JD tone or priorities\n"
        "- 'Potentially exaggerated': the tailored version makes a stronger "
        "claim than the original in a way that may not be fully supported\n\n"
        "Return maximum 8 changes. Focus on the most meaningful ones."
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=3000,
        system=CHANGE_TRACKING_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw_text = "\n".join(
        b.text for b in response.content if getattr(b, "type", None) == "text"
    ).strip()
    stop_reason = getattr(response, "stop_reason", "unknown")

    # DEBUG: include length + stop_reason so we can tell whether Claude was
    # cut off by max_tokens (stop_reason == "max_tokens").
    print(
        f"track_resume_changes raw response "
        f"(len={len(raw_text)}, stop_reason={stop_reason}): "
        f"{raw_text[:500]}"
    )

    parsed = _parse_list_json(raw_text)
    if not isinstance(parsed, list):
        print(
            f"track_resume_changes warning: parsed result is "
            f"{type(parsed).__name__}, not a list. Returning []."
        )
        return []
    return parsed


__all__ = [
    "tailor_resume",
    "analyze_match",
    "fallback_analysis",
    "extract_jd_insights",
    "fallback_jd_insights",
    "track_resume_changes",
]
