"""ResuMatch — Streamlit UI with three job-sourcing modes."""

from __future__ import annotations

import difflib
import html
import os
import re

import streamlit as st
from dotenv import load_dotenv

from src.agent import run_agent
from src.docx_generator import generate_tailored_docx
from src.greenhouse_search import COMPANY_SLUGS
from src.resume_parser import parse_resume

load_dotenv()

st.set_page_config(page_title="ResuMatch", page_icon=":briefcase:", layout="wide")

# ============================================================
# Global CSS — color theme + base styling
# ============================================================
st.markdown(
    """
    <style>
    :root {
      --primary: #2E86AB;
      --primary-light: #A8DADC;
      --bg-main: #EFF9F9;
      --bg-card: #FFFFFF;
      --bg-section: #F0F7FF;
      --success: #52B788;
      --warning: #F4A261;
      --danger: #E63946;
      --text-primary: #1B2D3E;
      --text-secondary: #4A6572;
      --border: #C8E6E8;
    }

    .stApp { background-color: var(--bg-main); }

    .main .block-container {
      max-width: 1100px;
      padding-top: 2rem;
      padding-bottom: 4rem;
    }

    .stButton > button {
      background-color: var(--primary);
      color: white;
      border: none;
      border-radius: 8px;
      padding: 0.5rem 1.5rem;
      font-weight: 600;
      font-size: 0.95rem;
      transition: background-color 0.2s ease;
    }
    .stButton > button:hover { background-color: #1a6fa8; }
    .stButton > button:disabled {
      background-color: #b0c4ce;
      color: #f0f0f0;
    }

    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
      border: 1.5px solid var(--border);
      border-radius: 8px;
      background-color: white;
      color: var(--text-primary);
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 2px rgba(46,134,171,0.15);
    }

    .stSelectbox > div > div,
    .stMultiSelect > div > div {
      border: 1.5px solid var(--border);
      border-radius: 8px;
      background-color: white;
    }

    .streamlit-expanderHeader {
      background-color: var(--bg-section);
      border: 1.5px solid var(--border);
      border-radius: 10px;
      font-weight: 600;
      color: var(--text-primary);
    }
    .streamlit-expanderContent {
      border: 1.5px solid var(--border);
      border-top: none;
      border-radius: 0 0 10px 10px;
      background-color: var(--bg-card);
    }

    .stTabs [data-baseweb="tab-list"] {
      background-color: var(--bg-section);
      border-radius: 8px;
      padding: 4px;
      gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
      border-radius: 6px;
      color: #4A6572 !important;
      font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
      background-color: #2E86AB !important;
      color: white !important;
      border-radius: 6px !important;
    }
    /* Override Streamlit's default red tab-highlight underline */
    .stTabs [data-baseweb="tab-highlight"] {
      background-color: #2E86AB !important;
    }

    [data-testid="stMetric"] {
      background-color: var(--bg-section);
      border: 1.5px solid var(--border);
      border-radius: 10px;
      padding: 1rem;
    }

    .stCaption { color: var(--text-secondary); }

    [data-testid="stDataFrame"] {
      border: 1.5px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
    }

    [data-testid="stSidebar"] { background-color: #D6EEF0; }

    hr { border-color: var(--border); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Hero
# ============================================================
st.markdown(
    """
    <div style="
      background: linear-gradient(135deg, #2E86AB 0%, #52B788 100%);
      border-radius: 16px;
      padding: 2.5rem 3rem;
      margin-bottom: 2rem;
      color: white;
    ">
      <h1 style="margin:0; font-size:2.4rem; font-weight:700; letter-spacing:-0.5px;">
        ResuMatch
      </h1>
      <p style="margin:0.5rem 0 0 0; font-size:1.1rem; opacity:0.9;">
        Upload your resume. Find matched jobs. Get a tailored resume — automatically.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# UI helpers
# ============================================================
def render_step_indicator(current_step: int) -> None:
    """Three-step horizontal progress indicator (Upload / Find / View)."""
    steps = [(1, "Upload Resume"), (2, "Find Jobs"), (3, "View Results")]
    parts = ['<div style="display:flex; align-items:center; gap:0.5rem; margin:0 0 1.5rem 0;">']
    for i, (num, label) in enumerate(steps):
        if num < current_step:
            circle_bg = "#52B788"
            text_color = "#1B2D3E"
            weight = 600
        elif num == current_step:
            circle_bg = "#2E86AB"
            text_color = "#1B2D3E"
            weight = 700
        else:
            circle_bg = "#ddd"
            text_color = "#999"
            weight = 400
        parts.append(
            f'<div style="display:flex; align-items:center; gap:0.5rem;">'
            f'  <div style="width:32px; height:32px; border-radius:50%; background:{circle_bg};'
            f'              display:flex; align-items:center; justify-content:center;'
            f'              color:white; font-weight:700; font-size:0.9rem;">{num}</div>'
            f'  <span style="font-weight:{weight}; color:{text_color}; font-size:0.95rem;">{label}</span>'
            f'</div>'
        )
        if i < len(steps) - 1:
            line_color = "#52B788" if num < current_step else "#ddd"
            parts.append(
                f'<div style="flex:1; height:2px; background:{line_color}; margin:0 0.5rem;"></div>'
            )
    parts.append("</div>")
    st.markdown("\n".join(parts), unsafe_allow_html=True)


def section_header(title: str) -> None:
    """Gradient-bar styled section header (replaces st.header)."""
    st.markdown(
        f"""
        <div style="
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin: 1.5rem 0 1rem 0;
        ">
          <div style="
            width: 4px;
            height: 28px;
            background: linear-gradient(180deg, #2E86AB, #52B788);
            border-radius: 2px;
          "></div>
          <h2 style="
            margin: 0;
            font-size: 1.3rem;
            font-weight: 700;
            color: var(--text-primary);
          ">{html.escape(title)}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Per-mode expected step lists for the agent progress display.
EXPECTED_STEPS: dict[str, list[str]] = {
    "adzuna": [
        "Searching jobs via Adzuna",
        "Evaluating matches",
        "Extracting JD insights",
        "Tailoring resumes",
        "Generating analysis",
        "Complete",
    ],
    "greenhouse": [
        "Fetching company listings",
        "Evaluating matches",
        "Extracting JD insights",
        "Tailoring resumes",
        "Generating analysis",
        "Complete",
    ],
    "paste": [
        "Processing job description",
        "Extracting JD insights",
        "Tailoring resume",
        "Generating analysis",
        "Complete",
    ],
}


def init_step_state(mode: str) -> None:
    """Reset the agent step status for a fresh run."""
    steps = EXPECTED_STEPS.get(mode, [])
    st.session_state.agent_run_mode = mode
    st.session_state.agent_step_status = {s: "pending" for s in steps}
    if steps:
        st.session_state.agent_step_status[steps[0]] = "in_progress"


def _set_in_progress(steps: list[str], status: dict, idx: int) -> None:
    for j in range(idx):
        status[steps[j]] = "completed"
    if status.get(steps[idx]) != "completed":
        status[steps[idx]] = "in_progress"


def _complete_through(steps: list[str], status: dict, idx: int) -> None:
    for j in range(idx + 1):
        status[steps[j]] = "completed"
    if idx + 1 < len(steps) and status[steps[idx + 1]] == "pending":
        status[steps[idx + 1]] = "in_progress"


def advance_steps(message: str) -> None:
    """Map an agent log message into a step-status transition."""
    mode = st.session_state.get("agent_run_mode")
    if not mode or mode not in EXPECTED_STEPS:
        return
    steps = EXPECTED_STEPS[mode]
    status = st.session_state.agent_step_status
    msg = message.lower()

    def find(prefix: str) -> int:
        for i, s in enumerate(steps):
            if s.lower().startswith(prefix.lower()):
                return i
        return -1

    if "done" in msg or "complete" in msg:
        for s in steps:
            status[s] = "completed"
        return

    if "tailoring" in msg or "tailor" in msg:
        idx = find("Tailoring")
        if idx >= 0:
            _set_in_progress(steps, status, idx)
        return

    if "tracking" in msg or "analysis" in msg or "analyzing" in msg:
        idx = find("Generating")
        if idx == -1:
            idx = find("Evaluating")
        if idx >= 0:
            _set_in_progress(steps, status, idx)
        return

    if "insights" in msg or "extracted" in msg or "extracting" in msg:
        idx = find("Extracting")
        if idx >= 0:
            if "extracting" in msg:
                _set_in_progress(steps, status, idx)
            else:
                _complete_through(steps, status, idx)
        return

    if "search" in msg or "fetch" in msg or "retrieved" in msg or "returned" in msg or "process" in msg:
        # First step (Searching / Fetching / Processing) is complete.
        _complete_through(steps, status, 0)
        return


def render_progress_steps() -> str:
    """HTML for the structured progress display."""
    mode = st.session_state.get("agent_run_mode")
    if not mode:
        return ""
    steps = EXPECTED_STEPS.get(mode, [])
    status = st.session_state.get("agent_step_status", {})
    rows = []
    for s in steps:
        st_state = status.get(s, "pending")
        if st_state == "completed":
            icon = "✅"
            color = "var(--text-primary)"
            weight = 500
        elif st_state == "in_progress":
            icon = "⏳"
            color = "var(--primary)"
            weight = 700
        else:
            icon = "○"
            color = "#aaa"
            weight = 400
        rows.append(
            f'<div style="display:flex; align-items:center; gap:0.6rem; padding:0.35rem 0;">'
            f'  <span style="font-size:1.1rem; width:1.4rem; text-align:center;">{icon}</span>'
            f'  <span style="color:{color}; font-weight:{weight};">{html.escape(s)}</span>'
            f'</div>'
        )
    return (
        '<div style="background: var(--bg-section); border: 1.5px solid var(--border);'
        ' border-radius: 12px; padding: 1.25rem 1.5rem; margin: 1rem 0;">'
        + "\n".join(rows)
        + "</div>"
    )


# ============================================================
# ============================================================
# Result-card helpers (summary bar, progress bar, side-by-side diff)
# ============================================================
def render_match_summary_bar(score: int, confidence: str) -> str:
    """Compact top-of-expander summary: badge + score + confidence."""
    if score >= 75:
        badge_bg, badge_color, badge_text = "#EAFAF1", "#1E8449", "Strong Match"
    elif score >= 50:
        badge_bg, badge_color, badge_text = "#FEF9E7", "#B7770D", "Moderate Match"
    else:
        badge_bg, badge_color, badge_text = "#FDECEA", "#C0392B", "Low Match"
    return f"""
    <div style="
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: #F8FBFF;
      border: 1px solid #C8E6E8;
      border-radius: 8px;
      padding: 0.6rem 1rem;
      margin-bottom: 1rem;
    ">
      <div style="display:flex; align-items:center; gap:1rem;">
        <span style="
          background:{badge_bg};
          color:{badge_color};
          padding:0.25rem 0.75rem;
          border-radius:20px;
          font-weight:700;
          font-size:0.85rem;
        ">{badge_text}</span>
        <span style="color:#4A6572; font-size:0.9rem;">
          Match Score: <strong style="color:#1B2D3E;">{score} / 100</strong>
        </span>
      </div>
      <span style="color:#4A6572; font-size:0.85rem;">
        Confidence: <strong style="color:#2E86AB;">{html.escape(str(confidence))}</strong>
      </span>
    </div>
    """


def render_match_progress_bar(score: int) -> str:
    """Visual progress bar that fills proportional to the match score."""
    if score >= 75:
        bar_color = "#52B788"
    elif score >= 50:
        bar_color = "#F4A261"
    else:
        bar_color = "#E63946"
    pct = max(0, min(100, score))
    return f"""
    <div style="
      background: #e8f0f2;
      border-radius: 20px;
      height: 8px;
      width: 100%;
      margin-top: 0.4rem;
      overflow: hidden;
    ">
      <div style="
        background: {bar_color};
        width: {pct}%;
        height: 100%;
        border-radius: 20px;
        transition: width 0.5s ease;
      "></div>
    </div>
    """


def render_side_by_side_diff(original: str, tailored: str) -> str:
    """Render a two-column added/removed/unchanged HTML diff."""
    original_lines = original.splitlines()
    tailored_lines = tailored.splitlines()
    matcher = difflib.SequenceMatcher(None, original_lines, tailored_lines)

    def _esc(line: str) -> str:
        # Empty lines need a non-breaking space so the row keeps its height.
        return html.escape(line) if line else "&nbsp;"

    def _unchanged(line: str) -> str:
        return f'<p style="margin:0.15rem 0; color:#1B2D3E; padding:0.1rem 0.4rem;">{_esc(line)}</p>'

    def _removed(line: str) -> str:
        return (
            '<p style="margin:0.15rem 0; background:#FDECEA; color:#C0392B; '
            'padding:0.1rem 0.4rem; border-radius:3px; '
            f'text-decoration:line-through;">{_esc(line)}</p>'
        )

    def _added(line: str) -> str:
        return (
            '<p style="margin:0.15rem 0; background:#EAFAF1; color:#1E8449; '
            f'padding:0.1rem 0.4rem; border-radius:3px;">{_esc(line)}</p>'
        )

    _placeholder = (
        '<p style="margin:0.15rem 0; padding:0.1rem 0.4rem; color:transparent;">.</p>'
    )

    left_parts: list[str] = []
    right_parts: list[str] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for line in original_lines[i1:i2]:
                left_parts.append(_unchanged(line))
                right_parts.append(_unchanged(line))
        elif tag == "replace":
            old_chunk = original_lines[i1:i2]
            new_chunk = tailored_lines[j1:j2]
            n = max(len(old_chunk), len(new_chunk))
            for k in range(n):
                left_parts.append(_removed(old_chunk[k]) if k < len(old_chunk) else _placeholder)
                right_parts.append(_added(new_chunk[k]) if k < len(new_chunk) else _placeholder)
        elif tag == "delete":
            for line in original_lines[i1:i2]:
                left_parts.append(_removed(line))
                right_parts.append(_placeholder)
        elif tag == "insert":
            for line in tailored_lines[j1:j2]:
                left_parts.append(_placeholder)
                right_parts.append(_added(line))

    left_html = "\n".join(left_parts) or _placeholder
    right_html = "\n".join(right_parts) or _placeholder

    return f"""
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:1rem; font-family:'Calibri',sans-serif; font-size:0.88rem; line-height:1.6;">
      <div style="background:white; border:1.5px solid #C8E6E8; border-radius:10px; overflow:hidden;">
        <div style="background:#F0F7FF; padding:0.6rem 1rem; font-weight:700; color:#1B2D3E; border-bottom:1px solid #C8E6E8; font-size:0.85rem;">
          Original Resume
        </div>
        <div style="padding:1rem; max-height:500px; overflow-y:auto;">
          {left_html}
        </div>
      </div>
      <div style="background:white; border:1.5px solid #C8E6E8; border-radius:10px; overflow:hidden;">
        <div style="background:#F0F7FF; padding:0.6rem 1rem; font-weight:700; color:#1B2D3E; border-bottom:1px solid #C8E6E8; font-size:0.85rem;">
          Tailored Resume
        </div>
        <div style="padding:1rem; max-height:500px; overflow-y:auto;">
          {right_html}
        </div>
      </div>
    </div>
    """


DIFF_LEGEND_HTML = (
    '<div style="display:flex; gap:1.5rem; margin-bottom:0.75rem; '
    'font-size:0.82rem; color:#4A6572;">'
    '<span><span style="background:#EAFAF1; color:#1E8449; padding:1px 8px; '
    'border-radius:3px;">Added</span></span>'
    '<span><span style="background:#FDECEA; color:#C0392B; padding:1px 8px; '
    'border-radius:3px; text-decoration:line-through;">Removed</span></span>'
    '<span><span style="background:white; border:1px solid #ddd; '
    'padding:1px 8px; border-radius:3px;">Unchanged</span></span>'
    '</div>'
)


# ============================================================
# Constants — US states and a city dictionary for the location picker
# ============================================================
US_STATES: list[str] = sorted([
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "District of Columbia", "Florida", "Georgia",
    "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky",
    "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan",
    "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "West Virginia", "Wisconsin", "Wyoming",
])

CITIES_BY_STATE: dict[str, list[str]] = {
    "California": ["San Francisco", "Los Angeles", "San Jose", "San Diego", "Other"],
    "New York": ["New York City", "Buffalo", "Albany", "Other"],
    "Washington": ["Seattle", "Bellevue", "Spokane", "Other"],
    "Texas": ["Austin", "Houston", "Dallas", "San Antonio", "Other"],
    "Massachusetts": ["Boston", "Cambridge", "Other"],
    "Illinois": ["Chicago", "Other"],
    "Georgia": ["Atlanta", "Other"],
    "Virginia": ["Arlington", "McLean", "Richmond", "Other"],
    "Maryland": ["Baltimore", "Bethesda", "Other"],
    "District of Columbia": ["Washington DC"],
}


# ============================================================
# Helpers
# ============================================================
def _company_display_options() -> dict[str, str]:
    """Build {display_name: greenhouse_slug}, deduped + sorted.

    COMPANY_SLUGS contains aliases (e.g. "scale" and "scale ai" both map to
    "scaleai"). For each unique slug we pick the longest alias as the
    canonical name and prettify it (Title Case, but keep "AI" uppercase).
    """
    slug_to_aliases: dict[str, list[str]] = {}
    for alias, slug in COMPANY_SLUGS.items():
        slug_to_aliases.setdefault(slug, []).append(alias)

    options: dict[str, str] = {}
    for slug, aliases in slug_to_aliases.items():
        canonical = max(aliases, key=len)
        words = canonical.split()
        display = " ".join(
            w.upper() if w.lower() == "ai" else w.title() for w in words
        )
        options[display] = slug
    return dict(sorted(options.items()))


def render_location_selector(key_prefix: str) -> str | None:
    """Render an optional State + City picker.

    Renders an enable-checkbox first; the State/City widgets only appear if
    the checkbox is checked.

    Returns:
        - None                              if the checkbox is unchecked
        - "remote"                          if State == Remote
        - whatever the user types           if State == International / Other
        - "<city>, <state>"                 if a known city is picked
        - "<free text>, <state>"            if city == Other and free text is given
        - "<state>"                         if no city detail available
    """
    use_location = st.checkbox(
        "Filter by location (optional)",
        value=False,
        key=f"{key_prefix}_use_location",
    )
    if not use_location:
        return None

    state_options = ["Remote"] + US_STATES + ["International / Other"]
    state = st.selectbox(
        "State",
        state_options,
        index=0,
        key=f"{key_prefix}_state",
    )

    if state == "Remote":
        return "remote"

    if state == "International / Other":
        free = st.text_input(
            "Location (free text)",
            placeholder="e.g. London, UK",
            key=f"{key_prefix}_intl",
        )
        return free.strip()

    cities = CITIES_BY_STATE.get(state, ["Other"])
    city = st.selectbox(
        "City",
        cities,
        key=f"{key_prefix}_city",
    )

    if city == "Other":
        free = st.text_input(
            f"City in {state}",
            placeholder="e.g. Sacramento",
            key=f"{key_prefix}_city_other",
        )
        free = free.strip()
        return f"{free}, {state}" if free else state

    return f"{city}, {state}"


# ============================================================
# Session state
# ============================================================
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "agent_result" not in st.session_state:
    st.session_state.agent_result = None
if "agent_logs" not in st.session_state:
    st.session_state.agent_logs = []


# ============================================================
# Top-level step indicator (1 → 2 → 3) — depends on session state
# ============================================================
def _current_step() -> int:
    if not st.session_state.get("resume_text"):
        return 1
    if not st.session_state.get("agent_result"):
        return 2
    return 3


render_step_indicator(_current_step())


# ============================================================
# Step 1: Resume upload (shared across all tabs)
# ============================================================
section_header("Step 1: Upload Your Resume")

# Open upload-area card
st.markdown(
    """
    <div style="
      background: white;
      border: 1.5px solid var(--border);
      border-radius: 12px;
      padding: 1.5rem;
      margin-bottom: 1rem;
    ">
    """,
    unsafe_allow_html=True,
)

uploaded = st.file_uploader("Resume (PDF or DOCX)", type=["pdf", "docx"])
if uploaded is not None:
    try:
        text = parse_resume(uploaded)
        st.session_state.resume_text = text
        ext = os.path.splitext(uploaded.name)[1].lstrip(".").upper() or "FILE"
        st.markdown(
            f"""
            <div style="
              display: inline-flex;
              align-items: center;
              gap: 0.4rem;
              background: #EAFAF1;
              color: #27AE60;
              border: 1px solid #A9DFBF;
              border-radius: 20px;
              padding: 0.3rem 0.9rem;
              font-size: 0.85rem;
              font-weight: 600;
              margin-top: 0.5rem;
            ">
              ✓ Resume ready ({ext}, {len(text):,} characters)
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("Preview parsed resume text"):
            st.text(text)
    except ValueError as exc:
        st.error(str(exc))
        st.session_state.resume_text = ""

# Close upload-area card
st.markdown("</div>", unsafe_allow_html=True)

resume_text = st.session_state.resume_text

# ---- Tailoring Mode selector ----
TAILORING_MODE_DESCRIPTIONS = {
    "Conservative": (
        "Minimal changes. Keywords are added naturally without altering "
        "sentence structure or style. Best for strong resumes that need "
        "light optimization."
    ),
    "Balanced": (
        "Moderate rewriting. Bullets are rephrased to better match JD "
        "language while staying close to your original meaning. Recommended "
        "for most users."
    ),
    "Aggressive": (
        "Substantial rewriting. Resume is restructured and reworded to "
        "maximize JD alignment. Review carefully before submitting."
    ),
}

tailoring_mode = st.radio(
    "Tailoring Mode",
    options=["Conservative", "Balanced", "Aggressive"],
    index=1,  # Balanced default
    horizontal=True,
    key="tailoring_mode",
)
st.caption(TAILORING_MODE_DESCRIPTIONS[tailoring_mode])
if tailoring_mode == "Aggressive":
    st.warning(
        "Aggressive mode makes significant changes. Always review the "
        "tailored resume carefully before submitting to ensure accuracy."
    )


# ============================================================
# Step 2: Pick a sourcing mode (3 tabs)
# ============================================================
section_header("Step 2: Choose how to find jobs")


def _run(mode: str, **kwargs) -> None:
    """Shared runner — clears prior state and renders the structured progress
    step display (one row per expected step)."""
    st.session_state.agent_result = None
    st.session_state.agent_logs = []
    init_step_state(mode)

    step_box = st.empty()
    step_box.markdown(render_progress_steps(), unsafe_allow_html=True)

    def log(msg: str) -> None:
        st.session_state.agent_logs.append(msg)
        advance_steps(msg)
        step_box.markdown(render_progress_steps(), unsafe_allow_html=True)

    with st.spinner("Agent at work..."):
        try:
            result = run_agent(
                resume_text=resume_text,
                mode=mode,
                log_callback=log,
                tailoring_mode=st.session_state.tailoring_mode,
                **kwargs,
            )
            st.session_state.agent_result = result
        except Exception as exc:
            st.error(f"Agent failed: {exc}")
        finally:
            # Mark every step completed so the indicator settles cleanly.
            for s in EXPECTED_STEPS.get(mode, []):
                st.session_state.agent_step_status[s] = "completed"
            step_box.markdown(render_progress_steps(), unsafe_allow_html=True)


tab_keyword, tab_company, tab_paste = st.tabs(
    ["🔍 Keyword Search (Adzuna)", "🏢 Company Search (Greenhouse)", "📋 Paste JD"]
)

# ---------- Tab 1: Keyword search via Adzuna ----------
with tab_keyword:
    st.subheader("Search by keyword and location")
    kw_keyword = st.text_input(
        "Job keyword", placeholder="e.g. data scientist", key="kw_keyword"
    )
    kw_location = render_location_selector(key_prefix="adzuna")

    kw_disabled = not (resume_text and kw_keyword.strip())
    if not resume_text:
        st.info("Upload a resume in Step 1 to enable the agent.")
    elif not kw_keyword.strip():
        st.info("Enter a job keyword to enable the agent.")
    if st.button(
        "Run Agent — Adzuna",
        type="primary",
        disabled=kw_disabled,
        key="btn_kw",
    ):
        _run("adzuna", keyword=kw_keyword, location=kw_location)

# ---------- Tab 2: Company search via Greenhouse ----------
with tab_company:
    st.subheader("Pull live openings from specific companies")

    company_options = _company_display_options()
    selected_companies = st.multiselect(
        "Select target companies",
        options=list(company_options.keys()),
        placeholder="Type to search companies...",
        key="gh_companies",
    )
    st.caption("Select at least one company to continue.")

    gh_filter = st.text_input(
        "Optional keyword filter (e.g. 'data analyst')",
        placeholder="Leave empty to fetch all openings",
        key="gh_filter",
    )

    gh_location = render_location_selector(key_prefix="greenhouse")
    st.caption(
        "Note: location filtering is applied after fetching listings and "
        "matches on job location text."
    )

    gh_disabled = not (resume_text and selected_companies)
    if not resume_text:
        st.info("Upload a resume in Step 1 to enable the agent.")
    elif not selected_companies:
        st.info("Select at least one company to enable the agent.")
    if st.button(
        "Run Agent — Greenhouse",
        type="primary",
        disabled=gh_disabled,
        key="btn_gh",
    ):
        _run(
            "greenhouse",
            company_names=selected_companies,
            keyword_filter=gh_filter,
            location=gh_location,
        )

# ---------- Tab 3: Paste JD ----------
with tab_paste:
    st.subheader("Already have a JD? Paste it here.")
    pj_title = st.text_input(
        "Job title", placeholder="e.g. Senior Data Scientist", key="pj_title"
    )
    pj_company = st.text_input(
        "Company name", placeholder="e.g. Acme Corp", key="pj_company"
    )
    pj_jd = st.text_area(
        "Paste the full job description",
        height=260,
        placeholder="Paste the job posting text here...",
        key="pj_jd",
    )
    pj_disabled = not (resume_text and pj_jd.strip())
    if not resume_text:
        st.info("Upload a resume in Step 1 to enable the agent.")
    elif not pj_jd.strip():
        st.info("Paste a job description to enable the agent.")
    if st.button(
        "Run Agent — Paste JD",
        type="primary",
        disabled=pj_disabled,
        key="btn_pj",
    ):
        _run(
            "paste",
            jd_text=pj_jd,
            job_title=pj_title or "Unknown",
            company_name=pj_company or "Unknown",
        )


# ============================================================
# Results (shared across all tabs)
# ============================================================
result = st.session_state.agent_result
if result and result.get("tailored_resumes"):
    section_header("Top Matches")
    mode_used = result.get("tailoring_mode") or st.session_state.get(
        "tailoring_mode", "Balanced"
    )
    st.caption(f"Tailoring mode: {mode_used}")
    original = resume_text
    match_analyses = result.get("match_analyses") or []
    jd_insights = result.get("jd_insights") or []
    change_tracking = result.get("change_tracking") or []

    for i, entry in enumerate(result["tailored_resumes"], 1):
        job = entry["job"]
        tailored = entry["tailored"]
        title = job.get("title", "(untitled)")
        company = job.get("company", "")
        loc = job.get("location", "")
        analysis = match_analyses[i - 1] if i - 1 < len(match_analyses) else None
        insights = jd_insights[i - 1] if i - 1 < len(jd_insights) else None
        changes = change_tracking[i - 1] if i - 1 < len(change_tracking) else []

        # Pull match-analysis fields up front so the top summary bar can use
        # them, and the Strong/Weak/Why row below can reuse them.
        if analysis:
            try:
                score = int(analysis.get("match_score", 0) or 0)
            except (TypeError, ValueError):
                score = 0
            confidence = analysis.get("confidence", "Low") or "Low"
            confidence_reason = analysis.get("confidence_reason", "") or ""
            strong_matches = analysis.get("strong_matches") or []
            weak_areas = analysis.get("weak_areas") or []
            why_selected = analysis.get("why_selected", "") or ""
        else:
            score = 0
            confidence = "Low"
            confidence_reason = ""
            strong_matches = []
            weak_areas = []
            why_selected = ""

        with st.expander(f"{i}. {title} — {company} ({loc})", expanded=(i == 1)):
            # ---------- Top summary bar + progress bar ----------
            if analysis:
                st.markdown(
                    render_match_summary_bar(score, confidence),
                    unsafe_allow_html=True,
                )
                st.markdown(
                    render_match_progress_bar(score),
                    unsafe_allow_html=True,
                )
                if confidence_reason:
                    st.caption(f"_{confidence_reason}_")
            # ---------- End summary bar ----------

            if job.get("url"):
                st.markdown(f"**[View job posting]({job['url']})**")

            salary_min = job.get("salary_min")
            salary_max = job.get("salary_max")
            if salary_min or salary_max:
                st.markdown(
                    f"**Salary:** {salary_min or '—'} – {salary_max or '—'}"
                )

            # ---------- JD Keywords + Visa Sponsorship ----------
            if insights:
                skill_kw = insights.get("skill_keywords") or []
                attr_kw = insights.get("attribute_keywords") or []
                visa = insights.get("visa_sponsorship", "Unknown") or "Unknown"
                visa_note = insights.get("visa_note", "") or ""

                kw_col, visa_col = st.columns([3, 1])
                with kw_col:
                    st.markdown("**JD Keywords**")
                    if not skill_kw and not attr_kw:
                        st.markdown("_No keywords extracted_")
                    else:
                        tag_html = ""
                        for kw in skill_kw:
                            tag_html += (
                                f'<span style="background-color:#e8f4fd;'
                                f'color:#1a6fa8;padding:2px 8px;'
                                f'border-radius:10px;margin:2px;'
                                f'font-size:0.85em;display:inline-block">'
                                f'{html.escape(str(kw))}</span> '
                            )
                        for attr in attr_kw:
                            tag_html += (
                                f'<span style="background-color:#f0f0f0;'
                                f'color:#555;padding:2px 8px;'
                                f'border-radius:10px;margin:2px;'
                                f'font-size:0.85em;display:inline-block">'
                                f'{html.escape(str(attr))}</span> '
                            )
                        st.markdown(tag_html, unsafe_allow_html=True)
                with visa_col:
                    st.markdown("**Visa Sponsorship**")
                    if visa == "Yes":
                        visa_html = (
                            '<span style="color:#16a34a;font-weight:700;'
                            'font-size:1rem">Yes</span>'
                        )
                    elif visa == "No":
                        visa_html = (
                            '<span style="color:#dc2626;font-weight:700;'
                            'font-size:1rem">No</span>'
                        )
                    else:
                        visa_html = (
                            '<span style="color:#6b7280;font-size:1rem">'
                            'Unknown</span>'
                        )
                    st.markdown(visa_html, unsafe_allow_html=True)
                    if visa_note:
                        st.caption(visa_note)
                st.divider()
            # ---------- End JD Keywords block ----------

            st.markdown("**Job description**")
            st.write(job.get("description", "")[:1500])

            # ---------- Match Analysis details (Strong / Weak / Why) ----------
            # Note: the score+badge+confidence+confidence_reason that used to
            # live here have moved to the top summary bar at the top of this
            # expander, per the new layout.
            if analysis:
                sm_col, wa_col, ws_col = st.columns(3)
                with sm_col:
                    st.markdown("**Strong Matches**")
                    if strong_matches:
                        for item in strong_matches:
                            st.markdown(f"- ✅ {item}")
                    else:
                        st.caption("_None identified._")
                with wa_col:
                    st.markdown("**Weak Areas**")
                    if weak_areas:
                        for item in weak_areas:
                            st.markdown(f"- ⚠️ {item}")
                    else:
                        st.caption("_None identified._")
                with ws_col:
                    st.markdown("**Why Selected**")
                    st.write(why_selected)

                st.caption(
                    "Match analysis is generated by AI based on text "
                    "comparison. Treat scores as directional estimates, not "
                    "objective assessments."
                )
                st.divider()
            # ---------- End Match Analysis ----------

            tab_tailored, tab_changes, tab_diff = st.tabs(
                ["Tailored resume", "Change Tracking", "Diff vs original"]
            )
            with tab_tailored:
                st.text_area(
                    f"Tailored resume #{i}",
                    value=tailored,
                    height=420,
                    key=f"tailored_{i}",
                )
            with tab_changes:
                st.caption(
                    "Changes are classified by AI. 'Potentially exaggerated' "
                    "flags are for your review and are not definitive "
                    "assessments."
                )
                if not changes:
                    st.info("No change tracking data available for this job.")
                else:
                    icon_by_type = {
                        "Added keyword": "🟢 Added keyword",
                        "Rewritten bullet": "🟡 Rewritten bullet",
                        "Potentially exaggerated": "🔴 Potentially exaggerated",
                    }
                    rows = []
                    for change in changes:
                        raw_type = change.get("change_type", "") or ""
                        rows.append(
                            {
                                "Type": icon_by_type.get(raw_type, raw_type or "Other"),
                                "Original": change.get("original") or "—",
                                "Tailored": change.get("tailored") or "",
                                "Explanation": change.get("explanation") or "",
                            }
                        )
                    st.dataframe(
                        rows,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Type": st.column_config.TextColumn("Type", width=180),
                            "Original": st.column_config.TextColumn("Original"),
                            "Tailored": st.column_config.TextColumn("Tailored"),
                            "Explanation": st.column_config.TextColumn("Explanation"),
                        },
                    )
            with tab_diff:
                st.markdown(DIFF_LEGEND_HTML, unsafe_allow_html=True)
                st.markdown(
                    render_side_by_side_diff(original, tailored or ""),
                    unsafe_allow_html=True,
                )

            # Download button — placed below the two tabs, still inside expander.
            if tailored:
                safe_title = re.sub(r"[^\w\-]+", "_", title).strip("_") or "Role"
                safe_company = re.sub(r"[^\w\-]+", "_", company).strip("_") or "Company"
                file_name = f"ResuMatch_{safe_title}_{safe_company}.docx"
                try:
                    docx_bytes = generate_tailored_docx(tailored, job)
                    st.download_button(
                        label="Download Tailored Resume (.docx)",
                        data=docx_bytes,
                        file_name=file_name,
                        mime=(
                            "application/vnd.openxmlformats-officedocument."
                            "wordprocessingml.document"
                        ),
                        key=f"dl_{i}",
                    )
                except Exception as exc:
                    st.warning(f"Could not generate .docx: {exc}")

st.divider()
st.caption(
    "Note: ResuMatch is an assistant, not an autopilot. Always review and "
    "edit the tailored resume yourself before sending it to an employer — "
    "make sure every claim is accurate."
)
