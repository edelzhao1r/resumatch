# ResuMatch

An AI agent that tailors your resume to matched job listings automatically.

Built with Claude (Anthropic), Adzuna API, and Greenhouse API.

## Live Demo

https://resumatchs.streamlit.app/

## Features

- Upload your resume (PDF or DOCX)
- Search jobs via keyword (Adzuna) or by company (Greenhouse), or paste a JD directly
- AI agent matches your resume to the best job listings
- Get a tailored resume for each matched job
- Match analysis with score, confidence, strong matches, and weak areas
- JD keyword extraction and visa sponsorship detection
- Change tracking showing exactly what was modified
- Side-by-side diff view
- Download tailored resume as .docx

## Tech Stack

- Streamlit — UI
- Anthropic Claude API — agent reasoning, resume tailoring, match analysis
- Adzuna API — job search
- Greenhouse Public API — company job listings
- pdfplumber / python-docx — resume parsing
- python-docx — tailored resume export

## Course Concepts Implemented

- Tool Use / Function Calling (Week 4): Agent registers and calls search_jobs_adzuna() and search_jobs_greenhouse() as tools
- Context Engineering (Week 3): System prompt with few-shot examples, structured JD + resume context injection, tailoring mode instructions

## Evaluation

ResuMatch was evaluated across 15 real job listings spanning Data Analytics,
Finance, Consulting, and Engineering roles. Three versions of the resume were
compared for each job: the original unmodified resume, a naive ChatGPT baseline
(single prompt, no system instructions, no few-shot examples), and ResuMatch output
(Balanced mode).

**Metrics used:**
- TF-IDF cosine similarity between the tailored resume and the JD
- Keyword coverage: percentage of top-20 JD keywords present in the resume
- Fabrication rate: percentage of newly added JD keywords that have no grounding
  in the original resume (measures hallucination risk)

**Results summary:**

| Metric | Original Resume | ChatGPT Baseline | ResuMatch |
|--------|----------------|-----------------|-----------|
| TF-IDF similarity (avg) | baseline | — | **+0.117 vs original** |
| Keyword coverage (avg) | — | 56.3% | **62.7%** |
| Fabrication rate (avg) | 0% | **48%** | **40%** |

**Key findings:**

1. ResuMatch vs original resume: TF-IDF similarity improved by +0.117 and
   keyword coverage improved by +26 percentage points on average across all
   15 test cases, confirming that the agent produces meaningful, JD-aligned rewrites.

2. ResuMatch vs ChatGPT baseline: The naive baseline achieves slightly higher
   keyword coverage (+9pp on average) but does so by fabricating experience —
   48% of the JD keywords it adds have no grounding in the candidate's original
   resume. ResuMatch's fabrication rate is 40%, with improvements of up to 20pp
   in cases such as EY (18% vs 44%), Stripe (45% vs 65%), and Notion (27% vs 50%).

3. The coverage gap between ResuMatch and the baseline is largely explained by
   the baseline's willingness to invent keywords. ResuMatch trades a small coverage
   advantage for meaningfully lower fabrication risk — a deliberate design choice
   given that candidates must defend every line of their resume in interviews.

**Notable cases:**
- Best ResuMatch performance: EY Audit Associate — fabrication rate 18% vs
  baseline 44%, while maintaining strong semantic alignment
- Cases where fabrication rates were comparable: Figma BI Analyst, McKinsey
  Strategy Analyst — both roles require niche skills absent from the resume,
  limiting what honest rewrites can achieve

**Test set:** 15 real job listings from Adzuna and Greenhouse APIs
**Baseline:** Claude simulating a naive ChatGPT user — single prompt, no system
prompt, no few-shot examples
**Evaluation script:** evaluation/run_eval.py (results cached in evaluation/cache.json)
**Full results:** evaluation/results.csv

## Local Setup

1. Clone the repo
2. Install dependencies:
   pip install -r requirements.txt
3. Copy the secrets template:
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
4. Fill in your API keys in .streamlit/secrets.toml
5. Run:
   streamlit run app.py

## Streamlit Cloud Deployment

1. Push this repo to GitHub
2. Go to share.streamlit.io
3. Connect your GitHub repo and select app.py as the entry point
4. Under Settings > Secrets, add:
   ADZUNA_APP_ID = "your_id"
   ADZUNA_APP_KEY = "your_key"
   ANTHROPIC_API_KEY = "your_key"
5. Click Deploy

## Important Notes

- Resumes are processed in-session only and never stored server-side
- Match scores are AI estimates and should be treated as directional guidance
- Always review tailored resumes before submitting applications
- Aggressive tailoring mode makes substantial changes — review carefully

## Project Structure

resumatch/
├── app.py                    # Streamlit UI
├── src/
│   ├── agent.py              # Claude agent + ReAct loop + Tool Use
│   ├── job_search.py         # Adzuna API tool
│   ├── greenhouse_search.py  # Greenhouse API tool
│   ├── resume_parser.py      # PDF and DOCX parsing
│   ├── tailoring.py          # Resume rewriting + match analysis
│   └── docx_generator.py     # Word document export
├── evaluation/
│   └── metrics.py            # Keyword coverage + semantic similarity
├── .streamlit/
│   ├── config.toml           # Theme configuration
│   └── secrets.toml.example  # Secrets template
├── .gitignore
├── requirements.txt
└── README.md
