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
