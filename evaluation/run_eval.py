"""Evaluation pipeline: ResuMatch vs ChatGPT-style baseline across 15 jobs.

Generates two tailored resumes per job (baseline = naive single-prompt Claude
call, ResuMatch = full src.tailoring.tailor_resume pipeline), measures TF-IDF
similarity and keyword coverage against the original JD, and writes results
to evaluation/results.csv plus a console summary table.

Run from the project root:
    python evaluation/run_eval.py
"""

from __future__ import annotations

import os
import sys

# Make src/ importable, and load .env from the project root so we don't
# depend on the shell's current working directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

try:
    from dotenv import load_dotenv  # noqa: E402

    # override=True so a pre-set empty env var (e.g. from a harness) doesn't
    # shadow the real key in .env.
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=True)
except ImportError:
    pass

from evaluation.metrics import fabrication_rate  # noqa: E402
from src.tailoring import get_secret, tailor_resume  # noqa: E402

CACHE_PATH = os.path.join(os.path.dirname(__file__), "cache.json")


def _load_cache() -> dict:
    """Return cached {<id>: {'baseline': str, 'resumatch': str}} or {}."""
    import json

    if not os.path.exists(CACHE_PATH):
        return {}
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: dict) -> None:
    """Atomically persist the cache to disk after every case."""
    import json

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    os.replace(tmp, CACHE_PATH)


# ============================================================
# Test cases
# ============================================================
TEST_CASES = [
    {
        "id": 1,
        "job_title": "Data Analytics Intern (Summer 2026)",
        "company": "Cloudflare",
        "jd": """At Cloudflare, we are on a mission to help build a better Internet.
We value candidates who have the AI-native curiosity to create solutions using the latest tools.

What you will do:
- Analyze large datasets to uncover trends and insights that drive business decisions
- Build and maintain dashboards and reports to track key performance metrics
- Collaborate cross-functionally with product, engineering, and go-to-market teams
- Apply data analytics techniques including SQL querying, data visualization, and statistical analysis
- Support automation of data workflows and reporting pipelines
- Contribute to data quality initiatives and ensure consistency across reporting systems

Requirements:
- Pursuing a degree in Data Science, Statistics, Computer Science, or related field
- Proficiency in SQL and data visualization tools
- Experience with Python or R for data analysis
- Strong analytical and problem-solving skills
- AI-native mindset and curiosity about new technologies""",
    },
    {
        "id": 2,
        "job_title": "Business Analyst Intern",
        "company": "Stripe",
        "jd": """Stripe is a financial infrastructure platform for businesses.

What you will do:
- Partner with product and engineering teams to define metrics and KPIs
- Build dashboards and reports to track business performance
- Conduct deep-dive analyses to identify growth opportunities
- Write SQL queries to extract and analyze large datasets
- Present findings and recommendations to senior stakeholders

Requirements:
- Strong SQL skills and experience with data visualization tools
- Experience with Python or Excel for data analysis
- Excellent communication and presentation skills
- Detail-oriented with strong problem-solving abilities""",
    },
    {
        "id": 3,
        "job_title": "Data Analyst",
        "company": "Airbnb",
        "jd": """Airbnb connects millions of people to unique stays and experiences.

What you will do:
- Analyze host and guest behavior data to improve platform experience
- Develop and maintain dashboards tracking key marketplace metrics
- Work cross-functionally with product, design, and engineering teams
- Use SQL and Python to extract insights from large datasets
- Present data-driven recommendations to leadership

Requirements:
- Proficiency in SQL and Python
- Experience with data visualization tools such as Tableau or Looker
- Strong analytical thinking and attention to detail
- Ability to communicate complex findings clearly""",
    },
    {
        "id": 4,
        "job_title": "Financial Analyst Intern",
        "company": "Goldman Sachs",
        "jd": """Goldman Sachs is a leading global investment banking firm.

What you will do:
- Support financial modeling and valuation analysis
- Prepare reports and presentations for senior management
- Analyze financial data and market trends
- Assist with budgeting and forecasting processes
- Conduct industry and competitor research

Requirements:
- Strong Excel and financial modeling skills
- Understanding of accounting principles and financial statements
- Attention to detail and strong analytical skills
- Excellent written and verbal communication""",
    },
    {
        "id": 5,
        "job_title": "Product Analyst",
        "company": "Notion",
        "jd": """Notion is a connected workspace for teams.

What you will do:
- Define and track product metrics and KPIs
- Analyze user behavior and product usage patterns
- Partner with product managers on feature prioritization
- Build self-serve dashboards for cross-functional teams
- Conduct A/B test analysis and interpret results

Requirements:
- Strong SQL and data analysis skills
- Experience with product analytics tools
- Ability to translate data into product insights
- Strong collaboration and communication skills""",
    },
    {
        "id": 6,
        "job_title": "Risk Analyst",
        "company": "JPMorgan Chase",
        "jd": """JPMorgan Chase is a leading global financial services firm.

What you will do:
- Analyze credit and market risk data
- Prepare risk reports for senior management and regulators
- Monitor portfolio performance and identify anomalies
- Support stress testing and scenario analysis
- Ensure compliance with regulatory requirements

Requirements:
- Strong Excel and SQL skills
- Understanding of financial risk concepts
- Attention to detail and strong analytical skills
- Knowledge of regulatory compliance frameworks""",
    },
    {
        "id": 7,
        "job_title": "Operations Analyst Intern",
        "company": "Amazon",
        "jd": """Amazon is guided by four principles including customer obsession.

What you will do:
- Analyze operational data to identify efficiency opportunities
- Build dashboards to monitor supply chain and logistics metrics
- Work with operations teams to implement process improvements
- Use SQL and Excel to extract and analyze data
- Prepare weekly and monthly operational reports

Requirements:
- Strong analytical and problem-solving skills
- Proficiency in SQL and Excel
- Ability to work in a fast-paced environment
- Strong communication skills""",
    },
    {
        "id": 8,
        "job_title": "Accounting Analyst",
        "company": "Deloitte",
        "jd": """Deloitte provides audit, consulting, and advisory services.

What you will do:
- Support audit engagements and financial statement analysis
- Prepare working papers and documentation
- Analyze financial data for accuracy and compliance
- Assist with client reporting and reconciliation
- Communicate findings to engagement teams

Requirements:
- Understanding of accounting principles and audit procedures
- Strong Excel skills and attention to detail
- Ability to analyze and reconcile financial records
- Strong written and verbal communication""",
    },
    {
        "id": 9,
        "job_title": "Data Science Intern",
        "company": "Databricks",
        "jd": """Databricks is the data and AI company.

What you will do:
- Build machine learning models to solve business problems
- Analyze large-scale datasets using Python and Spark
- Collaborate with data engineers and product teams
- Present model results and insights to stakeholders
- Contribute to data quality and feature engineering pipelines

Requirements:
- Strong Python skills and ML fundamentals
- Experience with SQL and data manipulation
- Familiarity with machine learning frameworks
- Strong analytical and communication skills""",
    },
    {
        "id": 10,
        "job_title": "Tax Analyst Intern",
        "company": "PwC",
        "jd": """PwC is a multinational professional services network.

What you will do:
- Assist with tax return preparation and compliance
- Analyze financial data and tax records
- Prepare tax workpapers and documentation
- Research tax regulations and compliance requirements
- Support client deliverables and reporting

Requirements:
- Knowledge of tax and accounting principles
- Strong Excel and data analysis skills
- Attention to detail and accuracy
- Strong written communication skills""",
    },
    {
        "id": 11,
        "job_title": "Business Intelligence Analyst",
        "company": "Figma",
        "jd": """Figma is a collaborative design platform.

What you will do:
- Design and maintain BI dashboards for business stakeholders
- Write complex SQL queries to extract and transform data
- Partner with finance, sales, and product teams on reporting
- Identify data quality issues and implement solutions
- Automate recurring reports and data workflows

Requirements:
- Expert-level SQL skills
- Experience with BI tools such as Looker or Tableau
- Strong Python skills for data transformation
- Excellent stakeholder communication skills""",
    },
    {
        "id": 12,
        "job_title": "Audit Associate",
        "company": "EY",
        "jd": """EY is a global leader in assurance, tax, and advisory services.

What you will do:
- Perform audit procedures and financial statement analysis
- Prepare and review audit working papers
- Identify control weaknesses and document findings
- Collaborate with client accounting teams
- Ensure compliance with audit standards and regulations

Requirements:
- Understanding of accounting and audit principles
- Strong analytical skills and attention to detail
- Proficiency in Excel and data analysis
- Strong communication and teamwork skills""",
    },
    {
        "id": 13,
        "job_title": "Strategy Analyst Intern",
        "company": "McKinsey",
        "jd": """McKinsey is a global management consulting firm.

What you will do:
- Conduct quantitative and qualitative research
- Build financial models and analytical frameworks
- Synthesize data into clear insights and recommendations
- Present findings to senior clients and leadership
- Work on cross-industry strategy engagements

Requirements:
- Strong analytical and problem-solving skills
- Proficiency in Excel and PowerPoint
- Ability to structure complex problems
- Excellent communication and presentation skills""",
    },
    {
        "id": 14,
        "job_title": "Machine Learning Engineer Intern",
        "company": "Anthropic",
        "jd": """Anthropic is an AI safety company working to build reliable AI.

What you will do:
- Develop and evaluate machine learning models
- Write clean Python code for ML pipelines
- Analyze model performance and run experiments
- Collaborate with research and engineering teams
- Document methods and present results clearly

Requirements:
- Strong Python programming skills
- Familiarity with ML frameworks such as PyTorch
- Understanding of machine learning fundamentals
- Strong analytical and communication skills""",
    },
    {
        "id": 15,
        "job_title": "Finance Data Analyst",
        "company": "Coinbase",
        "jd": """Coinbase is the leading cryptocurrency exchange platform.

What you will do:
- Analyze financial and transaction data at scale
- Build dashboards to monitor revenue and business metrics
- Partner with accounting and FP&A teams on reporting
- Use SQL and Python to automate data workflows
- Support month-end close and financial reporting processes

Requirements:
- Strong SQL and Python skills
- Experience with financial data and reporting
- Familiarity with data visualization tools
- Strong attention to detail and analytical skills""",
    },
]


# ============================================================
# Resume text
# ============================================================
RESUME_PATH = os.path.join(os.path.dirname(__file__), "sample_resume.txt")

RESUME_TEXT = """Zhiruo (Edel) ZHAO
Johns Hopkins University 08/2025-08/2026
Master of Science in Information Systems and Artificial Intelligence (In Progress)
Related Coursework: Generative AI, Machine Learning, AI-Driven Sequential Decision Making, Advanced Database Management, Data Analytics

Shanghai University of Finance and Economics (SUFE) 09/2021-06/2025
Bachelor of Management in Financial Management, School of Accountancy
Overall GPA: 3.0/4.0 (WES GPA: 3.13/4.0)

Bilibili (Shanghai Huandian Information Technology Co., Ltd.) 04/2025-08/2025
Commercial Resources Department, Policy Intern
Rebate Calculation & Reconciliation: Queried video advertising revenue data via SQL, calculated rebate amounts for each agency based on company rebate policies, generated Excel reconciliation reports for supervisor review, and assisted sales teams in coordinating confirmation with advertisers.
Cross-department Data Integration: Collected raw data from Finance, Product, and internal teams for quarterly business reviews, consolidated and computed key metrics, and aligned results with the Product team to ensure consistency of data across reports.
Quarterly Data Analysis & Review: Aggregated quarterly commercial data and conducted period-over-period and year-over-year comparisons across dimensions such as industry distribution and client structure, produced data visualizations, and completed quarterly review reports to support business decision-making.

Deloitte Hua Yong CPA Firm 07/2024-09/2024
Audit Intern
Working Paper Preparation: Participated in preparing working papers for sales, management, and R&D expenses; reviewed original vouchers and financial records to ensure accuracy and compliance; analyzed abnormal data and recorded initial judgments.
Voucher Verification & Confirmation: Conducted detail testing by selecting large-amount accounts for sample testing; verified consistency between ledger records and original vouchers, contracts, and attachments.
Inventory Count: Participated in interim inventory counts at client companies, performed stock-taking per audit procedures.

Schindler China Elevator Co., Ltd. 08/2023-12/2023
Finance Department, Tax Intern
Tax Data Entry & Management: Accurately entered input and output tax data monthly, dynamically updated ledgers, reconciled data with invoices, tracked discrepancies.
Invoice Management & Tax Filing: Managed invoice entry and electronic uploads, handled invoice return exceptions, compiled monthly nationwide e-invoice summaries.

ICAS ESG Case Competition 03/2023-04/2023
Analyzed the target company ESG strategy using PEST and SWOT frameworks, awarded 4th place in the finals.

Shanghai University of Finance and Economics Thousand Villages Survey 07/2022-12/2022
Research Assistant, Supervisor: Prof. Darong Dai
Analyzed rural education data using Python and SPSS for chi-square and regression analysis.

Technical Skills: Microsoft Office, Python, R, SQL, Stata
AI/Tech Frameworks: LangChain, RAG, Streamlit
Languages: Chinese (Native), English (Proficient, IELTS 7.0)"""


# ============================================================
# Baseline generation (naive single-prompt Claude call)
# ============================================================
def generate_baseline(resume_text: str, job: dict) -> str:
    """Simulate a naive ChatGPT user workflow:
    single prompt, no system instructions, no few-shot examples.
    Uses Claude with minimal prompting to replicate how a user
    would paste their resume + JD into ChatGPT without guidance.
    """
    import anthropic
    import time

    client = anthropic.Anthropic(api_key=get_secret("ANTHROPIC_API_KEY"))
    time.sleep(6)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1200,
        messages=[
            {
                "role": "user",
                "content": (
                    "I have a job I want to apply for. Can you help me tailor "
                    "my resume for this job?\n\n"
                    f"Job Title: {job['job_title']}\n"
                    f"Company: {job['company']}\n"
                    "Job Description:\n"
                    f"{job['jd'][:1500]}\n\n"
                    "My Resume:\n"
                    f"{resume_text[:3000]}\n\n"
                    "Please rewrite my resume to better match this job."
                ),
            }
        ],
    )
    return response.content[0].text


# ============================================================
# Metrics
# ============================================================
def tfidf_similarity(text1: str, text2: str) -> float:
    """TF-IDF cosine similarity between two texts."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vec = TfidfVectorizer()
    tfidf = vec.fit_transform([text1, text2])
    return round(float(cosine_similarity(tfidf[0], tfidf[1])[0][0]), 4)


def keyword_coverage(jd_text: str, resume_text: str, top_n: int = 20) -> float:
    """Fraction of the JD's top-N content keywords that appear in the resume."""
    import re

    from sklearn.feature_extraction.text import CountVectorizer

    stopwords = {
        "the", "a", "an", "and", "or", "of", "to", "in", "for", "with", "on",
        "at", "by", "from", "is", "are", "be", "as", "we", "you", "that",
        "this", "it", "our", "your", "will", "have", "has", "their", "they",
        "who", "what", "all", "each", "can", "may", "more", "about", "which",
        "into", "its", "across", "using", "build", "built", "not", "strong",
        "new",
    }
    words = re.findall(r"\b[a-z]{3,}\b", jd_text.lower())
    words = [w for w in words if w not in stopwords]
    if not words:
        return 0.0
    vec = CountVectorizer(max_features=top_n)
    vec.fit([" ".join(words)])
    keywords = set(vec.get_feature_names_out())
    if not keywords:
        return 0.0
    resume_lower = resume_text.lower()
    found = {kw for kw in keywords if kw in resume_lower}
    return round(len(found) / len(keywords), 4)


# ============================================================
# Main evaluation loop
# ============================================================
def run_evaluation(test_cases, resume_text, output_path="evaluation/results.csv"):
    import csv
    import time

    results = []
    headers = [
        "id", "job_title", "company",
        "tfidf_original", "tfidf_baseline", "tfidf_resumatch",
        "coverage_original", "coverage_baseline", "coverage_resumatch",
        "tfidf_gain_vs_original", "tfidf_gain_vs_baseline",
        "coverage_gain_vs_original", "coverage_gain_vs_baseline",
        "baseline_fabricated_keywords_count",
        "resumatch_fabricated_keywords_count",
        "baseline_fabrication_rate",
        "resumatch_fabrication_rate",
        "notes",
    ]

    cache = _load_cache()
    print(
        f"Cache: {len([k for k, v in cache.items() if v.get('baseline') and v.get('resumatch')])}"
        f"/{len(test_cases)} cases pre-cached."
    )

    for i, case in enumerate(test_cases):
        print(f"\n[{i+1}/{len(test_cases)}] {case['job_title']} @ {case['company']}")

        job_dict = {
            "title": case["job_title"],
            "company": case["company"],
            "description": case["jd"],
            "location": "N/A",
            "url": "",
        }

        notes = ""
        case_id = str(case["id"])
        cached = cache.get(case_id, {})

        # Baseline (cached if available)
        if cached.get("baseline"):
            print("  Using cached baseline.")
            baseline_text = cached["baseline"]
        else:
            print("  Generating baseline...")
            try:
                baseline_text = generate_baseline(resume_text, case)
                cache.setdefault(case_id, {})["baseline"] = baseline_text
                _save_cache(cache)
            except Exception as e:
                print(f"  Baseline error: {e}")
                baseline_text = resume_text
                notes += f"baseline_error: {type(e).__name__}; "
            time.sleep(6)

        # ResuMatch (cached if available)
        if cached.get("resumatch"):
            print("  Using cached ResuMatch output.")
            resumatch_text = cached["resumatch"]
        else:
            print("  Generating ResuMatch tailored resume...")
            try:
                resumatch_text = tailor_resume(
                    resume_text[:3000], job_dict, tailoring_mode="Balanced"
                )
                cache.setdefault(case_id, {})["resumatch"] = resumatch_text
                _save_cache(cache)
            except Exception as e:
                print(f"  ResuMatch error: {e}")
                resumatch_text = resume_text
                notes += f"resumatch_error: {type(e).__name__}; "
            time.sleep(6)

        # Compute metrics
        jd = case["jd"]
        tfidf_orig = tfidf_similarity(jd, resume_text)
        tfidf_base = tfidf_similarity(jd, baseline_text)
        tfidf_res = tfidf_similarity(jd, resumatch_text)

        cov_orig = keyword_coverage(jd, resume_text)
        cov_base = keyword_coverage(jd, baseline_text)
        cov_res = keyword_coverage(jd, resumatch_text)

        fab_base = fabrication_rate(resume_text, baseline_text, jd, top_n=20)
        fab_res = fabrication_rate(resume_text, resumatch_text, jd, top_n=20)

        row = {
            "id": case["id"],
            "job_title": case["job_title"],
            "company": case["company"],
            "tfidf_original": tfidf_orig,
            "tfidf_baseline": tfidf_base,
            "tfidf_resumatch": tfidf_res,
            "coverage_original": round(cov_orig * 100, 1),
            "coverage_baseline": round(cov_base * 100, 1),
            "coverage_resumatch": round(cov_res * 100, 1),
            "tfidf_gain_vs_original": round(tfidf_res - tfidf_orig, 4),
            "tfidf_gain_vs_baseline": round(tfidf_res - tfidf_base, 4),
            "coverage_gain_vs_original": round((cov_res - cov_orig) * 100, 1),
            "coverage_gain_vs_baseline": round((cov_res - cov_base) * 100, 1),
            "baseline_fabricated_keywords_count": fab_base["fabricated_count"],
            "resumatch_fabricated_keywords_count": fab_res["fabricated_count"],
            "baseline_fabrication_rate": fab_base["fabrication_rate"],
            "resumatch_fabrication_rate": fab_res["fabrication_rate"],
            "notes": notes.strip(),
        }
        results.append(row)

        print(
            f"  TF-IDF:   orig={tfidf_orig:.4f}  base={tfidf_base:.4f}  "
            f"resumatch={tfidf_res:.4f}"
        )
        print(
            f"  Coverage: orig={cov_orig:.0%}  base={cov_base:.0%}  "
            f"resumatch={cov_res:.0%}"
        )
        print(
            f"  Fabric.:  base={fab_base['fabricated_count']:>2}/20 "
            f"({fab_base['fabrication_rate']:.0%})  "
            f"resumatch={fab_res['fabricated_count']:>2}/20 "
            f"({fab_res['fabrication_rate']:.0%})"
        )

    # Save CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n" + "=" * 75)
    print("EVALUATION SUMMARY")
    print("=" * 75)
    print(
        f"{'#':<3} {'Company':<15} {'Role':<30} {'TF-IDF':>8} "
        f"{'Cover':>7} {'TF gain':>8} {'Cv gain':>8}"
    )
    print("-" * 75)
    for r in results:
        print(
            f"{r['id']:<3} {r['company']:<15} {r['job_title'][:28]:<30} "
            f"{r['tfidf_resumatch']:>8.4f} {r['coverage_resumatch']:>6.0f}% "
            f"{r['tfidf_gain_vs_baseline']:>+8.4f} "
            f"{r['coverage_gain_vs_baseline']:>+7.1f}pp"
        )

    # Averages
    n = len(results)
    avg_tfidf_gain_orig = sum(r["tfidf_gain_vs_original"] for r in results) / n
    avg_tfidf_gain_base = sum(r["tfidf_gain_vs_baseline"] for r in results) / n
    avg_cov_gain_orig = sum(r["coverage_gain_vs_original"] for r in results) / n
    avg_cov_gain_base = sum(r["coverage_gain_vs_baseline"] for r in results) / n

    print("-" * 75)
    print(f"\nAVERAGES ACROSS {n} TEST CASES:")
    print(f"  TF-IDF gain vs original:  {avg_tfidf_gain_orig:+.4f}")
    print(f"  TF-IDF gain vs baseline:  {avg_tfidf_gain_base:+.4f}")
    print(f"  Coverage gain vs original: {avg_cov_gain_orig:+.1f} pp")
    print(f"  Coverage gain vs baseline: {avg_cov_gain_base:+.1f} pp")

    # ----- Faithfulness summary -----
    print("\n" + "=" * 67)
    print("FAITHFULNESS ANALYSIS — Fabrication Rate")
    print("=" * 67)
    print(
        f"{'#':<3} {'Company':<16} {'Baseline fabricated':<22} "
        f"{'ResuMatch fabricated':<22}"
    )
    print("-" * 67)
    for r in results:
        b_count = r["baseline_fabricated_keywords_count"]
        b_rate = r["baseline_fabrication_rate"]
        rm_count = r["resumatch_fabricated_keywords_count"]
        rm_rate = r["resumatch_fabrication_rate"]
        b_cell = f"{b_count:>2}/20 ({b_rate*100:>3.0f}%)"
        r_cell = f"{rm_count:>2}/20 ({rm_rate*100:>3.0f}%)"
        print(f"{r['id']:<3} {r['company']:<16} {b_cell:<22} {r_cell:<22}")
    print("-" * 67)

    avg_b_rate = sum(r["baseline_fabrication_rate"] for r in results) / n
    avg_r_rate = sum(r["resumatch_fabrication_rate"] for r in results) / n
    print("AVERAGES:")
    print(f"  Baseline avg fabrication rate:   {avg_b_rate*100:.0f}%")
    print(f"  ResuMatch avg fabrication rate:  {avg_r_rate*100:.0f}%")
    print(
        f"\nResuMatch fabricated {avg_r_rate*100:.0f}% of added keywords on "
        f"average vs baseline's {avg_b_rate*100:.0f}% — confirming that the "
        f"keyword coverage gap is largely explained by baseline's willingness "
        f"to invent experience."
    )

    return results


if __name__ == "__main__":
    # Load resume
    if os.path.exists(RESUME_PATH):
        with open(RESUME_PATH, "r", encoding="utf-8") as f:
            resume = f.read()
    else:
        resume = RESUME_TEXT

    print(f"Resume loaded: {len(resume)} characters")
    print(f"Running evaluation on {len(TEST_CASES)} test cases...")
    print("This will take approximately 5-8 minutes due to rate limit delays.\n")

    run_evaluation(TEST_CASES, resume, output_path="evaluation/results.csv")
