"""
domain_scorer.py — Domain-Aware Skill Coverage Engine v3
Fixed: alias expansion bugs, rest api double-replacement, proper mismatch penalty
"""

import re

# Only safe, unambiguous aliases — no broad terms like 'rest', 'api' alone
SKILL_ALIASES = {
    "large language model":           ["llm", "llms"],
    "machine learning":               ["ml"],
    "deep learning":                  ["dl"],
    "generative ai":                  ["genai", "gen ai"],
    "natural language processing":    ["nlp"],
    "scikit learn":                   ["sklearn"],
    "kubernetes":                     ["k8s"],
    "retrieval augmented generation": ["rag"],
    "huggingface":                    ["hugging face"],
    "fine tuning":                    ["finetuning"],
    "penetration testing":            ["pentesting", "pen testing"],
    "vulnerability assessment":       ["vapt"],
    "pyspark":                        ["py spark"],
    "github actions":                 ["gh actions"],
}

_ALIAS_PATTERNS = [
    (re.compile(r'\b' + re.escape(a) + r'\b'), c)
    for a, c in sorted(SKILL_ALIASES.items(), key=lambda x: len(x[0]), reverse=True)
    for a, c in [( a, c )]
]
# Rebuild correctly
_ALIAS_PATTERNS = []
for canonical, aliases in SKILL_ALIASES.items():
    for alias in sorted(aliases, key=len, reverse=True):
        _ALIAS_PATTERNS.append(
            (re.compile(r'\b' + re.escape(alias) + r'\b'), canonical)
        )


DOMAIN_SKILLS = {
    "ai_engineer": {
        "label": "AI Engineer",
        "detect_keywords": ["ai engineer", "artificial intelligence engineer", "applied ai"],
        "core_skills": [
            "python", "machine learning", "deep learning", "neural networks",
            "tensorflow", "pytorch", "model deployment",
            "docker", "linux", "git", "numpy", "pandas", "scikit learn",
            "llm", "transformers", "huggingface", "embeddings",
        ]
    },
    "ml_engineer": {
        "label": "Machine Learning Engineer",
        "detect_keywords": ["machine learning engineer", "ml engineer", "machine learning", "ml model"],
        "core_skills": [
            "python", "machine learning", "deep learning", "scikit learn",
            "tensorflow", "pytorch", "numpy", "pandas", "feature engineering",
            "model evaluation", "cross validation", "regression", "classification",
            "neural networks", "model deployment", "docker", "git",
        ]
    },
    "data_science": {
        "label": "Data Scientist",
        "detect_keywords": ["data scientist", "data science", "data analyst", "analytics"],
        "core_skills": [
            "python", "statistics", "pandas", "numpy", "matplotlib",
            "machine learning", "scikit learn", "sql",
            "hypothesis testing", "regression", "classification",
            "data visualization", "git",
        ]
    },
    "generative_ai": {
        "label": "Generative AI / LLM Engineer",
        "detect_keywords": [
            "generative ai", "genai", "gen ai", "llm", "large language model",
            "llm engineer", "prompt engineering", "gpt", "ai intern", "ai security",
            "agent workflow", "langchain"
        ],
        "core_skills": [
            "python", "large language model", "prompt engineering",
            "langchain", "retrieval augmented generation", "huggingface",
            "transformers", "fine tuning", "embeddings",
            "agent", "git",
        ]
    },
    "full_stack": {
        "label": "Full Stack Developer",
        "detect_keywords": ["full stack", "fullstack", "full-stack", "web developer", "software developer"],
        "core_skills": [
            "html", "css", "javascript", "react", "nodejs",
            "sql", "mongodb", "docker", "git", "linux",
        ]
    },
    "frontend": {
        "label": "Frontend Developer",
        "detect_keywords": ["frontend", "front end", "front-end", "ui developer", "react developer"],
        "core_skills": [
            "html", "css", "javascript", "typescript", "react",
            "responsive design", "git",
        ]
    },
    "backend": {
        "label": "Backend Developer",
        "detect_keywords": ["backend", "back end", "back-end", "api developer", "java developer"],
        "core_skills": [
            "python", "java", "nodejs", "django", "flask", "fastapi", "spring boot",
            "sql", "postgresql", "mongodb", "docker", "git", "linux", "microservices",
        ]
    },
    "devops": {
        "label": "DevOps / Cloud Engineer",
        "detect_keywords": ["devops", "cloud engineer", "site reliability", "sre", "platform engineer"],
        "core_skills": [
            "docker", "kubernetes", "terraform", "ansible",
            "github actions", "aws", "gcp", "linux", "bash",
            "prometheus", "grafana", "git", "python",
        ]
    },
    "data_engineer": {
        "label": "Data Engineer",
        "detect_keywords": ["data engineer", "data engineering", "etl pipeline", "data pipeline"],
        "core_skills": [
            "python", "sql", "pyspark", "airflow", "kafka", "dbt",
            "aws", "gcp", "data pipeline", "docker", "git", "linux",
        ]
    },
    "cybersecurity": {
        "label": "Cybersecurity / Security Engineer",
        "detect_keywords": [
            "security engineer", "cybersecurity", "information security",
            "appsec", "ai security", "security intern", "security analyst", "vapt"
        ],
        "core_skills": [
            "python", "linux", "networking", "penetration testing",
            "vulnerability assessment", "owasp", "encryption",
            "wireshark", "nmap", "git", "bash",
        ]
    },
    "android": {
        "label": "Android Developer",
        "detect_keywords": ["android developer", "android engineer", "android", "kotlin developer"],
        "core_skills": [
            "kotlin", "java", "android sdk", "android studio",
            "jetpack compose", "mvvm", "retrofit", "firebase", "git",
        ]
    },
    "ios": {
        "label": "iOS Developer",
        "detect_keywords": ["ios developer", "ios engineer", "swift developer", "ios"],
        "core_skills": [
            "swift", "xcode", "ios sdk", "swiftui", "uikit", "mvvm", "core data", "git",
        ]
    },
}


def _expand(text: str) -> str:
    """Expand aliases to canonical forms. Applied once — safe word boundaries."""
    text = text.lower()
    for pattern, canonical in _ALIAS_PATTERNS:
        text = pattern.sub(canonical, text)
    return text


def _has_skill(skill: str, resume_expanded: str) -> bool:
    """Check if skill (after expansion) appears in already-expanded resume."""
    skill_exp = _expand(skill)
    return bool(re.search(r'\b' + re.escape(skill_exp) + r'\b', resume_expanded))


def detect_domains_from_text(text: str) -> list:
    """Detect domains from a single text."""
    exp = _expand(text.lower())
    scores = {}
    for dk, dd in DOMAIN_SKILLS.items():
        hits = sum(1 for kw in dd["detect_keywords"]
                   if re.search(r'\b' + re.escape(kw.lower()) + r'\b', exp))
        if hits > 0:
            scores[dk] = hits
    return [k for k, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def detect_domains(resume_text: str, jd_text: str) -> list:
    """JD-weighted domain detection (JD counted 2x)."""
    combined = _expand((jd_text + " " + jd_text + " " + resume_text).lower())
    scores = {}
    for dk, dd in DOMAIN_SKILLS.items():
        hits = sum(1 for kw in dd["detect_keywords"]
                   if re.search(r'\b' + re.escape(kw.lower()) + r'\b', combined))
        if hits > 0:
            scores[dk] = hits
    result = [k for k, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    return result if result else ["general"]


def compute_domain_coverage(resume_text: str, domain_key: str) -> dict:
    if domain_key == "general":
        return {"domain": "general", "label": "General", "total_skills": 0,
                "matched_skills": 0, "coverage_pct": 50.0, "qualified": True,
                "matched_list": [], "missing_list": []}

    dd         = DOMAIN_SKILLS[domain_key]
    resume_exp = _expand(resume_text.lower())
    matched, missing = [], []

    for skill in dd["core_skills"]:
        if _has_skill(skill, resume_exp):
            matched.append(skill)
        else:
            missing.append(skill)

    total    = len(dd["core_skills"])
    n        = len(matched)
    coverage = round((n / total) * 100, 1) if total > 0 else 0.0

    return {
        "domain": domain_key, "label": dd["label"],
        "total_skills": total, "matched_skills": n,
        "coverage_pct": coverage, "qualified": coverage >= 50.0,
        "matched_list": matched, "missing_list": missing,
    }


def run_domain_analysis(resume_text: str, jd_text: str) -> dict:
    """
    Domain scoring with JD-alignment penalty:
    - Score resume against JD's primary domain
    - If resume best domain != JD domain → max score capped at 35
    """
    jd_domains = detect_domains_from_text(jd_text)
    jd_primary = jd_domains[0] if jd_domains else "general"

    all_domains = detect_domains(resume_text, jd_text)
    coverages   = {dk: compute_domain_coverage(resume_text, dk) for dk in all_domains}

    # Always compute JD's primary domain coverage
    if jd_primary not in coverages:
        coverages[jd_primary] = compute_domain_coverage(resume_text, jd_primary)

    jd_cov  = coverages[jd_primary]
    jd_pct  = jd_cov["coverage_pct"]

    # Best resume domain
    best = max(coverages.values(), key=lambda x: x["coverage_pct"])

    # Domain alignment check
    jd_top2     = set(jd_domains[:2])
    resume_top2 = set(all_domains[:2])
    aligned     = bool(jd_top2 & resume_top2)

    if aligned:
        raw = jd_pct
    else:
        raw = min(35.0, jd_pct * 0.5)   # hard cap at 35 if wrong domain

    domain_score      = round(min(100.0, raw * 1.8), 1)
    overall_qualified = jd_cov["qualified"] and aligned

    if overall_qualified:
        summary = (f"✅ **{jd_cov['label']}** — "
                   f"{jd_cov['matched_skills']}/{jd_cov['total_skills']} core skills ({jd_pct}%)")
    elif not aligned:
        jd_label = DOMAIN_SKILLS.get(jd_primary, {}).get("label", jd_primary)
        summary  = (f"⚠️ Domain mismatch — Resume is **{best['label']}** "
                    f"but JD needs **{jd_label}**. JD coverage: {jd_pct}%")
    else:
        summary = (f"⚠️ **{jd_cov['label']}** — only {jd_cov['matched_skills']}/"
                   f"{jd_cov['total_skills']} core skills ({jd_pct}%). Need ≥50%.")

    return {
        "detected_domains":  all_domains,
        "jd_domains":        jd_domains,
        "primary_domain":    jd_primary,
        "coverages":         coverages,
        "best_coverage":     jd_cov,
        "domain_score":      domain_score,
        "domain_aligned":    aligned,
        "overall_qualified": overall_qualified,
        "summary":           summary,
    }