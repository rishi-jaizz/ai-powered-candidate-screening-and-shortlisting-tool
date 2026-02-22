"""
scorer.py — Discriminative scoring engine v3

Key improvements for wider score distribution:
1. Skills: exponential penalty for low overlap (steeper curve)
2. Experience: even for intern roles, score based on resume richness
   (projects, internships, certifications) not just flat 85
3. Education: CGPA/GPA bonus + institute quality signals
4. Sigmoid stretch on final score → pushes weak candidates lower, strong higher
5. Domain: unchanged (already highly discriminative)
"""

import re
import math
from backend.embeddings import compute_semantic_score
from backend.domain_scorer import run_domain_analysis

# ── Degree patterns ───────────────────────────────────────────────────────────
DEGREE_LEVELS = {
    r'\bph\.?d\b|\bdoctorate\b':                   100,
    r'\bm\.?tech\b|\bmaster\s+of\s+tech':           90,
    r'\bmsc\b|\bm\.sc\b|\bmaster\s+of\s+sci':       85,
    r'\bmba\b':                                      85,
    r'\bmaster\b|\bm\.?s\b|\bmca\b|\bme\b':         82,
    r'\bb\.?tech\b|\bbachelor\s+of\s+tech':          70,
    r'\bbe\b|\bb\.e\b|\bbachelor\s+of\s+eng':        70,
    r'\bbsc\b|\bb\.sc\b|\bbachelor\s+of\s+sci':      65,
    r'\bbca\b|\bb\.ca\b':                            65,
    r'\bbachelor\b|\bb\.?a\b':                       63,
    r'\bdiploma\b':                                  45,
    r'\bcertificate\b':                              35,
}

SKILL_SYNONYMS = {
    "machine learning":            ["machine learning", "ml"],
    "deep learning":               ["deep learning", "dl"],
    "artificial intelligence":     ["artificial intelligence", "ai"],
    "generative ai":               ["generative ai", "genai", "gen ai"],
    "large language model":        ["large language model", "llm", "llms"],
    "natural language processing": ["natural language processing", "nlp"],
    "transformer":                 ["transformer", "transformers", "bert", "gpt", "t5"],
    "prompt engineering":          ["prompt engineering", "prompting"],
    "rag":                         ["rag", "retrieval augmented generation"],
    "langchain":                   ["langchain", "lang chain"],
    "huggingface":                 ["huggingface", "hugging face"],
    "tensorflow":                  ["tensorflow"],
    "pytorch":                     ["pytorch", "torch"],
    "scikit learn":                ["scikit learn", "scikit-learn", "sklearn"],
    "python":                      ["python"],
    "javascript":                  ["javascript", "js", "nodejs"],
    "java":                        ["java", "spring boot"],
    "sql":                         ["sql", "mysql", "postgresql", "postgres"],
    "linux":                       ["linux", "unix"],
    "docker":                      ["docker"],
    "kubernetes":                  ["kubernetes", "k8s"],
    "aws":                         ["aws", "amazon web services"],
    "gcp":                         ["gcp", "google cloud"],
    "git":                         ["git", "github", "gitlab"],
    "rest api":                    ["rest api", "restful", "rest"],
    "cybersecurity":               ["cybersecurity", "cyber security", "infosec", "appsec"],
    "penetration testing":         ["penetration testing", "pentesting"],
    "vulnerability assessment":    ["vulnerability assessment", "vapt"],
    "agent":                       ["agent", "agents", "agentic"],
    "ci cd":                       ["ci cd", "cicd", "devops", "github actions"],
}

_SYNONYM_PATTERNS = []
for canonical, variants in SKILL_SYNONYMS.items():
    for v in sorted(variants, key=len, reverse=True):
        _SYNONYM_PATTERNS.append((re.compile(r'\b' + re.escape(v) + r'\b'), canonical))

_STOP = {
    "the","and","for","with","this","that","have","from","will","are","was",
    "been","our","you","your","they","their","also","which","about","into",
    "more","than","such","each","both","can","has","its","not","but","all",
    "any","one","may","use","per","etc","a","an","in","on","at","of","to",
    "is","we","be","do","as","by","or","if","it","up","so","no","he","she",
    "role","work","team","help","using","used","strong","good","great",
    "required","preferred","responsible","including","candidate","position",
    "company","join","looking","build","develop","working","manage","ensure",
    "high","level","ability","knowledge","understanding","skill","skills",
    "experience","year","years","month","months",
}


# ═══════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════

def _normalize_text(text):
    text = text.lower()
    for pattern, canonical in _SYNONYM_PATTERNS:
        text = pattern.sub(canonical, text)
    return text


def _keywords(text):
    norm = _normalize_text(text)
    tokens = [t for t in re.findall(r'[a-z][a-z0-9+#\-]{1,}', norm)
              if t not in _STOP and len(t) >= 2 and not t.isdigit()]
    bigrams = {f"{tokens[i]} {tokens[i+1]}"
               for i in range(len(tokens)-1)
               if len(tokens[i]) >= 3 and len(tokens[i+1]) >= 3}
    return set(tokens) | bigrams


def _sigmoid_stretch(x, center=50, k=0.07):
    """
    Sigmoid stretch — pushes scores apart.
    Scores near center (50) stay close to 50.
    Scores far from center get pushed further away.
    k=0.07 gives good spread without extreme compression.
    """
    return round(100 / (1 + math.exp(-k * (x - center))), 1)


# ═══════════════════════════════════════════
# COMPONENT SCORERS
# ═══════════════════════════════════════════

def _extract_experience_years(text):
    patterns = [
        r'(\d+(?:\.\d+)?)\+?\s*years?\s*(?:of\s+)?(?:experience|exp)',
        r'experience[:\s]+(\d+(?:\.\d+)?)\+?\s*years?',
        r'(\d+(?:\.\d+)?)\+?\s*years?\s+(?:in|at|with|of)',
    ]
    years = []
    for pat in patterns:
        for m in re.finditer(pat, text.lower()):
            try:
                y = float(m.group(1))
                if 0 < y <= 45:
                    years.append(y)
            except Exception:
                pass
    return max(years) if years else 0.0


def _jd_required_years(jd_text):
    patterns = [
        r'(\d+)\+?\s*years?\s*(?:of\s+)?(?:experience|exp)',
        r'minimum\s+(\d+)\s*years?',
        r'at\s*least\s*(\d+)\s*years?',
        r'(\d+)\s*-\s*\d+\s*years?',
    ]
    years = []
    for pat in patterns:
        for m in re.finditer(pat, jd_text.lower()):
            try:
                years.append(float(m.group(1)))
            except Exception:
                pass
    return min(years) if years else 0.0


def _resume_richness(text):
    """
    Score resume richness (0-100) based on:
    - Number of projects
    - Number of internships/work experiences
    - Number of certifications
    - Number of technologies mentioned
    - Length of resume (more content = more experience)
    Used to differentiate candidates when years of exp are equal.
    """
    lower = text.lower()

    projects    = len(re.findall(r'\b(project|built|developed|created|implemented|designed)\b', lower))
    internships = len(re.findall(r'\b(intern|internship|worked at|joined|employed)\b', lower))
    certs       = len(re.findall(r'\b(certification|certified|certificate|course|coursera|udemy|nptel)\b', lower))
    achievements = len(re.findall(r'\b(award|achievement|winner|rank|top|gold|silver|hackathon|competition)\b', lower))
    tech_count  = len(_keywords(text))

    # Weighted sum, capped at 100
    richness = min(100.0,
        projects    * 6  +
        internships * 10 +
        certs       * 5  +
        achievements * 8 +
        min(tech_count, 30) * 1.0 +
        min(len(text) / 50, 10)   # length bonus
    )
    return round(richness, 1)


def _skills_score(resume_text, jd_text):
    """
    Exponential scaling: low recall is punished heavily.
    recall=0.10 → ~10, recall=0.25 → ~40, recall=0.40 → ~70, recall=0.60+ → ~95
    This creates wider separation between strong and weak skill matches.
    """
    jd_kw  = _keywords(jd_text)
    res_kw = _keywords(resume_text)
    if not jd_kw:  return 50.0
    if not res_kw: return 5.0

    overlap   = len(jd_kw & res_kw)
    recall    = overlap / len(jd_kw)
    precision = overlap / len(res_kw)

    # Exponential recall penalty: small overlap → very low score
    recall_score = min(100.0, (recall ** 0.7) * 120)

    # Precision bonus (prevents keyword stuffing)
    precision_score = min(100.0, precision * 100)

    # Weighted blend: recall matters more
    score = 0.75 * recall_score + 0.25 * precision_score

    # Length bonus
    length_bonus = min(5.0, len(resume_text) / 800)
    return round(max(5.0, min(100.0, score + length_bonus)), 1)


def _experience_score(resume_text, jd_text):
    """
    For intern/fresher JDs (required=0): score based on RESUME RICHNESS
    (projects, internships, certifications) instead of flat 85 for everyone.
    For senior roles: use year-based ratio as before.
    """
    required = _jd_required_years(jd_text)
    actual   = _extract_experience_years(resume_text)
    lower    = resume_text.lower()
    jd_lower = jd_text.lower()

    # Detect if JD is for intern/fresher
    is_entry_level = any(w in jd_lower for w in [
        "intern", "fresher", "entry", "graduate", "student", "0-1", "0-2", "trainee"
    ]) or required == 0

    if is_entry_level:
        # Score based on resume richness — differentiates between candidates
        richness = _resume_richness(resume_text)
        # Base: 40 (everyone gets something), + richness contribution
        return round(min(95.0, 40 + richness * 0.55), 1)

    # For experienced roles: infer years if not stated
    if actual == 0:
        senior = len(re.findall(r'\b(senior|lead|principal|head|director|architect|vp|chief|founder|manager|cto)\b', lower))
        mid    = len(re.findall(r'\b(engineer|developer|analyst|scientist|consultant|specialist|associate|sde)\b', lower))
        intern = len(re.findall(r'\b(intern|internship|trainee|fresher|graduate)\b', lower))
        if senior >= 1:   actual = 5.0
        elif mid >= 3:    actual = 3.0
        elif mid >= 1:    actual = 1.5
        elif intern >= 1: actual = 0.5
        else:             actual = 1.0

    if required == 0:
        if any(w in jd_lower for w in ["senior","lead","7+","8+","10+"]):   required = 6.0
        elif any(w in jd_lower for w in ["5+","6+"]):                        required = 5.0
        elif any(w in jd_lower for w in ["3+","4+"]):                        required = 3.0
        elif any(w in jd_lower for w in ["junior","1+","2+"]):               required = 1.0
        else:                                                                required = 2.0

    ratio = actual / required
    if ratio >= 2.0:   return 100.0
    elif ratio >= 1.3: return 90.0
    elif ratio >= 1.0: return 80.0
    elif ratio >= 0.7: return 65.0
    elif ratio >= 0.4: return 50.0
    elif ratio >= 0.2: return 35.0
    else:              return 20.0


def _education_score(resume_text, jd_text):
    """
    Degree level + CGPA bonus + institute quality signals.
    """
    text_lower = resume_text.lower()
    best = 0
    for pattern, score in DEGREE_LEVELS.items():
        if re.search(pattern, text_lower):
            best = max(best, score)

    if best == 0:
        if re.search(r'\bdegree\b', text_lower):   best = 60
        elif re.search(r'\bcollege\b|\buniversity\b|\binstitute\b', text_lower): best = 55

    # CGPA/GPA bonus (up to +10 points)
    cgpa_bonus = 0
    cgpa_match = re.search(r'(?:cgpa|gpa|score)[:\s]*(\d+(?:\.\d+)?)\s*(?:/\s*(\d+))?', text_lower)
    if cgpa_match:
        cgpa = float(cgpa_match.group(1))
        scale = float(cgpa_match.group(2)) if cgpa_match.group(2) else (10.0 if cgpa <= 10 else 100.0)
        pct = (cgpa / scale) * 100
        if pct >= 85:   cgpa_bonus = 10
        elif pct >= 75: cgpa_bonus = 6
        elif pct >= 65: cgpa_bonus = 2

    # Premier institute bonus (up to +5)
    institute_bonus = 0
    premier = ["iit", "nit", "iiit", "bits", "iim", "iisc", "vit", "manipal"]
    if any(inst in text_lower for inst in premier):
        institute_bonus = 5

    jd_lower = jd_text.lower()
    required_level = 0
    for pattern, score in DEGREE_LEVELS.items():
        if re.search(pattern, jd_lower):
            required_level = max(required_level, score)

    if required_level == 0:
        if re.search(r'\bphd\b|\bdoctorate\b', jd_lower):            required_level = 95
        elif re.search(r'\bmaster\b|\bmtech\b|\bm\.tech\b', jd_lower): required_level = 82
        else:                                                         required_level = 65

    if best == 0:
        return 35.0

    base = min(100.0, (best / required_level) * 100)
    return round(min(100.0, base + cgpa_bonus + institute_bonus), 1)


# ═══════════════════════════════════════════
# MAIN FUNCTION
# ═══════════════════════════════════════════

def score_candidates(resume_text, jd_text, skill_w, exp_w, edu_w, semantic_w):
    """
    5-factor discriminative scoring:
      Skills     — exponential scaling (punishes low overlap)
      Experience — richness-based for intern roles (not flat 85)
      Education  — degree + CGPA + institute bonus
      Semantic   — TF-IDF + Jaccard similarity
      Domain     — hardcoded core skill set coverage (20% fixed)

    Final = sigmoid_stretch(base_score) to widen distribution.
    """
    total = skill_w + exp_w + edu_w + semantic_w
    if total == 0: total = 1.0
    skill_w /= total; exp_w /= total; edu_w /= total; semantic_w /= total

    skills_score     = _skills_score(resume_text, jd_text)
    experience_score = _experience_score(resume_text, jd_text)
    education_score  = _education_score(resume_text, jd_text)

    try:
        sem_raw = compute_semantic_score(jd_text, resume_text)
        semantic_score = round(min(100.0, max(0.0, (sem_raw - 0.05) / 0.85 * 100)), 1)
    except Exception:
        semantic_score = round(skills_score * 0.85, 1)

    # Domain coverage score
    domain_result = run_domain_analysis(resume_text, jd_text)
    domain_score  = domain_result["domain_score"]

    # Weighted base (80% user weights)
    base_score = (
        skills_score     * skill_w +
        experience_score * exp_w   +
        education_score  * edu_w   +
        semantic_score   * semantic_w
    )

    # Blend base + domain
    blended = base_score * 0.80 + domain_score * 0.20

    # ── Sigmoid stretch: widens score distribution ──
    # Candidates close to 50 stay near 50
    # Strong candidates (>65) get pushed higher
    # Weak candidates (<35) get pushed lower
    final = _sigmoid_stretch(blended, center=50, k=0.07)

    return {
        "skills":          round(skills_score, 1),
        "experience":      round(experience_score, 1),
        "education":       round(education_score, 1),
        "semantic":        round(semantic_score, 1),
        "domain":          round(domain_score, 1),
        "final":           final,
        "_years_found":    _extract_experience_years(resume_text),
        "_years_required": _jd_required_years(jd_text) or 2.0,
        "_domain_result":  domain_result,
    }