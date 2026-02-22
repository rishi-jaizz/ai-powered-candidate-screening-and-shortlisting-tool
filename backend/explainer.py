"""
explainer.py — Generates accurate, detailed per-candidate explanation.
Uses actual score data + JD context to produce meaningful narrative.
"""

import re


def _bar(score: float, width: int = 20) -> str:
    filled = int(round(score / 100 * width))
    empty  = width - filled
    return "🟦" * filled + "⬜" * empty


def _grade(score: float) -> tuple:
    """Returns (label, emoji) for a score."""
    if score >= 85: return "Excellent", "🌟"
    if score >= 70: return "Good",      "✅"
    if score >= 55: return "Average",   "🟡"
    if score >= 40: return "Below Avg", "⚠️"
    return "Poor", "❌"


def _skill_narrative(score: float, jd_text: str) -> str:
    if score >= 85:
        return ("The resume demonstrates strong alignment with the required skills in the Job Description. "
                "Most key technical competencies, tools, and domain keywords are present.")
    elif score >= 70:
        return ("Good skill coverage — the candidate has the majority of skills listed in the JD. "
                "A few niche tools or technologies may be missing.")
    elif score >= 50:
        return ("Moderate skill match. The candidate covers some core requirements, but several "
                "important JD keywords and technologies appear absent from the resume.")
    elif score >= 30:
        return ("Low skill overlap with the Job Description. Many required skills, tools, or "
                "domain keywords are not reflected in this resume.")
    else:
        return ("Very few relevant skills detected. The resume appears to be from a significantly "
                "different technical domain than what the JD requires.")


def _exp_narrative(score: float, years_found: float, years_required: float) -> str:
    yf  = f"{years_found:.0f}" if years_found else "not explicitly stated"
    yr  = f"{years_required:.0f}" if years_required > 0 else "not specified"

    if years_found == 0:
        inferred = "(inferred from job titles/context in resume)"
    else:
        inferred = ""

    if years_required == 0 or score >= 82:
        return (f"Experience level is appropriate for this role. "
                f"Years detected: **{yf}** {inferred} | Required: **{yr}**.")
    elif score >= 68:
        return (f"Experience is close to the requirement but slightly short. "
                f"Years detected: **{yf}** {inferred} | Required: **{yr}**.")
    elif score >= 52:
        return (f"Experience is below the stated requirement. "
                f"Years detected: **{yf}** {inferred} | Required: **{yr}**. "
                f"Candidate may need further assessment for this seniority level.")
    else:
        return (f"Significant experience gap. "
                f"Years detected: **{yf}** {inferred} | Required: **{yr}**. "
                f"The candidate appears junior relative to this role's demands.")


def _edu_narrative(score: float) -> str:
    if score >= 95:
        return "Education exceeds the JD requirement — postgraduate qualification detected."
    elif score >= 85:
        return "Education level meets the JD requirement well."
    elif score >= 70:
        return "Education qualification is broadly aligned with what the role expects."
    elif score >= 50:
        return "Education appears to partially meet the requirement — degree level may be lower than preferred."
    elif score >= 35:
        return "Limited formal education detected, or qualification is below the JD threshold."
    else:
        return "No formal education qualification clearly detected in the resume."


def _semantic_narrative(score: float) -> str:
    if score >= 75:
        return ("High semantic similarity — the overall content, domain context, and language of "
                "the resume closely mirrors the JD. This candidate is likely from the same field.")
    elif score >= 50:
        return ("Moderate semantic relevance. There is meaningful overlap in domain language, "
                "but some areas of the resume are outside the JD's scope.")
    elif score >= 25:
        return ("Low semantic relevance. While some matching terms exist, the resume's overall "
                "content does not strongly align with the JD's domain and context.")
    else:
        return ("Very low semantic similarity — the resume appears to be from a different field "
                "or domain compared to what the Job Description targets.")


def _overall_verdict(score: float) -> tuple:
    if score >= 85:
        return "🌟 Strong Candidate", "Highly recommended for interview. Excellent match across most criteria."
    elif score >= 70:
        return "✅ Good Candidate", "Recommended for interview. Strong fit with minor gaps."
    elif score >= 60:
        return "🟡 Borderline Candidate", "Worth reviewing further. Meets some criteria but has notable gaps."
    elif score >= 45:
        return "⚠️ Weak Candidate", "Significant gaps in required criteria. May not be suitable for this role."
    else:
        return "❌ Not Recommended", "Resume does not align well with the Job Description requirements."


def _strengths_and_gaps(score: dict) -> tuple:
    """Generate top strengths and gaps from score breakdown."""
    components = {
        "Skills Match":         score.get("skills", 0),
        "Experience Level":     score.get("experience", 0),
        "Education":            score.get("education", 0),
        "Semantic Relevance":   score.get("semantic", 0),
    }
    sorted_c = sorted(components.items(), key=lambda x: x[1], reverse=True)

    strengths = [f"**{k}** ({v}/100)" for k, v in sorted_c if v >= 65]
    gaps      = [f"**{k}** ({v}/100)" for k, v in sorted_c if v < 55]

    return strengths, gaps


def generate_explanation(score: dict, candidate_name: str = "Candidate", jd_text: str = "") -> str:
    s  = score.get("skills", 0)
    e  = score.get("experience", 0)
    ed = score.get("education", 0)
    sm = score.get("semantic", 0)
    f  = score.get("final", 0)
    yf = score.get("_years_found", 0)
    yr = score.get("_years_required", 2.0)

    verdict_title, verdict_detail = _overall_verdict(f)
    strengths, gaps = _strengths_and_gaps(score)

    grade_s,  emoji_s  = _grade(s)
    grade_e,  emoji_e  = _grade(e)
    grade_ed, emoji_ed = _grade(ed)
    grade_sm, emoji_sm = _grade(sm)

    strengths_text = ", ".join(strengths) if strengths else "No standout strengths above threshold"
    gaps_text      = ", ".join(gaps)      if gaps      else "No major gaps detected"

    explanation = f"""
**{verdict_title}** — {verdict_detail}

---

### 💪 Key Strengths
{strengths_text}

### 🔴 Notable Gaps
{gaps_text}

---

### 🛠️ Skills Match — {s}/100 &nbsp; {emoji_s} {grade_s}
{_bar(s)}
{_skill_narrative(s, jd_text)}

### 💼 Experience Level — {e}/100 &nbsp; {emoji_e} {grade_e}
{_bar(e)}
{_exp_narrative(e, yf, yr)}

### 🎓 Education — {ed}/100 &nbsp; {emoji_ed} {grade_ed}
{_bar(ed)}
{_edu_narrative(ed)}

### 🔍 Semantic Relevance — {sm}/100 &nbsp; {emoji_sm} {grade_sm}
{_bar(sm)}
{_semantic_narrative(sm)}

---
*Scoring uses keyword overlap (F1-recall weighted), regex-based experience & degree detection, and TF-IDF + Jaccard semantic similarity — all computed locally without any external API.*
""".strip()

    return explanation


def generate_domain_section(domain_result: dict) -> str:
    """Generate a detailed domain coverage section for the explanation."""
    best     = domain_result.get("best_coverage", {})
    domains  = domain_result.get("detected_domains", [])
    coverages = domain_result.get("coverages", {})

    if not best or best.get("domain") == "general":
        return "*Domain detection: could not identify a specific domain.*"

    label    = best.get("label", "Unknown")
    pct      = best.get("coverage_pct", 0)
    matched  = best.get("matched_list", [])
    missing  = best.get("missing_list", [])
    total    = best.get("total_skills", 0)
    n_match  = best.get("matched_skills", 0)
    qual     = best.get("qualified", False)

    bar = _bar(pct)
    status = "✅ **Qualified**" if qual else "❌ **Not Qualified** (need ≥50%)"

    matched_str = ", ".join(f"`{s}`" for s in matched) if matched else "None"
    missing_str = ", ".join(f"`{s}`" for s in missing[:8]) if missing else "None"
    if len(missing) > 8:
        missing_str += f" *(+{len(missing)-8} more)*"

    lines = [
        f"### 🎯 Domain Coverage — {label}",
        f"{bar}  **{pct}%** ({n_match}/{total} core skills) — {status}",
        f"",
        f"**✅ Skills Present:** {matched_str}",
        f"",
        f"**❌ Key Missing Skills:** {missing_str}",
    ]

    # Show all detected domains if multiple
    if len(domains) > 1 and len(coverages) > 1:
        lines.append("")
        lines.append("**Other detected domains:**")
        for dk, cov in coverages.items():
            if dk != best.get("domain") and dk != "general":
                q = "✅" if cov["qualified"] else "⚠️"
                lines.append(f"- {q} {cov['label']}: {cov['coverage_pct']}% ({cov['matched_skills']}/{cov['total_skills']})")

    return "\n".join(lines)