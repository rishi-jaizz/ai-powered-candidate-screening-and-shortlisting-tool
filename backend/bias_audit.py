"""
bias_audit.py — Checks for potential bias in screening configuration and results.
"""

import pandas as pd


def run_bias_audit(df: pd.DataFrame,
                   skill_w: float, exp_w: float,
                   edu_w: float, semantic_w: float,
                   threshold: int) -> dict:
    """
    Performs multiple bias checks and returns a report.

    Returns:
        {
            "issues": [list of warning strings],
            "details": str (markdown summary)
        }
    """
    issues = []
    details_parts = []

    total_w = skill_w + exp_w + edu_w + semantic_w
    if total_w == 0:
        return {"issues": ["All weights are zero — no scoring performed."], "details": ""}

    # Normalize
    sw = skill_w / total_w
    ew = exp_w   / total_w
    edw = edu_w  / total_w
    smw = semantic_w / total_w

    # ── Check 1: Education overweighting ─────────────────────────────────────
    if edw > 0.4:
        issues.append(
            f"Education weight ({edw*100:.0f}%) is very high. This may disadvantage "
            f"candidates with equivalent practical experience but lower degrees."
        )

    # ── Check 2: Experience overweighting ────────────────────────────────────
    if ew > 0.6:
        issues.append(
            f"Experience weight ({ew*100:.0f}%) is dominant. Entry-level or career-change "
            f"candidates may be unfairly penalised regardless of their skills."
        )

    # ── Check 3: Skills underweighting ───────────────────────────────────────
    if sw < 0.15 and total_w > 0:
        issues.append(
            f"Skills weight ({sw*100:.0f}%) is very low. Keyword/skill matching is "
            f"often the most direct signal of job fit."
        )

    # ── Check 4: Score distribution analysis ─────────────────────────────────
    if len(df) >= 3:
        scores = df["ATS Score"]
        std    = scores.std()
        mean   = scores.mean()

        if std < 5:
            issues.append(
                f"Score distribution is very narrow (std={std:.1f}). "
                f"All candidates received similar scores — the model may not be discriminating effectively."
            )

        if mean < 30:
            issues.append(
                f"Average ATS score is very low ({mean:.1f}/100). "
                f"Consider reviewing the Job Description — it may be too restrictive or poorly worded."
            )

    # ── Check 5: Education vs Experience gap in results ──────────────────────
    if len(df) > 0:
        avg_edu = df["Education"].mean()
        avg_exp = df["Experience"].mean()
        if avg_edu > avg_exp + 25:
            issues.append(
                f"Education scores ({avg_edu:.1f}) are significantly higher than "
                f"Experience scores ({avg_exp:.1f}) on average. "
                f"This may indicate education credentials are driving rankings over practical skills."
            )

    # ── Check 6: Threshold extreme ───────────────────────────────────────────
    if threshold >= 90:
        issues.append(
            f"Shortlist threshold ({threshold}) is very high. This may exclude qualified "
            f"candidates, especially from non-traditional backgrounds."
        )
    elif threshold <= 20:
        issues.append(
            f"Shortlist threshold ({threshold}) is very low — almost all candidates will be shortlisted, "
            f"defeating the purpose of automated screening."
        )

    # ── Build details markdown ────────────────────────────────────────────────
    details = f"""
### ⚖️ Weight Configuration
| Criterion | Weight |
|-----------|--------|
| 🛠️ Skills | {sw*100:.1f}% |
| 💼 Experience | {ew*100:.1f}% |
| 🎓 Education | {edw*100:.1f}% |
| 🔍 Semantic | {smw*100:.1f}% |

### 📊 Score Distribution
| Metric | Value |
|--------|-------|
| Candidates Evaluated | {len(df)} |
| Shortlisted | {df['Shortlisted'].sum()} |
| Avg ATS Score | {df['ATS Score'].mean():.1f} |
| Score Std Dev | {df['ATS Score'].std():.1f} |
| Min Score | {df['ATS Score'].min():.1f} |
| Max Score | {df['ATS Score'].max():.1f} |

### ℹ️ About This Audit
This audit checks for:
- **Configuration bias** — extreme weight settings that may unfairly favour/penalise groups
- **Distribution bias** — whether scores are too narrow or skewed
- **Criteria imbalance** — education vs experience scoring gaps
- **Threshold issues** — thresholds that are too strict or too lenient

> ⚠️ This tool does **not** check for demographic bias (gender, race, age) as resumes are anonymised.
> For full compliance auditing, consult an HR specialist.
""".strip()

    return {"issues": issues, "details": details}