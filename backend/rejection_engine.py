def rejection_reasons(candidate, jd):

    reasons = []

    if candidate["skill_coverage"] < 0.4:
        reasons.append("Low skill match with job requirements")

    if candidate["experience_years"] < jd["required_experience"]:
        reasons.append("Insufficient experience level")

    if not candidate["education_match"]:
        reasons.append("Education criteria not met")

    if candidate["semantic_score"] < 0.3:
        reasons.append("Low role relevance")

    return reasons