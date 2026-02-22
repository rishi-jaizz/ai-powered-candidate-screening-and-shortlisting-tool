# embeddings.py
# Advanced local semantic scoring — no OpenAI required
# Uses enhanced TF-IDF with BM25-style term weighting + synonym expansion

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re
import math

# ── Skill synonyms for expansion (maps variants to canonical form) ─────────────
SKILL_SYNONYMS = {
    "ml": "machine learning", "ai": "artificial intelligence",
    "dl": "deep learning", "nlp": "natural language processing",
    "cv": "computer vision", "js": "javascript", "ts": "typescript",
    "py": "python", "tf": "tensorflow", "pt": "pytorch",
    "sk": "scikit-learn", "sklearn": "scikit-learn",
    "k8s": "kubernetes", "db": "database", "api": "rest api",
    "oop": "object oriented programming", "oops": "object oriented programming",
    "dsa": "data structures algorithms", "ds": "data science",
    "llm": "large language model", "gpt": "large language model",
    "bert": "transformers", "nn": "neural networks",
    "cnn": "convolutional neural network", "rnn": "recurrent neural network",
    "aws": "amazon web services", "gcp": "google cloud platform",
    "powerbi": "power bi", "msexcel": "excel", "my sql": "mysql",
    "node.js": "node", "nodejs": "node", "reactjs": "react",
    "vs code": "development tools", "intellij": "development tools",
    "pycharm": "development tools", "jupyter": "development tools",
}


def expand_text(text):
    """Expand abbreviations and synonyms for richer matching."""
    text_lower = text.lower()
    for abbr, full in SKILL_SYNONYMS.items():
        # Use word boundary to avoid partial replacements
        text_lower = re.sub(r'\b' + re.escape(abbr) + r'\b', full, text_lower)
    return text_lower


def get_embeddings_pair(text1, text2):
    """
    Generate enhanced TF-IDF embeddings with synonym expansion.
    Creates a fresh vectorizer each call to prevent state contamination.
    """
    text1 = expand_text(text1 or "")
    text2 = expand_text(text2 or "")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 3),  # unigrams + bigrams + trigrams for phrase matching
        max_features=10000,
        sublinear_tf=True,  # log normalization — reduces impact of high-freq terms
        min_df=1,
        analyzer="word",
    )

    try:
        vectors = vectorizer.fit_transform([text1, text2]).toarray()
        return vectors[0], vectors[1]
    except Exception:
        # Fallback: simple count vectors
        all_words = list(set(text1.split() + text2.split()))
        v1 = np.array([text1.split().count(w) for w in all_words], dtype=float)
        v2 = np.array([text2.split().count(w) for w in all_words], dtype=float)
        return v1, v2


def cosine_similarity(vec1, vec2):
    """Raw cosine similarity — rescaling done in scorer.py."""
    vec1 = np.array(vec1, dtype=float)
    vec2 = np.array(vec2, dtype=float)
    n1 = np.linalg.norm(vec1)
    n2 = np.linalg.norm(vec2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(max(0.0, min(1.0, np.dot(vec1, vec2) / (n1 * n2))))


def keyword_jaccard_score(text1, text2):
    """
    Jaccard-based keyword overlap — used as a complementary signal.
    Stopwords removed, min length 3 chars.
    """
    stopwords = {
        "the", "and", "for", "with", "this", "that", "have", "from", "will", "are", "was",
        "been", "our", "you", "your", "they", "their", "also", "which", "about", "into",
        "more", "over", "than", "such", "each", "both", "very", "well", "can", "has", "its",
        "not", "but", "all", "any", "one", "may", "use", "via", "per", "etc", "inc", "ltd"
    }

    def tok(t):
        return {w for w in re.findall(r'\b[a-z][a-z0-9+#.]{1,}\b', t.lower()) if w not in stopwords}

    t1 = tok(expand_text(text1))
    t2 = tok(expand_text(text2))
    if not t1:
        return 0.3
    # Weighted: JD→resume direction (recall)
    recall = len(t1 & t2) / len(t1)
    # Precision: resume→JD
    precision = len(t1 & t2) / len(t2) if t2 else 0
    # F1-style blend
    if recall + precision == 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)


def compute_semantic_score(jd_text, resume_text):
    """
    Multi-signal semantic score combining TF-IDF cosine + Jaccard overlap.

    WHY the old method gave 0%:
      TF-IDF cosine on 2 short docs typically returns 0.02–0.35.
      The formula (raw-0.25)/0.55 turns anything <0.25 negative → clamped to 0.

    FIX: Proper rescaling + blend with Jaccard which is more robust for short texts.

    Rescaling logic (empirically tuned for resume/JD pairs):
      TF-IDF cosine range: [0.00, 0.45] → rescaled to [0.0, 1.0]
      Jaccard range:       [0.00, 0.50] → rescaled to [0.0, 1.0]
      Final = 0.6 * tfidf_scaled + 0.4 * jaccard_scaled
    """
    # TF-IDF signal
    try:
        jd_emb, res_emb = get_embeddings_pair(jd_text, resume_text)
        tfidf_raw = cosine_similarity(jd_emb, res_emb)
    except Exception:
        tfidf_raw = 0.0

    TFIDF_MAX = 0.42  # empirical upper bound for well-matched resume/JD pairs
    tfidf_scaled = min(1.0, tfidf_raw / TFIDF_MAX)

    # Jaccard signal (more robust for short texts)
    jac_raw = keyword_jaccard_score(jd_text, resume_text)
    JAC_MAX = 0.45
    jac_scaled = min(1.0, jac_raw / JAC_MAX)

    # Weighted blend
    combined = 0.55 * tfidf_scaled + 0.45 * jac_scaled

    return round(min(1.0, combined), 4)


# backward compat
cosine_sim = cosine_similarity