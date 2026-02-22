import pdfplumber
import json

# -------- PDF TEXT EXTRACTION --------
def extract_text(file):

    text = ""

    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""

    return text


# -------- SAFE JSON PARSER (VERY IMPORTANT) --------
def safe_json_parse(raw_text):

    try:
        start = raw_text.find("{")
        end = raw_text.rfind("}") + 1
        cleaned = raw_text[start:end]
        return json.loads(cleaned)

    except:
        return None


# -------- SIMPLE KEYWORD EXTRACTION --------
def get_keywords(jd_text):

    words = jd_text.lower().split()
    return list(set(words))
