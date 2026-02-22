# config.py

# -------- OPENAI MODELS --------
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"

# -------- DEFAULT WEIGHTS --------
DEFAULT_WEIGHTS = {
    "skills": 0.3,
    "semantic": 0.3,
    "experience": 0.3,
    "education": 0.1
}

# -------- PROTECTED ATTRIBUTES (Bias Audit) --------
PROTECTED_TERMS = [
    "gender",
    "male",
    "female",
    "married",
    "religion",
    "age",
    "nationality"
]

# -------- UI SETTINGS --------
MAX_UPLOAD = 15
APP_TITLE = "AI Candidate Screening & Shortlisting"
