# ai_extractor.py
from groq import Groq
import os, json, re
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def extract_resume_data(text):
    prompt = f"""
You are a resume parser. Extract structured data from the resume below.

STRICT RULES:
- Return ONLY valid JSON — no markdown, no explanation, no backticks
- "skills" = flat list of strings (technical + soft skills)
- "experience" = list of objects with keys: position, company, duration, achievements (list of strings)
- "education" = list of objects with keys: degree, institution, year
- If a field is missing, use empty string "" or empty list []
- Do NOT summarise or truncate — extract exactly what is written

JSON structure:
{{
  "name": "",
  "email": "",
  "phone": "",
  "summary": "",
  "skills": ["skill1", "skill2"],
  "experience": [
    {{
      "position": "",
      "company": "",
      "duration": "",
      "achievements": ["", ""]
    }}
  ],
  "education": [
    {{
      "degree": "",
      "institution": "",
      "year": ""
    }}
  ],
  "projects": ""
}}

Resume text:
{text[:6000]}
"""
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=2000,
        )
        content = response.choices[0].message.content.strip()
        # Strip markdown fences
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.MULTILINE)
        content = re.sub(r"\s*```$",          "", content, flags=re.MULTILINE)
        content = content.strip()

        try:
            return json.loads(content)
        except Exception:
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except Exception:
                    pass
    except Exception:
        pass

    return {
        "name": "Unknown", "email": "", "phone": "", "summary": "",
        "skills": [], "experience": [], "education": [], "projects": ""
    }