import re

SKILL_KEYWORDS = [
    "python", "sql", "excel", "power bi", "tableau", "aws", "azure", "gcp",
    "r", "java", "c++", "javascript", "html", "css",
    "machine learning", "deep learning", "nlp", "data analysis",
    "spark", "hadoop", "kubernetes", "docker", "git", "linux"
]

def extract_skills(text):
    found = []
    text_lower = text.lower()
    for skill in SKILL_KEYWORDS:
        if re.search(rf"\b{re.escape(skill)}\b", text_lower):
            found.append(skill.title())
    return list(set(found))
