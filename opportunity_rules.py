import re

OPPORTUNITY_KEYWORDS = [
    "internship",
    "intern",
    "hiring",
    "apply",
    "application",
    "career opportunity",
    "opening",
    "position"
]

PAID_KEYWORDS = [
    "registration fee",
    "enrollment fee",
    "training fee",
    "certificate",
    "pay",
    "₹",
    "$"
]

def detect_opportunity(text):
    text = text.lower()
    return any(word in text for word in OPPORTUNITY_KEYWORDS)

def detect_paid_risk(text):
    text = text.lower()
    if any(word in text for word in PAID_KEYWORDS):
        return "probably_paid"
    return "free_or_unclear"

def extract_links(text):
    links = re.findall(r"https?://\S+", text)
    return list(dict.fromkeys(links)) 

def extract_deadline(text):
    match = re.search(r"(apply by|last date|deadline)[^.\n]*", text.lower())
    return match.group(0) if match else None
