INJECTION_PATTERNS = [
    "ignore previous instructions",
    "disregard above",
    "system prompt",
    "developer message",
    "you are chatgpt",
]

def looks_like_prompt_injection(user_query: str) -> bool:
    q = user_query.lower()
    return any(p in q for p in INJECTION_PATTERNS)