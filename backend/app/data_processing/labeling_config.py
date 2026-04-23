"""
Labeling configuration (weak supervision rules)
Separated from Settings to avoid Pydantic issues
"""

URGENCY_KEYWORDS = [
    r"\brefund\b",
    r"\bbroken\b",
    r"\bcancel(?:led)?\b",
    r"\bdelay(?:ed)?\b",
    r"\bdown\b",
    r"\bhelp\b",
    r"\burgent\b",
    r"\bemergency\b",
    r"\basap\b",
    r"\bimmediately\b",
    r"\bmissing\b",
    r"\blost\b",
    r"\bstranded\b",
    r"\bissue\b",
    r"\bproblem\b",
    r"\bwaiting\b",
]

CRITICAL_KEYWORDS = [
    r"\bstranded\b",
    r"\bunsafe\b",
    r"\bmedical\b",
    r"\bpolice\b",
    r"\bsue\b",
    r"\blawsuit\b",
    r"\bunacceptable\b",
    r"\bdisaster\b",
    r"\bfraud\b",
]

PROFANITY_KEYWORDS = [
    r"\bfuck\b",
    r"\bshit\b",
    r"\bdamn\b",
    r"\bcrap\b",
    r"\bhell\b",
    r"\bworst\b",
    r"\bawful\b",
    r"\bterrible\b",
    r"\bhorrible\b",
]

SARCASM_PATTERNS = [
    r"\boh great\b",
    r"\bwonderful\b",
    r"\bfantastic\b",
    r"\bperfect\b",
    r"\bthanks for nothing\b",
    r"\blove this\b",
]

DELAY_PATTERNS = [
    r"\b\d+\s?(hours?|hrs?|days?|minutes?|mins?)\b",
    r"\bwaiting\b",
    r"\bstill waiting\b",
    r"\bdelayed\b",
    r"\bdelay\b",
    r"\bcancelled\b",
    r"\bcanceled\b",
    r"\bmissed\b",
    r"\bmissed flight\b",
    r"\bmissed connection\b",
    r"\bstranded\b",
    r"\blate\b",
    r"\bon hold\b",
]

LABEL_WEIGHTS = {
    "profanity": 3,
    "delay": 3,
    "urgency": 2,
    "sarcasm": 1,
    "exclamation": 1,
    "question": 1,
    "caps": 1,
}

LABEL_THRESHOLDS = {
    "critical": 6,
    "urgent": 3,
}