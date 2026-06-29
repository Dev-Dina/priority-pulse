import re
from typing import Dict


def engineer_features(text: str) -> Dict[str, float]:
    """Extract features from ticket text"""
    text_lower = text.lower()

    return {
        'text_length': len(text),
        'word_count': len(text.split()),
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'caps_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),

        # Urgency keywords
        'has_refund': int(bool(re.search(r'\brefund\b', text_lower))),
        'has_cancel': int(bool(re.search(r'\bcancel', text_lower))),
        'has_delay': int(bool(re.search(r'\bdelay', text_lower))),
        'has_help': int(bool(re.search(r'\bhelp\b', text_lower))),
        'has_broken': int(bool(re.search(r'\bbroken\b', text_lower))),

        # Critical keywords
        'has_stranded': int(bool(re.search(r'\bstranded\b', text_lower))),
        'has_medical': int(bool(re.search(r'\bmedical\b', text_lower))),

        # Profanity
        'profanity_count': len(re.findall(r'\b(fuck|shit|damn|worst|awful|terrible|horrible)\b', text_lower)),

        # Time mentions
        'has_time_mention': int(bool(re.search(r'\b\d+\s?(hours?|hrs?|days?)', text_lower))),
    }
