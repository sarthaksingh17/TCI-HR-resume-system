"""
Input sanitisation and PII masking utilities.
Security measures against prompt injection and PII leakage in logs.
"""

import re
import logging

logger = logging.getLogger("hr_agent.security")

_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"ignore\s+(all\s+)?above", re.IGNORECASE),
    re.compile(r"system\s*:", re.IGNORECASE),
    re.compile(r"you\s+are\s+now", re.IGNORECASE),
    re.compile(r"forget\s+(all\s+)?previous", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?previous", re.IGNORECASE),
    re.compile(r"new\s+instructions?\s*:", re.IGNORECASE),
    re.compile(r"\[INST\]", re.IGNORECASE),
    re.compile(r"<\|im_start\|>", re.IGNORECASE),
]


def sanitise_text(text: str, source: str = "input") -> str:
    """Sanitise user text before inserting into LLM prompts.
    Strips control chars, detects injection patterns (logs warning, still processes).
    """
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    for pattern in _INJECTION_PATTERNS:
        match = pattern.search(cleaned)
        if match:
            logger.warning(
                "[SECURITY] Potential prompt injection in %s: '%s'. Processing continues.",
                source, match.group()[:50],
            )
            break
    return cleaned


def mask_pii(name: str) -> str:
    """Mask a name for logging: 'Sarthak Singh' -> 'S****k S***h'."""
    if not name or not name.strip():
        return "***"
    parts = name.strip().split()
    masked = []
    for part in parts:
        if len(part) <= 1:
            masked.append("*")
        elif len(part) == 2:
            masked.append(part[0] + "*")
        else:
            masked.append(part[0] + "*" * (len(part) - 2) + part[-1])
    return " ".join(masked)
