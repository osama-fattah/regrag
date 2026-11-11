"""Input guardrails: PII redaction before anything is logged or sent to a model, plus
prompt-injection flagging. Deliberately simple and auditable; swap in Presidio or a cloud
PII service for production."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_PII = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("PHONE", re.compile(r"(?<!\w)\+?\d[\d\s().-]{8,}\d\b")),
    ("PATIENT_ID", re.compile(r"\b(?:MRN|patient\s*(?:id|no\.?|number))[\s:#-]*\w+", re.I)),
    ("DOB", re.compile(r"\b(?:dob|date of birth)[\s:]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", re.I)),
]

_INJECTION = re.compile(
    r"ignore (?:all |any |the )?(?:previous|prior|above) instructions|disregard (?:the )?system"
    r"|reveal (?:your|the) (?:system )?prompt|you are now",
    re.I,
)


@dataclass
class GuardResult:
    text: str
    redactions: list[str] = field(default_factory=list)
    injection_suspected: bool = False


def check_input(text: str) -> GuardResult:
    redactions = []
    for label, pattern in _PII:
        if pattern.search(text):
            redactions.append(label)
            text = pattern.sub(f"[{label}]", text)
    return GuardResult(text, redactions, bool(_INJECTION.search(text)))
