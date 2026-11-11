"""Small text utilities shared by retrieval, heuristics and evaluation."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+(?:[.%][0-9]+)?%?")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

STOPWORDS = frozenset(
    """a about above after again all am an and any are as at be because been before being below
    between both but by can could did do does doing down during each few for from further had has
    have having he her here hers him his how i if in into is it its itself just me more most my no
    nor not of off on once only or other our ours out over own same she should so some such than
    that the their theirs them then there these they this those through to too under until up very
    was we were what when where which while who whom why will with would you your yours we us
    must may might shall also per within many much long often quickly""".split()
)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _stem(t: str) -> str:
    # Light plural folding ("assessments" -> "assessment"); enough for policy text, no NLTK needed.
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 4 and t.endswith("s") and not t.endswith(("ss", "us", "is")):
        return t[:-1]
    return t


def content_tokens(text: str) -> list[str]:
    return [_stem(t) for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text.replace("\n", " ")) if s.strip()]
