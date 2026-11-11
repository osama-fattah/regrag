"""The decisions the agent makes at each step, in two interchangeable strategies.

* LLMReasoner  - uses a chat model (OpenAI / Azure OpenAI / Bedrock) for each step.
* HeuristicReasoner - deterministic, extractive and offline. Used in CI so the pipeline,
  the graph and the evaluation gate can be tested without API keys or cost.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol

from regrag import prompts
from regrag.ingest import Chunk
from regrag.llm import LLM
from regrag.text import content_tokens, sentences

INSUFFICIENT = "INSUFFICIENT_CONTEXT"


@dataclass
class Draft:
    text: str
    cited: list[int]  # 1-based indices into the chunks passed to generate()


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0


class Reasoner(Protocol):
    name: str
    usage: Usage

    def grade(self, question: str, chunks: list[Chunk]) -> list[int]: ...
    def rewrite(self, question: str) -> str: ...
    def generate(self, question: str, chunks: list[Chunk]) -> Draft: ...
    def verify(self, answer: str, chunks: list[Chunk]) -> tuple[bool, list[str]]: ...


def _citations(text: str) -> list[int]:
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})


def _parse_json(text: str, default):
    m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
    if not m:
        return default
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return default


class LLMReasoner:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.name = llm.name
        self.usage = Usage()

    def _ask(self, prompt: str, max_tokens: int = 600) -> str:
        r = self.llm.complete(prompts.SYSTEM, prompt, max_tokens)
        self.usage.input_tokens += r.input_tokens
        self.usage.output_tokens += r.output_tokens
        self.usage.calls += 1
        return r.text.strip()

    def grade(self, question, chunks):
        out = self._ask(
            prompts.GRADE.format(question=question, sources=prompts.format_sources([c.text for c in chunks])),
            100,
        )
        picked = _parse_json(out, [])
        return [i for i in picked if isinstance(i, int) and 1 <= i <= len(chunks)]

    def rewrite(self, question):
        return self._ask(prompts.REWRITE.format(question=question), 60).strip('"')

    def generate(self, question, chunks):
        text = self._ask(
            prompts.GENERATE.format(
                question=question, sources=prompts.format_sources([c.text for c in chunks])
            )
        )
        if INSUFFICIENT in text:
            return Draft(INSUFFICIENT, [])
        return Draft(text, [i for i in _citations(text) if 1 <= i <= len(chunks)])

    def verify(self, answer, chunks):
        out = _parse_json(
            self._ask(
                prompts.VERIFY.format(
                    sources=prompts.format_sources([c.text for c in chunks]), answer=answer
                ),
                300,
            ),
            {"grounded": False, "unsupported_claims": ["verifier returned no JSON"]},
        )
        return bool(out.get("grounded")), list(out.get("unsupported_claims", []))


@dataclass
class HeuristicReasoner:
    """Lexical stand-ins for each LLM step, weighted by corpus IDF so generic words ("company",
    "policy") count for little and words the corpus never uses count for a lot. Extractive, so
    answers are grounded by construction."""

    idf: dict[str, float] = field(default_factory=dict)
    min_overlap: float = 0.5
    name: str = "heuristic-extractive"
    usage: Usage = field(default_factory=Usage)

    def _w(self, t: str) -> float:
        # Unseen terms are the most informative: a question about them is probably out of scope.
        return self.idf.get(t, max(self.idf.values(), default=1.0))

    def _overlap(self, q: set[str], text: str) -> float:
        total = sum(self._w(t) for t in q)
        return sum(self._w(t) for t in q & set(content_tokens(text))) / total if total else 0.0

    def grade(self, question, chunks):
        q = set(content_tokens(question))
        return [i for i, c in enumerate(chunks, 1) if self._overlap(q, c.retrieval_text) >= self.min_overlap]

    def rewrite(self, question):
        return " ".join(content_tokens(question))

    def generate(self, question, chunks):
        q = set(content_tokens(question))
        scored = []
        for i, c in enumerate(chunks, 1):
            for s in sentences(c.text):
                q_hits = sum(self._w(t) for t in q & set(content_tokens(s)))
                if q_hits:
                    scored.append((q_hits / (1 + 0.02 * len(s.split())), i, s))
        if not scored:
            return Draft(INSUFFICIENT, [])
        scored.sort(key=lambda x: -x[0])
        best = [x for x in scored[:2] if x[0] >= scored[0][0] * 0.6]
        text = " ".join(f"{s} [{i}]" for _, i, s in best)
        return Draft(text, sorted({i for _, i, _ in best}))

    def verify(self, answer, chunks):
        ctx = set(content_tokens(" ".join(c.text for c in chunks)))
        unsupported = []
        for s in sentences(re.sub(r"\[\d+\]", "", answer)):
            toks = set(content_tokens(s))
            if toks and len(toks & ctx) / len(toks) < 0.8:
                unsupported.append(s)
        return not unsupported, unsupported


def get_reasoner(llm: LLM | None, idf: dict[str, float] | None = None) -> Reasoner:
    return LLMReasoner(llm) if llm is not None else HeuristicReasoner(idf=idf or {})
