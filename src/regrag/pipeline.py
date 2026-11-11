"""Public entry point: guardrails -> LangGraph agent -> cited answer -> audit log."""

from __future__ import annotations

import hashlib
import time
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

from regrag.audit import AuditLog
from regrag.config import Settings
from regrag.embeddings import get_embedder
from regrag.graph import build_graph, initial_state
from regrag.guardrails import check_input
from regrag.index import HybridIndex
from regrag.llm import get_llm
from regrag.reasoners import Reasoner, get_reasoner

ABSTAIN_MESSAGE = (
    "I can't answer that from the approved documents. Please check with the document owner "
    "or Quality Assurance."
)


@dataclass
class Citation:
    n: int
    chunk_id: str
    doc_id: str
    title: str
    section: str


@dataclass
class Answer:
    request_id: str
    question: str
    answer: str
    citations: list[Citation]
    grounded: bool
    abstained: bool
    abstain_reason: str | None
    retrieved: list[str]
    redactions: list[str]
    injection_suspected: bool
    latency_ms: int
    trace: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    rejected_draft: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class RegRAG:
    def __init__(
        self,
        index: HybridIndex,
        reasoner: Reasoner,
        audit: AuditLog | None = None,
        top_k: int = 6,
        max_rewrites: int = 1,
    ):
        self.index = index
        self.reasoner = reasoner
        self.audit = audit or AuditLog(None)
        self.graph = build_graph(index, reasoner, top_k=top_k, max_rewrites=max_rewrites)

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> RegRAG:
        s = settings or Settings.from_env()
        index = HybridIndex.load(s.index_dir, get_embedder(s))
        if s.reranker:
            from regrag.rerank import load_reranker

            index.reranker = load_reranker(s.reranker)
        reasoner = get_reasoner(get_llm(s), idf=index.bm25.idf)
        return cls(index, reasoner, AuditLog(s.audit_log), s.top_k, s.max_rewrites)

    def ask(self, question: str, user: str = "anonymous") -> Answer:
        answer = None
        for kind, payload in self.stream(question, user):
            if kind == "answer":
                answer = payload
        return answer

    def stream(self, question: str, user: str = "anonymous") -> Iterator[tuple[str, Any]]:
        """Run the agent, yielding ("step", {...}) after each LangGraph node and finally
        ("answer", Answer). Used by ask() and by the server-sent-events endpoint."""
        t0 = time.perf_counter()
        request_id = uuid.uuid4().hex[:12]
        guard = check_input(question)
        before = (self.reasoner.usage.input_tokens, self.reasoner.usage.output_tokens)

        state: dict = dict(initial_state(guard.text))
        for update in self.graph.stream(state, stream_mode="updates"):
            for node, changes in update.items():
                state.update(changes or {})
                yield (
                    "step",
                    {
                        "request_id": request_id,
                        "node": node,
                        "detail": (state.get("trace") or [""])[-1],
                        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
                    },
                )

        relevant = state.get("relevant") or []
        draft = state.get("draft")
        abstained = bool(state.get("abstain_reason"))
        citations = []
        if not abstained and draft:
            for n in draft.cited:
                c = relevant[n - 1]
                citations.append(Citation(n, c.chunk_id, c.doc_id, c.title, c.section))

        ans = Answer(
            request_id=request_id,
            question=guard.text,
            answer=ABSTAIN_MESSAGE if abstained else draft.text,
            citations=citations,
            grounded=bool(state.get("grounded")),
            abstained=abstained,
            abstain_reason=state.get("abstain_reason"),
            retrieved=[h.chunk.chunk_id for h in state.get("hits", [])],
            redactions=guard.redactions,
            injection_suspected=guard.injection_suspected,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            trace=state.get("trace", []),
            unsupported_claims=state.get("unsupported") or [],
            rejected_draft=draft.text if (abstained and draft and state.get("unsupported")) else None,
        )
        self.audit.write(
            {
                "request_id": request_id,
                "user": hashlib.sha256(user.encode()).hexdigest()[:16],
                "question_redacted": guard.text,
                "redactions": guard.redactions,
                "injection_suspected": guard.injection_suspected,
                "model": self.reasoner.name,
                "embedder": self.index.embedder.name,
                "retrieved": ans.retrieved,
                "cited": [c.chunk_id for c in citations],
                "grounded": ans.grounded,
                "abstained": abstained,
                "abstain_reason": ans.abstain_reason,
                "unsupported_claims": ans.unsupported_claims,
                "rejected_draft": ans.rejected_draft,
                "input_tokens": self.reasoner.usage.input_tokens - before[0],
                "output_tokens": self.reasoner.usage.output_tokens - before[1],
                "latency_ms": ans.latency_ms,
            }
        )
        yield "answer", ans
