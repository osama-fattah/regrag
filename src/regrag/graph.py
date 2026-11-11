"""The agent as a LangGraph state machine.

retrieve -> grade --(relevant)--> generate -> verify --(grounded)--> END
              |                       |              \\--(unsupported)--> abstain -> END
              |                       \\--(no answer in context)--> abstain
              \\--(nothing relevant, retries left)--> rewrite -> retrieve
              \\--(nothing relevant, no retries)--> abstain
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from regrag.index import Hit, HybridIndex
from regrag.ingest import Chunk
from regrag.reasoners import INSUFFICIENT, Draft, Reasoner


class State(TypedDict, total=False):
    question: str
    query: str
    attempts: int
    hits: list[Hit]
    relevant: list[Chunk]
    draft: Draft | None
    grounded: bool
    unsupported: list[str]
    abstain_reason: str
    trace: list[str]


def build_graph(index: HybridIndex, reasoner: Reasoner, top_k: int = 6, max_rewrites: int = 1):
    def retrieve(s: State) -> State:
        hits = index.search(s["query"], k=top_k)
        return {"hits": hits, "trace": s["trace"] + [f"retrieve({s['query']!r}) -> {len(hits)}"]}

    def grade(s: State) -> State:
        chunks = [h.chunk for h in s["hits"]]
        keep = reasoner.grade(s["question"], chunks)
        relevant = [chunks[i - 1] for i in keep]
        return {"relevant": relevant, "trace": s["trace"] + [f"grade -> {len(relevant)} relevant"]}

    def after_grade(s: State) -> str:
        if s["relevant"]:
            return "generate"
        return "rewrite" if s["attempts"] < max_rewrites else "abstain"

    def rewrite(s: State) -> State:
        q = reasoner.rewrite(s["question"])
        return {"query": q, "attempts": s["attempts"] + 1, "trace": s["trace"] + [f"rewrite -> {q!r}"]}

    def generate(s: State) -> State:
        draft = reasoner.generate(s["question"], s["relevant"])
        return {"draft": draft, "trace": s["trace"] + [f"generate -> cites {draft.cited}"]}

    def after_generate(s: State) -> str:
        return "abstain" if s["draft"].text == INSUFFICIENT else "verify"

    def verify(s: State) -> State:
        ok, unsupported = reasoner.verify(s["draft"].text, s["relevant"])
        return {
            "grounded": ok,
            "unsupported": unsupported,
            "trace": s["trace"] + [f"verify -> grounded={ok}"],
        }

    def after_verify(s: State) -> str:
        return END if s["grounded"] else "abstain"

    def abstain(s: State) -> State:
        if not s.get("relevant"):
            reason = "no relevant documents found"
        elif s.get("draft") and s["draft"].text == INSUFFICIENT:
            reason = "documents found but they do not answer the question"
        else:
            reason = "draft answer contained claims not supported by the sources"
        return {"grounded": False, "abstain_reason": reason, "trace": s["trace"] + ["abstain"]}

    g = StateGraph(State)
    for name, fn in [
        ("retrieve", retrieve),
        ("grade", grade),
        ("rewrite", rewrite),
        ("generate", generate),
        ("verify", verify),
        ("abstain", abstain),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade, ["generate", "rewrite", "abstain"])
    g.add_edge("rewrite", "retrieve")
    g.add_conditional_edges("generate", after_generate, ["verify", "abstain"])
    g.add_conditional_edges("verify", after_verify, [END, "abstain"])
    g.add_edge("abstain", END)
    return g.compile()


def initial_state(question: str) -> State:
    return {"question": question, "query": question, "attempts": 0, "trace": [], "draft": None}
