"""Evaluation metrics.

Retrieval (no LLM needed): hit@k, MRR and recall@k at document level.
Answer (no LLM needed): key-fact recall, citation precision, abstention accuracy.
LLM-as-judge (when a model is configured): answer relevancy. Faithfulness comes from the
pipeline's own verify step, and is reported as the share of answered questions that passed it.
"""

from __future__ import annotations

import re
import unicodedata
from statistics import mean


def doc_ranking(chunk_ids: list[str]) -> list[str]:
    seen, out = set(), []
    for cid in chunk_ids:
        d = cid.split("#")[0]
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def hit_at_k(ranked_docs: list[str], gold: list[str], k: int) -> float:
    return float(any(d in gold for d in ranked_docs[:k]))


def reciprocal_rank(ranked_docs: list[str], gold: list[str]) -> float:
    for i, d in enumerate(ranked_docs, 1):
        if d in gold:
            return 1.0 / i
    return 0.0


def recall_at_k(ranked_docs: list[str], gold: list[str], k: int) -> float:
    return len(set(ranked_docs[:k]) & set(gold)) / len(gold) if gold else 0.0


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).lower()


def key_fact_recall(answer: str, facts: list[str]) -> float:
    a = _norm(answer)
    return sum(_norm(f) in a for f in facts) / len(facts) if facts else 1.0


def citation_precision(cited_docs: list[str], gold: list[str]) -> float:
    return sum(d in gold for d in cited_docs) / len(cited_docs) if cited_docs else 0.0


def summarise(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["answerable"]]
    un = [r for r in rows if not r["answerable"]]
    answered = [r for r in ans if not r["abstained"]]

    def avg(key, rs):
        vals = [r[key] for r in rs if r.get(key) is not None]
        return round(mean(vals), 3) if vals else None

    return {
        "n_questions": len(rows),
        "hit@1": avg("hit@1", ans),
        "hit@3": avg("hit@3", ans),
        "mrr": avg("mrr", ans),
        "recall@5": avg("recall@5", ans),
        "answer_rate": round(len(answered) / len(ans), 3) if ans else None,
        "key_fact_recall": avg("key_fact_recall", ans),
        "citation_precision": avg("citation_precision", answered),
        "grounded_rate": avg("grounded", answered),
        "abstention_accuracy": round(mean(r["abstained"] for r in un), 3) if un else None,
        "answer_relevancy_llm": avg("answer_relevancy", answered),
        "mean_latency_ms": avg("latency_ms", rows),
    }
