"""Run the gold set through the pipeline, write a report, and optionally fail on regressions."""

from __future__ import annotations

import json
from pathlib import Path

from regrag import prompts
from regrag.eval import metrics as m
from regrag.pipeline import RegRAG
from regrag.reasoners import LLMReasoner, _parse_json

DEFAULT_GATES = {"hit@3": 0.9, "key_fact_recall": 0.7, "abstention_accuracy": 0.66}


def load_gold(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(rag: RegRAG, gold: list[dict]) -> tuple[list[dict], dict]:
    judge = rag.reasoner if isinstance(rag.reasoner, LLMReasoner) else None
    rows = []
    for g in gold:
        a = rag.ask(g["question"], user="eval")
        ranked = m.doc_ranking(a.retrieved)
        answerable = g.get("answerable", True)
        row = {
            "id": g["id"],
            "question": g["question"],
            "answerable": answerable,
            "abstained": a.abstained,
            "abstain_reason": a.abstain_reason,
            "answer": a.answer,
            "rejected_draft": a.rejected_draft,
            "unsupported_claims": a.unsupported_claims,
            "cited_docs": [c.doc_id for c in a.citations],
            "grounded": float(a.grounded),
            "latency_ms": a.latency_ms,
        }
        if answerable:
            row.update(
                {
                    "hit@1": m.hit_at_k(ranked, g["gold_docs"], 1),
                    "hit@3": m.hit_at_k(ranked, g["gold_docs"], 3),
                    "mrr": m.reciprocal_rank(ranked, g["gold_docs"]),
                    "recall@5": m.recall_at_k(ranked, g["gold_docs"], 5),
                    "key_fact_recall": 0.0 if a.abstained else m.key_fact_recall(a.answer, g["key_facts"]),
                    "citation_precision": None
                    if a.abstained
                    else m.citation_precision(row["cited_docs"], g["gold_docs"]),
                }
            )
            if judge and not a.abstained:
                out = judge._ask(prompts.JUDGE_RELEVANCY.format(question=g["question"], answer=a.answer), 30)
                row["answer_relevancy"] = float(_parse_json(out, {"score": 0}).get("score", 0))
        rows.append(row)
    return rows, m.summarise(rows)


def check_gates(summary: dict, gates: dict | None = None) -> list[str]:
    failures = []
    for k, threshold in (gates or DEFAULT_GATES).items():
        v = summary.get(k)
        if v is None or v < threshold:
            failures.append(f"{k} = {v} (minimum {threshold})")
    return failures


def write_report(rows: list[dict], summary: dict, out_dir: str | Path, label: str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_results.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8"
    )
    lines = [f"# Evaluation report ({label})", "", "| Metric | Value |", "| --- | --- |"]
    lines += [f"| {k} | {v} |" for k, v in summary.items() if v is not None]
    misses = [r for r in rows if r["answerable"] and (r["abstained"] or r["key_fact_recall"] < 1)]
    if misses:
        lines += ["", "## Questions to review", "", "| id | question | abstained | key-fact recall |"]
        lines += ["| --- | --- | --- | --- |"]
        lines += [
            f"| {r['id']} | {r['question']} | {r['abstained']} | {r['key_fact_recall']:.2f} |" for r in misses
        ]
    path = out / "eval_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
