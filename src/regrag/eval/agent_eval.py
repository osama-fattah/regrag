"""Evaluate the tool-calling agent: does it pick the right tools, cite the right records and
passages, and abstain when it should?

Each gold item names the tool *groups* the question needs ("register" for the CAPA register,
"sops" for any document tool) rather than exact tool names, because ask_sops and search_sops are
both a correct way to consult the documents. Expected sources are CAPA ids or document ids; a
cited chunk id ("qa-014-deviation-capa#003") counts for its document.

All metrics are deterministic string/set comparisons, so the offline planner gives the same
numbers on every run and the eval can gate CI. With a model configured the same harness scores
the LLM planner.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from statistics import mean

from regrag.agent import AgentResult, ToolAgent
from regrag.eval.metrics import key_fact_recall

TOOL_GROUPS = {
    "query_capa_register": "register",
    "ask_sops": "sops",
    "search_sops": "sops",
    "list_documents": "sops",
}

# Date the gold set was written against: overdue answers depend on it.
DEFAULT_AS_OF = date(2026, 11, 12)

DEFAULT_GATES = {"tool_selection_accuracy": 0.9, "source_recall": 0.8, "abstention_accuracy": 1.0}


def tool_groups(result: AgentResult) -> set[str]:
    return {TOOL_GROUPS.get(s.tool, s.tool) for s in result.steps}


def _source_key(source: str) -> str:
    """CAPA ids compare case-insensitively; chunk ids reduce to their document id."""
    return source.upper() if source.lower().startswith("capa-") else source.split("#")[0]


def source_scores(cited: list[str], expected: list[str]) -> tuple[float | None, float | None]:
    """(recall, precision) of cited sources against the expected CAPA / document ids."""
    got = {_source_key(s) for s in cited}
    want = {_source_key(s) for s in expected}
    recall = len(got & want) / len(want) if want else None
    precision = len(got & want) / len(got) if got else None
    return recall, precision


def score(item: dict, result: AgentResult) -> dict:
    used = tool_groups(result)
    expected = set(item["expected_tools"])
    answerable = item.get("answerable", True)
    recall, precision = source_scores(result.sources, item["expected_sources"])
    row = {
        "id": item["id"],
        "question": item["question"],
        "answerable": answerable,
        "expected_tools": sorted(expected),
        "used_tools": sorted(used),
        "tool_selection_ok": used == expected,
        "tool_recall": len(used & expected) / len(expected) if expected else 1.0,
        "n_steps": len(result.steps),
        "abstained": result.abstained,
        "sources": result.sources,
        "dropped_sources": result.dropped_sources,
        "structured_ok": result.structured_ok,
        "answer": result.answer,
        "latency_ms": result.latency_ms,
    }
    if answerable:
        row["source_recall"] = 0.0 if result.abstained else recall
        row["source_precision"] = None if result.abstained else precision
        row["key_fact_recall"] = (
            None
            if not item["key_facts"]
            else 0.0
            if result.abstained
            else key_fact_recall(result.answer, item["key_facts"])
        )
    return row


def summarise(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["answerable"]]
    un = [r for r in rows if not r["answerable"]]

    def avg(key: str, rs: list[dict]) -> float | None:
        vals = [float(r[key]) for r in rs if r.get(key) is not None]
        return round(mean(vals), 3) if vals else None

    return {
        "n_questions": len(rows),
        "tool_selection_accuracy": avg("tool_selection_ok", rows),
        "tool_recall": avg("tool_recall", rows),
        "answer_rate": round(mean(not r["abstained"] for r in ans), 3) if ans else None,
        "source_recall": avg("source_recall", ans),
        "source_precision": avg("source_precision", ans),
        "key_fact_recall": avg("key_fact_recall", ans),
        "abstention_accuracy": round(mean(r["abstained"] for r in un), 3) if un else None,
        "structured_output_rate": avg("structured_ok", rows),
        "dropped_source_rate": round(mean(bool(r["dropped_sources"]) for r in rows), 3) if rows else None,
        "mean_steps": avg("n_steps", rows),
        "mean_latency_ms": avg("latency_ms", rows),
    }


def evaluate_agent(agent: ToolAgent, gold: list[dict]) -> tuple[list[dict], dict]:
    rows = [score(item, agent.run(item["question"])) for item in gold]
    return rows, summarise(rows)


def write_agent_report(rows: list[dict], summary: dict, out_dir: str | Path, label: str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "agent_eval_results.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8"
    )
    lines = [f"# Agent evaluation report ({label})", "", "| Metric | Value |", "| --- | --- |"]
    lines += [f"| {k} | {v} |" for k, v in summary.items() if v is not None]
    review = [
        r
        for r in rows
        if not r["tool_selection_ok"]
        or (
            r["answerable"] and (r["abstained"] or (r["source_precision"] or 0) < 1 or r["source_recall"] < 1)
        )
        or (not r["answerable"] and not r["abstained"])
    ]
    if review:
        lines += ["", "## Questions to review", ""]
        lines += ["| id | question | expected tools | used tools | source recall | source precision |"]
        lines += ["| --- | --- | --- | --- | --- | --- |"]
        for r in review:
            sr, sp = r.get("source_recall"), r.get("source_precision")
            lines.append(
                f"| {r['id']} | {r['question']} | {', '.join(r['expected_tools'])} | "
                f"{', '.join(r['used_tools']) or '-'} | {'-' if sr is None else f'{sr:.2f}'} | "
                f"{'-' if sp is None else f'{sp:.2f}'} |"
            )
    path = out / "agent_eval_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
