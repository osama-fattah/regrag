from datetime import date
from pathlib import Path

import pytest

from regrag.agent import AgentResult, Step, ToolAgent, _has_policy_question, _procedure_filter
from regrag.eval.agent_eval import (
    DEFAULT_AS_OF,
    DEFAULT_GATES,
    evaluate_agent,
    score,
    source_scores,
    write_agent_report,
)
from regrag.eval.run import check_gates, load_gold
from regrag.register import CapaRegister
from regrag.tools import Toolbox

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def agent(rag) -> ToolAgent:
    register = CapaRegister.load(ROOT / "data/registers/capa_register.csv")
    return ToolAgent(Toolbox(rag, register), today=DEFAULT_AS_OF)


def _result(tools: list[str], sources: list[str], answer: str = "x", abstained: bool = False) -> AgentResult:
    steps = [Step(t, {}, "") for t in tools]
    return AgentResult("rid", "q", answer, sources, steps, abstained, True)


# ---- scoring -----------------------------------------------------------------------------------
def test_source_scores_match_chunks_to_documents_and_ignore_capa_case():
    recall, precision = source_scores(
        [
            "capa-2026-006",
            "qa-014-deviation-capa#003",
            "qa-014-deviation-capa#004",
            "pv-001-icsr-handling#001",
        ],
        ["CAPA-2026-006", "qa-014-deviation-capa"],
    )
    assert recall == 1.0
    assert precision == pytest.approx(2 / 3)  # three distinct sources, two expected
    assert source_scores([], ["CAPA-2026-001"]) == (0.0, None)


def test_score_treats_any_document_tool_as_the_sops_group():
    item = {
        "id": "t",
        "question": "q",
        "expected_tools": ["sops"],
        "expected_sources": ["qa-020-data-integrity"],
        "key_facts": ["monthly"],
    }
    row = score(
        item, _result(["search_sops", "ask_sops"], ["qa-020-data-integrity#002"], "Reviewed monthly.")
    )
    assert row["tool_selection_ok"] and row["source_recall"] == 1.0 and row["key_fact_recall"] == 1.0

    over = score(item, _result(["query_capa_register", "ask_sops"], ["qa-020-data-integrity#002"]))
    assert not over["tool_selection_ok"] and over["tool_recall"] == 1.0


def test_abstaining_on_an_answerable_question_scores_zero():
    item = {
        "id": "t",
        "question": "q",
        "expected_tools": ["sops"],
        "expected_sources": ["x"],
        "key_facts": ["y"],
    }
    row = score(item, _result(["ask_sops"], [], abstained=True))
    assert row["source_recall"] == 0.0 and row["key_fact_recall"] == 0.0 and row["source_precision"] is None


# ---- planner behaviour the eval set exercises ---------------------------------------------------
def test_policy_question_after_a_record_reference_is_detected():
    assert _has_policy_question("CAPA-2026-007 is an excursion. What happens to frozen biologics?")
    assert not _has_policy_question("What is the status of CAPA-2026-009?")
    assert not _has_policy_question("Which critical CAPAs are overdue?")


def test_procedure_filter_from_topic_or_code():
    assert _procedure_filter("are any pharmacovigilance capas overdue?") == "PV-001"
    assert _procedure_filter("open capas under qa-014") == "QA-014"
    assert _procedure_filter("which critical capas are overdue?") is None


def test_agent_answers_both_halves_of_a_mixed_question(agent):
    r = agent.run(
        "CAPA-2026-007 is a temperature excursion. "
        "What happens to a biologic product that was frozen below 0 °C?"
    )
    assert [s.tool for s in r.steps] == ["query_capa_register", "ask_sops"]
    assert "rejected" in r.answer.lower()
    assert "CAPA-2026-007" in r.sources and any(s.startswith("gdp-003") for s in r.sources)


def test_agent_filters_register_by_procedure_topic(agent):
    r = agent.run("Are any pharmacovigilance CAPAs overdue?")
    assert set(r.sources) == {"CAPA-2026-005", "CAPA-2026-006"}


# ---- end to end: the offline baseline must pass its own gate -----------------------------------
def test_offline_agent_eval_passes_gates(agent, tmp_path):
    rows, summary = evaluate_agent(agent, load_gold(ROOT / "data/eval/agent_gold.jsonl"))
    assert summary["n_questions"] == len(rows) == 20
    assert check_gates(summary, DEFAULT_GATES) == []
    report = write_agent_report(rows, summary, tmp_path, "offline")
    assert report.read_text(encoding="utf-8").startswith("# Agent evaluation report (offline)")
    assert (tmp_path / "agent_eval_results.json").exists()


def test_default_as_of_is_the_gold_set_date():
    assert DEFAULT_AS_OF == date(2026, 11, 12)
