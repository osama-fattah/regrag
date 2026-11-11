import asyncio
import json
from datetime import date
from pathlib import Path

import pytest

from regrag.agent import ToolAgent
from regrag.llm import ToolCall, ToolTurn
from regrag.register import CapaRegister, is_overdue
from regrag.tools import Toolbox

ROOT = Path(__file__).resolve().parents[1]
TODAY = date(2026, 11, 12)


@pytest.fixture
def toolbox(rag) -> Toolbox:
    return Toolbox(rag, CapaRegister.load(ROOT / "data/registers/capa_register.csv"))


# ---- register --------------------------------------------------------------------------------
def test_register_filters_and_overdue(toolbox):
    reg = toolbox.register
    assert [r["capa_id"] for r in reg.query(capa_id="capa-2026-007")] == ["CAPA-2026-007"]
    overdue = {r["capa_id"] for r in reg.query(classification="critical", overdue_as_of="2026-11-12")}
    assert overdue == {"CAPA-2026-003", "CAPA-2026-007"}  # 011 is open but not yet due
    row = reg.query(capa_id="CAPA-2026-006")[0]
    # in effectiveness check: judged against the check date, not the original due date
    assert is_overdue(row, date(2026, 10, 1)) is False and is_overdue(row, date(2026, 10, 4)) is True


def test_toolbox_reports_bad_arguments_as_data(toolbox):
    out = toolbox.call("search_sops", {"querry": "ICSR"})
    assert "error" in out
    assert "unknown tool" in toolbox.call("delete_everything", {})["error"]


# ---- offline agent ---------------------------------------------------------------------------
def test_agent_combines_register_and_sops(toolbox):
    r = ToolAgent(toolbox, today=TODAY).run(
        "CAPA-2026-006 is in effectiveness check. When must the check be done per our SOP?"
    )
    assert [s.tool for s in r.steps] == ["query_capa_register", "ask_sops"]
    assert "90 days" in r.answer and "CAPA-2026-006" in r.answer
    assert "CAPA-2026-006" in r.sources and any(s.startswith("qa-014") for s in r.sources)


def test_agent_register_only_question(toolbox):
    r = ToolAgent(toolbox, today=TODAY).run("Which critical CAPAs are overdue?")
    assert [s.tool for s in r.steps] == ["query_capa_register"]
    assert set(r.sources) == {"CAPA-2026-003", "CAPA-2026-007"}


def test_agent_abstains_out_of_scope(toolbox):
    r = ToolAgent(toolbox, today=TODAY).run("What is the parental leave policy?")
    assert r.abstained and r.sources == []


# ---- LLM tool-calling loop (scripted model, no network) ---------------------------------------
class ScriptedChat:
    name = "scripted"

    def __init__(self, turns):
        self.turns = list(turns)
        self.tool_results = []

    def start(self, system, user, tools):
        self.tool_names = {t.name for t in tools}

    def step(self):
        return self.turns.pop(0)

    def add_tool_results(self, results):
        self.tool_results.extend(results)

    def add_user(self, text):
        pass


def test_llm_loop_executes_tools_and_validates_sources(toolbox):
    chat = ScriptedChat(
        [
            ToolTurn("", [ToolCall("1", "query_capa_register", {"capa_id": "CAPA-2026-007"})]),
            ToolTurn(
                "",
                [
                    ToolCall(
                        "2", "ask_sops", {"question": "What happens to quarantined stock after an excursion?"}
                    )
                ],
            ),
            ToolTurn(
                json.dumps(
                    {
                        "answer": "CAPA-2026-007 is open; quarantined stock waits for Quality.",
                        "sources": ["CAPA-2026-007", "made-up-doc#999"],
                    }
                ),
                [],
            ),
        ]
    )
    r = ToolAgent(toolbox, chat, today=TODAY).run("Status of CAPA-2026-007 and what happens to the stock?")
    assert "query_capa_register" in chat.tool_names
    assert [s.tool for s in r.steps] == ["query_capa_register", "ask_sops"]
    assert r.structured_ok and not r.abstained
    assert r.sources == ["CAPA-2026-007"]  # guardrail: source not seen in any tool result is dropped
    assert r.dropped_sources == ["made-up-doc#999"]


def test_llm_answer_without_tools_is_refused(toolbox):
    chat = ScriptedChat([ToolTurn('{"answer": "15 days, trust me", "sources": []}', [])])
    r = ToolAgent(toolbox, chat, today=TODAY).run("Serious ICSR deadline?")
    assert r.abstained and "consulting" in r.answer


def test_llm_invalid_json_gets_one_repair(toolbox):
    chat = ScriptedChat(
        [
            ToolTurn("", [ToolCall("1", "list_documents", {})]),
            ToolTurn("There are eight documents.", []),
            ToolTurn('{"answer": "There are 8 approved documents.", "sources": []}', []),
        ]
    )
    r = ToolAgent(toolbox, chat, today=TODAY).run("How many documents are there?")
    assert r.structured_ok and r.answer.startswith("There are 8")


# ---- MCP server ------------------------------------------------------------------------------
def test_mcp_server_exposes_read_only_tools_and_resources(toolbox):
    from regrag.mcp_server import build_server

    server = build_server(toolbox)

    async def run():
        tools = {t.name: t for t in await server.list_tools()}
        assert set(tools) == {"search_sops", "ask_sops", "list_documents", "query_capa_register"}
        assert all(t.annotations.read_only_hint for t in tools.values())
        res = await server.call_tool("ask_sops", {"question": "Are shared accounts allowed on GxP systems?"})
        assert "prohibited" in res.content[0].text.lower()
        templates = await server.list_resource_templates()
        assert templates[0].uri_template == "regrag://documents/{doc_id}"
        doc = await server.read_resource("regrag://documents/qa-020-data-integrity")
        assert "ALCOA" in doc[0].content

    asyncio.run(run())
