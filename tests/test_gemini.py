"""Gemini provider, tested with Google's real SDK against a mocked HTTP transport.

This checks the actual request payloads (system instruction, tool declarations, function
responses, and that an "AQ." auth key is sent in the native API-key header) and parses real
response JSON shapes. No network or key needed.
"""

import json
from datetime import date

import httpx
import pytest

genai = pytest.importorskip("google.genai")
from google.genai import types  # noqa: E402

from regrag.agent import ToolAgent  # noqa: E402
from regrag.llm import GeminiChat, GeminiToolChat  # noqa: E402
from regrag.pipeline import RegRAG  # noqa: E402
from regrag.reasoners import LLMReasoner  # noqa: E402

KEY = "AQ.test-auth-key"


def _candidate(parts):
    return {
        "candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 7, "totalTokenCount": 18},
    }


def client_with(responses, seen):
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(
            {"url": str(request.url), "headers": dict(request.headers), "body": json.loads(request.content)}
        )
        return httpx.Response(200, json=queue.pop(0))

    http = httpx.Client(transport=httpx.MockTransport(handler))
    return genai.Client(api_key=KEY, http_options=types.HttpOptions(httpx_client=http))


def test_gemini_chat_sends_native_request_with_auth_key():
    seen = []
    chat = GeminiChat("gemini-flash-latest", client=client_with([_candidate([{"text": "[1, 2]"}])], seen))
    r = chat.complete("system rules", "grade these", max_tokens=100)
    assert r.text == "[1, 2]" and (r.input_tokens, r.output_tokens) == (11, 7)
    req = seen[0]
    assert (
        "generativelanguage.googleapis.com" in req["url"]
        and "gemini-flash-latest:generateContent" in req["url"]
    )
    assert req["headers"].get("x-goog-api-key") == KEY  # native header, so AQ. keys work
    body = req["body"]
    assert body["systemInstruction"]["parts"][0]["text"] == "system rules"
    assert body["generationConfig"]["temperature"] == 0
    assert body["generationConfig"]["maxOutputTokens"] >= 4096  # headroom for thinking models


def test_gemini_tool_chat_round_trip_through_agent(toolbox_with_register):
    seen = []
    responses = [
        _candidate([{"functionCall": {"name": "query_capa_register", "args": {"capa_id": "CAPA-2026-007"}}}]),
        _candidate(
            [
                {
                    "text": json.dumps(
                        {"answer": "CAPA-2026-007 is open and overdue.", "sources": ["CAPA-2026-007"]}
                    )
                }
            ]
        ),
    ]
    chat = GeminiToolChat("gemini-flash-latest", client=client_with(responses, seen))
    r = ToolAgent(toolbox_with_register, chat, today=date(2026, 11, 12)).run("Status of CAPA-2026-007?")

    assert [s.tool for s in r.steps] == ["query_capa_register"]
    assert r.structured_ok and r.sources == ["CAPA-2026-007"] and not r.abstained
    first, second = seen[0]["body"], seen[1]["body"]
    decl = first["tools"][0]["functionDeclarations"]
    assert {d["name"] for d in decl} >= {"search_sops", "ask_sops", "query_capa_register"}
    assert {"parametersJsonSchema", "parameters_json_schema"} & set(decl[0])
    # second request carries the model's function call and our function response
    roles = [c["role"] for c in second["contents"]]
    assert roles == ["user", "model", "user"]
    fr = second["contents"][2]["parts"][0]["functionResponse"]
    assert fr["name"] == "query_capa_register" and fr["response"]["result"][0]["capa_id"] == "CAPA-2026-007"


def test_gemini_reasoner_runs_pipeline(index):
    seen = []
    responses = [
        _candidate([{"text": "[1]"}]),  # grade
        _candidate([{"text": "Serious cases must be submitted within 15 calendar days [1]."}]),  # generate
        _candidate([{"text": '{"grounded": true, "unsupported_claims": []}'}]),  # verify
    ]
    rag = RegRAG(index, LLMReasoner(GeminiChat("gemini-flash-latest", client=client_with(responses, seen))))
    a = rag.ask("How many days to submit a serious ICSR?")
    assert a.grounded and not a.abstained and "15 calendar days" in a.answer
    assert len(seen) == 3


@pytest.fixture
def toolbox_with_register(rag):
    from pathlib import Path

    from regrag.register import CapaRegister
    from regrag.tools import Toolbox

    root = Path(__file__).resolve().parents[1]
    return Toolbox(rag, CapaRegister.load(root / "data/registers/capa_register.csv"))


def test_gemini_retries_on_overloaded_503(monkeypatch):
    """Free-tier 503 'model overloaded' responses are retried with backoff, then succeed."""
    from regrag.llm import _gemini_client

    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(
                503, json={"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}}
            )
        return httpx.Response(200, json=_candidate([{"text": "ok"}]))

    monkeypatch.setattr("time.sleep", lambda s: None)  # don't actually wait in tests
    client = _gemini_client(http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert GeminiChat("gemini-flash-latest", client=client).complete("s", "u").text == "ok"
    assert len(calls) == 3
