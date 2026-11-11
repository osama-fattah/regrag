"""Chat LLM providers behind one small interface: OpenAI, Azure OpenAI and AWS Bedrock (Converse API)."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Protocol

from regrag.config import Settings


class _Throttle:
    """Spaces out calls to stay under a requests-per-minute cap (REGRAG_MAX_RPM), useful on free
    tiers such as Gemini or GitHub Models. 0 or unset means no limit."""

    def __init__(self):
        self.rpm = float(os.getenv("REGRAG_MAX_RPM", "0") or 0)
        self.last = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        if self.rpm <= 0:
            return
        with self.lock:
            gap = 60.0 / self.rpm - (time.monotonic() - self.last)
            if gap > 0:
                time.sleep(gap)
            self.last = time.monotonic()


_throttle = _Throttle()


def _openai_client(azure: bool):
    import openai

    # Retries with exponential backoff on 429 / 5xx, which free tiers return often.
    cls = openai.AzureOpenAI if azure else openai.OpenAI
    return cls(max_retries=int(os.getenv("REGRAG_MAX_RETRIES", "8")))


@dataclass
class LLMResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str, max_tokens: int = 600) -> LLMResult: ...


class OpenAIChat:
    def __init__(self, model: str, azure: bool = False):
        self.client = _openai_client(azure)
        self.model = model
        self.name = f"{'azure' if azure else 'openai'}:{model}"

    def complete(self, system: str, user: str, max_tokens: int = 600) -> LLMResult:
        _throttle.wait()
        r = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        u = r.usage
        return LLMResult(r.choices[0].message.content or "", u.prompt_tokens, u.completion_tokens)


class BedrockChat:
    def __init__(self, model: str, region: str):
        import boto3

        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model = model
        self.name = f"bedrock:{model}"

    def complete(self, system: str, user: str, max_tokens: int = 600) -> LLMResult:
        r = self.client.converse(
            modelId=self.model,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0},
        )
        text = "".join(b.get("text", "") for b in r["output"]["message"]["content"])
        u = r.get("usage", {})
        return LLMResult(text, u.get("inputTokens", 0), u.get("outputTokens", 0))


class GeminiChat:
    """Google Gemini via the native google-genai SDK (works with the new "AQ." auth keys, which the
    OpenAI-compatible endpoint rejects). Reads GEMINI_API_KEY or GOOGLE_API_KEY."""

    def __init__(self, model: str, client=None):
        if client is None:
            from google import genai

            client = genai.Client()
        self.client, self.model = client, model
        self.name = f"gemini:{model}"

    def complete(self, system: str, user: str, max_tokens: int = 600) -> LLMResult:
        _throttle.wait()
        r = self.client.models.generate_content(
            model=self.model, contents=user, config=_gemini_config(system, max_tokens)
        )
        return LLMResult(r.text or "", *_gemini_usage(r))


def _gemini_config(system: str, max_tokens: int, tools: list | None = None):
    from google.genai import types

    return types.GenerateContentConfig(
        system_instruction=system,
        temperature=0,
        # Thinking models spend output tokens on reasoning first; leave headroom so short
        # answers (e.g. a JSON list from the grader) aren't truncated to nothing.
        max_output_tokens=max(max_tokens * 4, 4096),
        tools=tools,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True) if tools else None,
    )


def _gemini_usage(r) -> tuple[int, int]:
    u = getattr(r, "usage_metadata", None)
    return (getattr(u, "prompt_token_count", 0) or 0, getattr(u, "candidates_token_count", 0) or 0)


# ---- native tool calling ---------------------------------------------------------------------


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class ToolTurn:
    text: str
    calls: list[ToolCall]
    input_tokens: int = 0
    output_tokens: int = 0


class ToolChat(Protocol):
    """A multi-turn conversation in which the model may request tool calls."""

    name: str

    def start(self, system: str, user: str, tools: list) -> None: ...
    def step(self) -> ToolTurn: ...
    def add_tool_results(self, results: list[tuple[ToolCall, object]]) -> None: ...
    def add_user(self, text: str) -> None: ...


def _as_dict(result: object) -> dict:
    return result if isinstance(result, dict) else {"result": result}


class OpenAIToolChat:
    """OpenAI / Azure OpenAI chat completions with `tools`."""

    def __init__(self, model: str, azure: bool = False, client=None):
        if client is None:
            client = _openai_client(azure)
        self.client, self.model = client, model
        self.name = f"{'azure' if azure else 'openai'}:{model}"
        self.messages: list[dict] = []
        self.tools: list[dict] = []

    def start(self, system, user, tools):
        self.messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        self.tools = [
            {
                "type": "function",
                "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
            }
            for t in tools
        ]

    def step(self) -> ToolTurn:
        _throttle.wait()
        r = self.client.chat.completions.create(
            model=self.model, messages=self.messages, tools=self.tools, temperature=0
        )
        msg = r.choices[0].message
        calls = [
            ToolCall(tc.id, tc.function.name, json.loads(tc.function.arguments or "{}"))
            for tc in (msg.tool_calls or [])
        ]
        entry: dict = {"role": "assistant", "content": msg.content or ""}
        if calls:
            entry["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(c.args)},
                }
                for c in calls
            ]
        self.messages.append(entry)
        u = r.usage
        return ToolTurn(msg.content or "", calls, u.prompt_tokens, u.completion_tokens)

    def add_tool_results(self, results):
        for call, result in results:
            self.messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, default=str)}
            )

    def add_user(self, text):
        self.messages.append({"role": "user", "content": text})


class BedrockToolChat:
    """AWS Bedrock Converse API with `toolConfig`."""

    def __init__(self, model: str, region: str, client=None):
        if client is None:
            import boto3

            client = boto3.client("bedrock-runtime", region_name=region)
        self.client, self.model = client, model
        self.name = f"bedrock:{model}"
        self.messages: list[dict] = []
        self.system: list[dict] = []
        self.tool_config: dict = {}

    def start(self, system, user, tools):
        self.system = [{"text": system}]
        self.messages = [{"role": "user", "content": [{"text": user}]}]
        self.tool_config = {
            "tools": [
                {
                    "toolSpec": {
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": {"json": t.parameters},
                    }
                }
                for t in tools
            ]
        }

    def step(self) -> ToolTurn:
        r = self.client.converse(
            modelId=self.model,
            system=self.system,
            messages=self.messages,
            toolConfig=self.tool_config,
            inferenceConfig={"maxTokens": 1000, "temperature": 0},
        )
        msg = r["output"]["message"]
        self.messages.append(msg)
        text = "".join(b.get("text", "") for b in msg["content"])
        calls = [
            ToolCall(b["toolUse"]["toolUseId"], b["toolUse"]["name"], b["toolUse"].get("input") or {})
            for b in msg["content"]
            if "toolUse" in b
        ]
        u = r.get("usage", {})
        return ToolTurn(text, calls, u.get("inputTokens", 0), u.get("outputTokens", 0))

    def add_tool_results(self, results):
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": call.id,
                            "content": [{"json": json.loads(json.dumps(_as_dict(result), default=str))}],
                        }
                    }
                    for call, result in results
                ],
            }
        )

    def add_user(self, text):
        self.messages.append({"role": "user", "content": [{"text": text}]})


class GeminiToolChat:
    """Gemini function calling via google-genai. The model's full turn (including any thought
    signatures) is appended to the history unchanged, as the API requires."""

    def __init__(self, model: str, client=None):
        if client is None:
            from google import genai

            client = genai.Client()
        self.client, self.model = client, model
        self.name = f"gemini:{model}"
        self.contents: list = []
        self.system = ""
        self.tools: list = []
        self._n = 0

    def start(self, system, user, tools):
        from google.genai import types

        self.system = system
        self.contents = [types.Content(role="user", parts=[types.Part.from_text(text=user)])]
        self.tools = [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name=t.name, description=t.description, parameters_json_schema=t.parameters
                    )
                    for t in tools
                ]
            )
        ]

    def step(self) -> ToolTurn:
        _throttle.wait()
        r = self.client.models.generate_content(
            model=self.model, contents=self.contents, config=_gemini_config(self.system, 1000, self.tools)
        )
        if r.candidates and r.candidates[0].content is not None:
            self.contents.append(r.candidates[0].content)
        calls = []
        for fc in r.function_calls or []:
            self._n += 1
            calls.append(ToolCall(fc.id or f"call_{self._n}", fc.name, dict(fc.args or {})))
        text = "" if calls else (r.text or "")
        return ToolTurn(text, calls, *_gemini_usage(r))

    def add_tool_results(self, results):
        from google.genai import types

        parts = [
            types.Part.from_function_response(
                name=call.name, response=json.loads(json.dumps(_as_dict(result), default=str))
            )
            for call, result in results
        ]
        self.contents.append(types.Content(role="user", parts=parts))

    def add_user(self, text):
        from google.genai import types

        self.contents.append(types.Content(role="user", parts=[types.Part.from_text(text=text)]))


def get_tool_chat(settings: Settings) -> ToolChat | None:
    p, m = settings.llm_provider, settings.llm_model
    if p == "offline":
        return None
    if p == "openai":
        return OpenAIToolChat(m or "gpt-4o-mini")
    if p == "azure":
        return OpenAIToolChat(m or "gpt-4o-mini", azure=True)
    if p == "bedrock":
        return BedrockToolChat(m or "anthropic.claude-3-5-haiku-20241022-v1:0", settings.aws_region)
    if p == "gemini":
        return GeminiToolChat(m or "gemini-flash-latest")
    raise ValueError(f"Unknown LLM provider: {p}")


def get_llm(settings: Settings) -> LLM | None:
    p, m = settings.llm_provider, settings.llm_model
    if p == "offline":
        return None
    if p == "openai":
        return OpenAIChat(m or "gpt-4o-mini")
    if p == "azure":
        return OpenAIChat(m or "gpt-4o-mini", azure=True)  # m = Azure deployment name
    if p == "bedrock":
        return BedrockChat(m or "anthropic.claude-3-5-haiku-20241022-v1:0", settings.aws_region)
    if p == "gemini":
        return GeminiChat(m or "gemini-flash-latest")
    raise ValueError(f"Unknown LLM provider: {p}")
