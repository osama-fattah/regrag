"""Chat LLM providers behind one small interface: OpenAI, Azure OpenAI and AWS Bedrock (Converse API)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from regrag.config import Settings


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
        import openai

        self.client = openai.AzureOpenAI() if azure else openai.OpenAI()
        self.model = model
        self.name = f"{'azure' if azure else 'openai'}:{model}"

    def complete(self, system: str, user: str, max_tokens: int = 600) -> LLMResult:
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
    raise ValueError(f"Unknown LLM provider: {p}")
