"""Runtime settings, read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


@dataclass(frozen=True)
class Settings:
    llm_provider: str  # offline | openai | azure | bedrock
    embed_provider: str  # hashing | openai | azure | bedrock
    llm_model: str
    embed_model: str
    index_dir: str
    audit_log: str
    top_k: int
    max_rewrites: int
    aws_region: str

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            llm_provider=_env("REGRAG_LLM", "offline").lower(),
            embed_provider=_env("REGRAG_EMBEDDINGS", "hashing").lower(),
            llm_model=_env("REGRAG_LLM_MODEL", ""),
            embed_model=_env("REGRAG_EMBED_MODEL", ""),
            index_dir=_env("REGRAG_INDEX_DIR", ".index"),
            audit_log=_env("REGRAG_AUDIT_LOG", "logs/audit.jsonl"),
            top_k=int(_env("REGRAG_TOP_K", "6")),
            max_rewrites=int(_env("REGRAG_MAX_REWRITES", "1")),
            aws_region=_env("AWS_REGION", "eu-west-1"),
        )
