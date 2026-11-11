"""One toolbox, two consumers: the tool-calling agent (agent.py) and the MCP server (mcp_server.py).

Each tool has a JSON schema so any LLM with native tool calling (OpenAI, Azure OpenAI, Bedrock
Converse) or any MCP client can call it. All tools are read-only.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from regrag.pipeline import RegRAG
from regrag.register import CapaRegister


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]


class Toolbox:
    def __init__(self, rag: RegRAG, register: CapaRegister | None = None):
        self.rag = rag
        self.register = register
        self.specs = {s.name: s for s in self._build()}

    def _build(self) -> list[ToolSpec]:
        specs = [
            ToolSpec(
                "search_sops",
                "Hybrid keyword + semantic search over the approved SOPs and policies. Returns the "
                "most relevant passages with their document, section and chunk id. Use it to find "
                "where a topic is covered; use ask_sops for a cited answer.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to look for"},
                        "k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                    },
                    "required": ["query"],
                },
                self.search_sops,
            ),
            ToolSpec(
                "ask_sops",
                "Ask a question about procedures and policies. Returns a grounded answer with "
                "citations, or abstains if the approved documents don't support an answer.",
                {
                    "type": "object",
                    "properties": {"question": {"type": "string"}},
                    "required": ["question"],
                },
                self.ask_sops,
            ),
            ToolSpec(
                "list_documents",
                "List the approved documents available, with their ids and titles.",
                {"type": "object", "properties": {}},
                self.list_documents,
            ),
        ]
        if self.register is not None:
            specs.append(
                ToolSpec(
                    "query_capa_register",
                    "Query the CAPA (corrective and preventive action) register. All filters are "
                    "optional and combined. Use overdue_as_of (YYYY-MM-DD) to find overdue CAPAs: open "
                    "ones past their due date, or ones in effectiveness check past that date.",
                    {
                        "type": "object",
                        "properties": {
                            "capa_id": {"type": "string", "description": "e.g. CAPA-2026-007"},
                            "deviation_id": {"type": "string", "description": "e.g. DEV-2026-0058"},
                            "status": {"type": "string", "enum": ["open", "effectiveness_check", "closed"]},
                            "classification": {"type": "string", "enum": ["minor", "major", "critical"]},
                            "procedure": {"type": "string", "description": "e.g. QA-014, PV-001"},
                            "overdue_as_of": {"type": "string", "description": "ISO date"},
                        },
                    },
                    self.query_capa_register,
                )
            )
        return specs

    # ---- tool implementations -------------------------------------------------------------
    def search_sops(self, query: str, k: int = 5) -> list[dict]:
        hits = self.rag.index.search(query, k=max(1, min(int(k), 10)))
        return [
            {
                "chunk_id": h.chunk.chunk_id,
                "document": h.chunk.title,
                "section": h.chunk.section,
                "text": h.chunk.text,
            }
            for h in hits
        ]

    def ask_sops(self, question: str) -> dict:
        a = self.rag.ask(question, user="agent")
        return {
            "answer": a.answer,
            "abstained": a.abstained,
            "grounded": a.grounded,
            "citations": [
                {"n": c.n, "document": c.title, "section": c.section, "chunk_id": c.chunk_id}
                for c in a.citations
            ],
        }

    def list_documents(self) -> list[dict]:
        seen: dict[str, str] = {}
        for c in self.rag.index.chunks:
            seen.setdefault(c.doc_id, c.title)
        return [{"doc_id": d, "title": t} for d, t in sorted(seen.items())]

    def query_capa_register(self, **filters: Any) -> list[dict]:
        allowed = {"capa_id", "deviation_id", "status", "classification", "procedure", "overdue_as_of"}
        return self.register.query(**{k: v for k, v in filters.items() if k in allowed and v})

    # ---- dispatch ---------------------------------------------------------------------------
    def call(self, name: str, args: dict | None) -> Any:
        if name not in self.specs:
            return {"error": f"unknown tool {name!r}; available: {sorted(self.specs)}"}
        try:
            return self.specs[name].fn(**(args or {}))
        except (TypeError, ValueError) as e:
            # Returned to the model as data so it can correct its arguments.
            return {"error": f"bad arguments for {name}: {e}"}
