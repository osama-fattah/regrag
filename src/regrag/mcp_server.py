"""MCP server: exposes RegRAG's tools to Claude Desktop, Cursor or any MCP client.

Run:  regrag mcp            (stdio transport; see README for client configuration)

Tools (all read-only): search_sops, ask_sops, list_documents, query_capa_register.
Resource: regrag://documents/{doc_id} returns a document's full text.
"""

from __future__ import annotations

import os
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp_types import ToolAnnotations

from regrag.pipeline import RegRAG
from regrag.register import CapaRegister
from regrag.tools import Toolbox

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)


def build_server(toolbox: Toolbox) -> MCPServer:
    server = MCPServer(
        name="regrag",
        instructions=(
            "Read-only access to a pharma company's approved SOPs and its CAPA register. Use ask_sops "
            "for cited answers about procedures, search_sops to find passages, and "
            "query_capa_register for CAPA records. Answers abstain when documents don't support them."
        ),
    )

    @server.tool(annotations=READ_ONLY)
    def search_sops(query: str, k: int = 5) -> list[dict[str, Any]]:
        """Hybrid keyword + semantic search over approved SOPs; returns passages with chunk ids."""
        return toolbox.search_sops(query, k)

    @server.tool(annotations=READ_ONLY)
    def ask_sops(question: str) -> dict[str, Any]:
        """Grounded, cited answer from the approved SOPs, or an abstention if they don't cover it."""
        return toolbox.ask_sops(question)

    @server.tool(annotations=READ_ONLY)
    def list_documents() -> list[dict[str, Any]]:
        """List the approved documents (id and title)."""
        return toolbox.list_documents()

    if toolbox.register is not None:

        @server.tool(annotations=READ_ONLY)
        def query_capa_register(
            capa_id: str | None = None,
            deviation_id: str | None = None,
            status: str | None = None,
            classification: str | None = None,
            procedure: str | None = None,
            overdue_as_of: str | None = None,
        ) -> list[dict[str, Any]]:
            """Query CAPA records. Filters are optional and combined.

            overdue_as_of: ISO date; keeps CAPAs overdue on that date (open past due, or in
            effectiveness check past the check date). status: open | effectiveness_check | closed.
            classification: minor | major | critical."""
            return toolbox.query_capa_register(
                capa_id=capa_id,
                deviation_id=deviation_id,
                status=status,
                classification=classification,
                procedure=procedure,
                overdue_as_of=overdue_as_of,
            )

    @server.resource("regrag://documents/{doc_id}", mime_type="text/plain")
    def document(doc_id: str) -> str:
        """Full text of one approved document, section by section."""
        chunks = [c for c in toolbox.rag.index.chunks if c.doc_id == doc_id]
        if not chunks:
            raise ValueError(f"Unknown document {doc_id!r}")
        return "\n\n".join(f"## {c.section}\n{c.text}" for c in chunks)

    return server


def main() -> None:
    register_path = os.getenv("REGRAG_CAPA_REGISTER", "data/registers/capa_register.csv")
    register = CapaRegister.load(register_path) if os.path.exists(register_path) else None
    build_server(Toolbox(RegRAG.from_settings(), register)).run("stdio")


if __name__ == "__main__":
    main()
