"""FastAPI service: POST /ask, GET /health."""

from __future__ import annotations

import json

from fastapi import FastAPI, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from regrag import __version__
from regrag.pipeline import RegRAG


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


def create_app(rag: RegRAG | None = None) -> FastAPI:
    app = FastAPI(title="RegRAG", version=__version__)
    state = {"rag": rag}

    def get_rag() -> RegRAG:
        if state["rag"] is None:
            state["rag"] = RegRAG.from_settings()
        return state["rag"]

    @app.get("/health")
    def health() -> dict:
        r = get_rag()
        return {"status": "ok", "chunks": len(r.index.chunks), "model": r.reasoner.name}

    @app.post("/ask")
    def ask(req: AskRequest, x_user: str = Header(default="anonymous")) -> dict:
        return get_rag().ask(req.question, user=x_user).to_dict()

    @app.post("/ask/stream")
    def ask_stream(req: AskRequest, x_user: str = Header(default="anonymous")) -> StreamingResponse:
        """Server-sent events: one `step` event per agent node, then a final `answer` event."""

        def events():
            for kind, payload in get_rag().stream(req.question, user=x_user):
                data = payload.to_dict() if kind == "answer" else payload
                yield f"event: {kind}\ndata: {json.dumps(data, default=str)}\n\n"

        return StreamingResponse(
            events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    return app


app = create_app()
