"""FastAPI service: POST /ask, GET /health."""

from __future__ import annotations

from fastapi import FastAPI, Header
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

    return app


app = create_app()
