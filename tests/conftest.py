from pathlib import Path

import pytest

from regrag.audit import AuditLog
from regrag.embeddings import HashingEmbedder
from regrag.index import HybridIndex
from regrag.ingest import load_corpus
from regrag.pipeline import RegRAG
from regrag.reasoners import HeuristicReasoner

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def index() -> HybridIndex:
    return HybridIndex.build(load_corpus(ROOT / "data/sample_corpus"), HashingEmbedder())


@pytest.fixture
def rag(index, tmp_path) -> RegRAG:
    return RegRAG(index, HeuristicReasoner(idf=index.bm25.idf), AuditLog(tmp_path / "audit.jsonl"))
