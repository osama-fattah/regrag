"""Hybrid retrieval: BM25 (lexical) + dense cosine similarity, fused with Reciprocal Rank Fusion."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from regrag.embeddings import Embedder
from regrag.ingest import Chunk
from regrag.text import content_tokens


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg = float(self.len.mean()) if len(docs) else 0.0
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        s = np.zeros(len(self.tf), dtype=np.float32)
        for i, tf in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.len[i] / (self.avg or 1))
            for t in query:
                if t in tf:
                    s[i] += self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm)
        return s


@dataclass
class Hit:
    chunk: Chunk
    score: float
    bm25_rank: int | None
    dense_rank: int | None


class HybridIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder: Embedder):
        self.chunks = chunks
        self.vectors = vectors
        self.embedder = embedder
        self.bm25 = BM25([content_tokens(c.retrieval_text) for c in chunks])

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> HybridIndex:
        return cls(chunks, embedder.embed([c.retrieval_text for c in chunks]), embedder)

    def search(self, query: str, k: int = 6, pool: int = 20, rrf_k: int = 60) -> list[Hit]:
        lex = self.bm25.scores(content_tokens(query))
        dense = self.vectors @ self.embedder.embed([query])[0]
        lex_order = [i for i in np.argsort(-lex)[:pool] if lex[i] > 0]
        dense_order = list(np.argsort(-dense)[:pool])
        fused: dict[int, float] = {}
        for order in (lex_order, dense_order):
            for rank, i in enumerate(order):
                fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (rrf_k + rank + 1)
        lex_rank = {int(i): r for r, i in enumerate(lex_order)}
        dense_rank = {int(i): r for r, i in enumerate(dense_order)}
        best = sorted(fused, key=fused.get, reverse=True)[:k]
        return [Hit(self.chunks[i], fused[i], lex_rank.get(i), dense_rank.get(i)) for i in best]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vectors)
        (path / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self.chunks], indent=1), encoding="utf-8"
        )
        (path / "meta.json").write_text(json.dumps({"embedder": self.embedder.name}), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder) -> HybridIndex:
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        if meta["embedder"] != embedder.name:
            raise ValueError(
                f"Index was built with {meta['embedder']} but current embedder is {embedder.name}; "
                "re-run `regrag ingest`."
            )
        chunks = [Chunk(**c) for c in json.loads((path / "chunks.json").read_text(encoding="utf-8"))]
        return cls(chunks, np.load(path / "vectors.npy"), embedder)
