"""Embedding providers. `hashing` runs fully offline and is used in CI."""

from __future__ import annotations

import hashlib
import json
from typing import Protocol

import numpy as np

from regrag.config import Settings
from regrag.text import content_tokens


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


class HashingEmbedder:
    """Deterministic feature-hashing embedder over unigrams and bigrams (no network, no model)."""

    def __init__(self, dim: int = 2048):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = content_tokens(text)
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return np.sign(v) * np.log1p(np.abs(v))

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalise(np.stack([self._vec(t) for t in texts]))


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small", azure: bool = False):
        import openai

        self.client = openai.AzureOpenAI() if azure else openai.OpenAI()
        self.model = model
        self.name = f"{'azure' if azure else 'openai'}:{model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 64):
            resp = self.client.embeddings.create(model=self.model, input=texts[i : i + 64])
            out.extend(d.embedding for d in resp.data)
        return _normalise(np.asarray(out, dtype=np.float32))


class BedrockEmbedder:
    def __init__(self, model: str = "amazon.titan-embed-text-v2:0", region: str = "eu-west-1"):
        import boto3

        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model = model
        self.name = f"bedrock:{model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for t in texts:
            resp = self.client.invoke_model(
                modelId=self.model, body=json.dumps({"inputText": t[:20000], "normalize": True})
            )
            out.append(json.loads(resp["body"].read())["embedding"])
        return _normalise(np.asarray(out, dtype=np.float32))


class GeminiEmbedder:
    def __init__(self, model: str = "gemini-embedding-001", client=None):
        if client is None:
            from google import genai

            client = genai.Client()
        self.client, self.model = client, model
        self.name = f"gemini:{model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 100):
            r = self.client.models.embed_content(model=self.model, contents=texts[i : i + 100])
            out.extend(e.values for e in r.embeddings)
        return _normalise(np.asarray(out, dtype=np.float32))


def get_embedder(settings: Settings) -> Embedder:
    p, m = settings.embed_provider, settings.embed_model
    if p == "hashing":
        return HashingEmbedder()
    if p == "openai":
        return OpenAIEmbedder(m or "text-embedding-3-small")
    if p == "azure":
        return OpenAIEmbedder(m or "text-embedding-3-small", azure=True)
    if p == "bedrock":
        return BedrockEmbedder(m or "amazon.titan-embed-text-v2:0", settings.aws_region)
    if p == "gemini":
        return GeminiEmbedder(m or "gemini-embedding-001")
    raise ValueError(f"Unknown embedding provider: {p}")
