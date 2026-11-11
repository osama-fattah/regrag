"""Cross-encoder reranking, optionally with a LoRA adapter fine-tuned on the SOP corpus.

Hybrid retrieval returns a candidate pool; the cross-encoder reads each (query, passage) pair
jointly and re-orders the pool. Requires the `finetune` extra (torch, transformers, peft).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

DEFAULT_BASE = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(
        self, base_model: str = DEFAULT_BASE, adapter: str | Path | None = None, max_length: int = 384
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(adapter or base_model))
        model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=1)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter))
        self.model = model.eval()
        self.max_length = max_length
        self.name = f"cross-encoder:{Path(base_model).name}" + ("+lora" if adapter else "")

    def score(self, query: str, passages: list[str], batch_size: int = 16) -> list[float]:
        out: list[float] = []
        with self.torch.no_grad():
            for i in range(0, len(passages), batch_size):
                batch = passages[i : i + batch_size]
                enc = self.tokenizer(
                    [query] * len(batch),
                    batch,
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                    return_tensors="pt",
                )
                out.extend(self.model(**enc).logits.squeeze(-1).reshape(-1).tolist())
        return out


def load_reranker(spec: str | None, base_model: str = DEFAULT_BASE) -> Reranker | None:
    """spec: None/"" = no reranking; "base" = pretrained cross-encoder; otherwise a LoRA adapter dir."""
    if not spec:
        return None
    if spec == "base":
        return CrossEncoderReranker(base_model)
    adapter = Path(spec)
    meta = adapter / "regrag_meta.txt"
    base = meta.read_text(encoding="utf-8").strip() if meta.exists() else base_model
    return CrossEncoderReranker(base, adapter)
