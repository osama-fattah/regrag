"""LoRA fine-tuning of a cross-encoder reranker on the document corpus.

1. Synthetic training queries: for each passage, an LLM (or an offline heuristic) writes
   questions the passage answers. Queries too similar to the evaluation set are dropped, so
   the gold questions never leak into training.
2. Hard negatives: other passages the hybrid retriever ranks highly for the same query.
3. LoRA: only low-rank adapters on the attention query/value projections (plus the scoring
   head) are trained; the base model stays frozen. Binary cross-entropy on (query, passage).
4. Evaluation: passage-level retrieval (is the exact passage containing the answer ranked
   first?) with and without the reranker.

Requires the `finetune` extra: pip install -e ".[finetune]".
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from regrag.eval.metrics import _norm
from regrag.index import HybridIndex
from regrag.ingest import Chunk
from regrag.prompts import SYSTEM
from regrag.reasoners import _parse_json
from regrag.text import content_tokens, sentences

QUERY_PROMPT = """Write {n} different questions an employee might ask that this passage answers.
Use your own wording; do not copy phrases from the passage. Return only a JSON list of strings.

Passage ({title} > {section}):
{text}"""


@dataclass
class Pair:
    query: str
    positive: str  # chunk_id
    negatives: list[str]


@dataclass
class TrainSummary:
    base_model: str
    n_pairs: int
    n_examples: int
    epochs: int
    lora_r: int
    trainable_params: int
    total_params: int
    epoch_losses: list[float] = field(default_factory=list)
    seconds: float = 0.0


# ---- training data ---------------------------------------------------------------------------
def heuristic_queries(chunk: Chunk, n: int = 3) -> list[str]:
    """Offline fallback: keyword-style queries from the passage's most informative sentences."""
    out = []
    for s in sentences(chunk.text)[:n]:
        toks = [t for t in content_tokens(s) if len(t) > 3][:6]
        if len(toks) >= 3:
            out.append(f"{chunk.section.lower()}: {' '.join(toks)}")
    return out


def llm_queries(llm) -> Callable[[Chunk, int], list[str]]:
    def gen(chunk: Chunk, n: int = 3) -> list[str]:
        r = llm.complete(
            SYSTEM, QUERY_PROMPT.format(n=n, title=chunk.title, section=chunk.section, text=chunk.text), 400
        )
        qs = _parse_json(r.text, [])
        return [q.strip() for q in qs if isinstance(q, str) and len(q.strip()) > 8][:n]

    return gen


def _jaccard(a: str, b: str) -> float:
    x, y = set(content_tokens(a)), set(content_tokens(b))
    return len(x & y) / len(x | y) if x | y else 0.0


def build_pairs(
    index: HybridIndex,
    query_fn: Callable[[Chunk, int], list[str]],
    exclude_questions: list[str] = (),
    per_chunk: int = 3,
    n_negatives: int = 3,
    leak_threshold: float = 0.5,
) -> tuple[list[Pair], int]:
    """Returns (pairs, number of queries dropped as too close to the evaluation set)."""
    pairs, dropped = [], 0
    saved, index.reranker = index.reranker, None  # mine negatives with plain hybrid retrieval
    try:
        for chunk in index.chunks:
            if chunk.section == "Overview":
                continue
            for q in query_fn(chunk, per_chunk):
                if any(_jaccard(q, g) >= leak_threshold for g in exclude_questions):
                    dropped += 1
                    continue
                negs = [
                    h.chunk.chunk_id
                    for h in index.search(q, k=n_negatives + 4)
                    if h.chunk.chunk_id != chunk.chunk_id
                ]
                if negs:
                    pairs.append(Pair(q, chunk.chunk_id, negs[:n_negatives]))
    finally:
        index.reranker = saved
    return pairs, dropped


# ---- training --------------------------------------------------------------------------------
def train_lora_reranker(
    pairs: list[Pair],
    index: HybridIndex,
    base_model: str,
    out_dir: str | Path,
    epochs: int = 2,
    lr: float = 3e-4,
    r: int = 8,
    alpha: int = 16,
    batch_size: int = 16,
    seed: int = 0,
    target_modules: list[str] | None = None,
) -> TrainSummary:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(seed)
    torch.manual_seed(seed)
    by_id = {c.chunk_id: c for c in index.chunks}
    examples = []
    for p in pairs:
        examples.append((p.query, by_id[p.positive].retrieval_text, 1.0))
        examples += [(p.query, by_id[n].retrieval_text, 0.0) for n in p.negatives]

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=1)
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.1,
        target_modules=target_modules or ["query", "value"],
    )
    model = get_peft_model(model, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    summary = TrainSummary(base_model, len(pairs), len(examples), epochs, r, trainable, total)
    t0 = time.perf_counter()
    model.train()
    for _ in range(epochs):
        random.shuffle(examples)
        losses = []
        for i in range(0, len(examples), batch_size):
            batch = examples[i : i + batch_size]
            enc = tokenizer(
                [b[0] for b in batch],
                [b[1] for b in batch],
                truncation=True,
                max_length=384,
                padding=True,
                return_tensors="pt",
            )
            labels = torch.tensor([b[2] for b in batch], dtype=torch.float32)
            logits = model(**enc).logits.squeeze(-1).reshape(-1)
            loss = loss_fn(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        summary.epoch_losses.append(round(sum(losses) / len(losses), 4))
    summary.seconds = round(time.perf_counter() - t0, 1)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    (out / "regrag_meta.txt").write_text(base_model, encoding="utf-8")
    (out / "training_summary.json").write_text(json.dumps(asdict(summary), indent=1), encoding="utf-8")
    (out / "training_pairs.jsonl").write_text(
        "\n".join(json.dumps(asdict(p)) for p in pairs) + "\n", encoding="utf-8"
    )
    return summary


# ---- passage-level retrieval evaluation ------------------------------------------------------
def evidence_chunks(index: HybridIndex, key_facts: list[str]) -> set[str]:
    """Chunks whose text contains every key fact: the passage that actually answers."""
    facts = [_norm(f) for f in key_facts]
    return {c.chunk_id for c in index.chunks if facts and all(f in _norm(c.text) for f in facts)}


def retrieval_eval(index: HybridIndex, gold: list[dict], k: int = 10) -> dict:
    hit1, mrr, n = 0.0, 0.0, 0
    for g in gold:
        if not g.get("answerable", True):
            continue
        ev = evidence_chunks(index, g["key_facts"])
        if not ev:
            continue
        n += 1
        ranked = [h.chunk.chunk_id for h in index.search(g["question"], k=k)]
        hit1 += float(bool(ranked) and ranked[0] in ev)
        mrr += next((1.0 / i for i, cid in enumerate(ranked, 1) if cid in ev), 0.0)
    return {
        "questions": n,
        "passage_hit@1": round(hit1 / n, 3) if n else None,
        "passage_mrr": round(mrr / n, 3) if n else None,
    }
