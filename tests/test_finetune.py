"""LoRA reranker fine-tuning, exercised end to end on a tiny randomly initialised BERT
cross-encoder built locally (no model download). Skipped unless the `finetune` extra is installed."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

from regrag.eval.run import load_gold  # noqa: E402
from regrag.finetune import (  # noqa: E402
    _jaccard,
    build_pairs,
    heuristic_queries,
    retrieval_eval,
    train_lora_reranker,
)
from regrag.rerank import CrossEncoderReranker, load_reranker  # noqa: E402
from regrag.text import tokenize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tiny_base(tmp_path_factory, index):
    d = tmp_path_factory.mktemp("tiny-cross-encoder")
    words = sorted({t for c in index.chunks for t in tokenize(c.retrieval_text)})
    (d / "vocab.txt").write_text(
        "\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words]), encoding="utf-8"
    )
    tok = transformers.BertTokenizerFast(vocab_file=str(d / "vocab.txt"), do_lower_case=True)
    tok.save_pretrained(d)
    cfg = transformers.BertConfig(
        vocab_size=tok.vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=512,
        num_labels=1,
    )
    torch.manual_seed(0)
    transformers.BertForSequenceClassification(cfg).save_pretrained(d)
    return str(d)


def test_training_pairs_exclude_eval_questions(index):
    gold = load_gold(ROOT / "data/eval/gold.jsonl") + load_gold(ROOT / "data/eval/retrieval_hard.jsonl")
    qs = [g["question"] for g in gold]
    pairs, dropped = build_pairs(index, heuristic_queries, qs)
    assert len(pairs) > 40
    assert all(_jaccard(p.query, q) < 0.5 for p in pairs for q in qs)  # no eval leakage
    assert all(p.positive not in p.negatives for p in pairs)


def test_lora_training_saves_adapter_and_reranks(index, tiny_base, tmp_path):
    pairs, _ = build_pairs(index, heuristic_queries)
    out = tmp_path / "adapter"
    s = train_lora_reranker(pairs[:30], index, tiny_base, out, epochs=4, lr=5e-3, batch_size=8)

    assert 0 < s.trainable_params < s.total_params  # only adapters + head are trained
    assert s.epoch_losses[-1] < s.epoch_losses[0]  # it learns
    assert (out / "adapter_config.json").exists() and (out / "regrag_meta.txt").read_text() == tiny_base

    rr = load_reranker(str(out))
    assert rr.name.endswith("+lora")
    scores = rr.score("serious case deadline", ["Serious cases within 15 calendar days.", "Store at 2 °C"])
    assert len(scores) == 2

    index.reranker = rr
    try:
        hits = index.search("shared user accounts", k=5)
        assert len(hits) == 5 and all(isinstance(h.score, float) for h in hits)
        metrics = retrieval_eval(index, load_gold(ROOT / "data/eval/retrieval_hard.jsonl"))
        assert metrics["questions"] == 22
    finally:
        index.reranker = None


def test_base_reranker_without_adapter(tiny_base):
    rr = CrossEncoderReranker(tiny_base)
    assert rr.name.startswith("cross-encoder:") and not rr.name.endswith("+lora")
    assert len(rr.score("q", ["a", "b", "c"])) == 3
