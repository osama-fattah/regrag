from pathlib import Path

from regrag.eval import metrics as m
from regrag.eval.run import check_gates, evaluate, load_gold

GOLD = Path(__file__).resolve().parents[1] / "data/eval/gold.jsonl"


def test_metric_functions():
    ranked = m.doc_ranking(["a#000", "a#001", "b#000", "c#003"])
    assert ranked == ["a", "b", "c"]
    assert m.hit_at_k(ranked, ["b"], 1) == 0 and m.hit_at_k(ranked, ["b"], 2) == 1
    assert m.reciprocal_rank(ranked, ["c"]) == 1 / 3
    assert m.key_fact_recall("Due in 15 calendar days", ["15 calendar days", "EudraVigilance"]) == 0.5


def test_offline_baseline_passes_quality_gate(rag):
    _, summary = evaluate(rag, load_gold(GOLD))
    assert check_gates(summary) == [], summary


def test_key_fact_matching_ignores_unicode_and_spacing_variants():
    # NBSP and the "℃" compatibility character must still match the gold "2 °C to 8 °C"
    assert m.key_fact_recall("Store at 2 °C to 8 °C.", ["2 °c to 8 °c"]) == 1.0
    assert m.key_fact_recall("within 24  hours", ["24 hours"]) == 1.0


def test_gold_file_is_read_as_utf8(tmp_path, monkeypatch):
    import locale

    p = tmp_path / "g.jsonl"
    p.write_text('{"id": "x", "question": "q", "gold_docs": [], "key_facts": ["2 °c"]}\n', encoding="utf-8")
    monkeypatch.setattr(locale, "getpreferredencoding", lambda *a, **k: "cp1252")
    assert load_gold(p)[0]["key_facts"] == ["2 °c"]
