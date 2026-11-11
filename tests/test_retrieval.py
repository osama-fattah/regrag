from regrag.embeddings import HashingEmbedder
from regrag.index import HybridIndex
from regrag.ingest import chunk_document
from regrag.text import content_tokens


def test_chunks_keep_title_and_section(index):
    c = next(c for c in index.chunks if c.doc_id == "pv-001-icsr-handling" and c.section == "Day zero")
    assert c.title.startswith("PV-001")
    assert "Day zero" in c.retrieval_text


def test_long_sections_are_windowed(tmp_path):
    f = tmp_path / "long.md"
    f.write_text("# Long\n\n## Body\n\n" + " ".join(f"word{i}" for i in range(400)))
    chunks = chunk_document(f, max_words=100, overlap=20)
    assert len(chunks) >= 4
    assert all(len(c.text.split()) <= 100 for c in chunks)


def test_stemming_folds_plurals():
    assert content_tokens("assessments policies") == ["assessment", "policy"]


def test_hybrid_search_finds_the_right_sop(index):
    top = index.search("serious case EudraVigilance deadline", k=3)
    assert top[0].chunk.doc_id == "pv-001-icsr-handling"


def test_dense_only_match_still_retrieved(index):
    # "frozen" never appears verbatim in BM25-friendly form in other docs; fusion should still work
    top = index.search("what if biologics are frozen", k=3)
    assert any(h.chunk.doc_id == "gdp-003-temperature-excursions" for h in top)


def test_index_round_trip(index, tmp_path):
    index.save(tmp_path / "idx")
    loaded = HybridIndex.load(tmp_path / "idx", HashingEmbedder())
    assert len(loaded.chunks) == len(index.chunks)
    assert loaded.search("ALCOA", k=1)[0].chunk.doc_id == "qa-020-data-integrity"


def test_index_rejects_mismatched_embedder(index, tmp_path):
    index.save(tmp_path / "idx")
    try:
        HybridIndex.load(tmp_path / "idx", HashingEmbedder(dim=512))
    except ValueError as e:
        assert "re-run" in str(e)
    else:
        raise AssertionError("expected ValueError")
