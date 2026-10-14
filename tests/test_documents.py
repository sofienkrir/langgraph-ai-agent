"""Tests for PDF ingestion and local retrieval.

Everything here runs without the embedding model or the vector store, so the
suite stays fast and offline. The parts that need Chroma are covered by the
cosine-space regression test, which asserts on configuration, not on a query.
"""

import agent.documents as documents
from agent.search import _dedup_key


def test_clean_rejoins_hyphenated_line_breaks():
    """PDF extraction splits words across lines; leaving them breaks retrieval."""
    assert documents._clean("photovol-\ntaique") == "photovoltaique"


def test_clean_collapses_whitespace_and_blank_lines():
    assert documents._clean("a   b\n\n\n\nc") == "a b\n\nc"
    assert documents._clean("non\xa0breaking") == "non breaking"


def test_clean_strips_surrounding_whitespace():
    assert documents._clean("\n\n  texte  \n\n") == "texte"


def test_search_returns_nothing_without_a_corpus(monkeypatch):
    """No documents indexed must mean no local hits, not a crash."""
    monkeypatch.setattr(documents, "has_documents", lambda: False)
    assert documents.search_documents("une question", "q1") == []


def test_search_survives_a_broken_store(monkeypatch):
    monkeypatch.setattr(documents, "has_documents", lambda: True)
    monkeypatch.setattr(documents, "get_store", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert documents.search_documents("une question", "q1") == []


def test_low_relevance_passages_are_dropped(monkeypatch):
    """The coverage gate is only trustworthy if weak hits never reach it."""
    class FakeDoc:
        page_content = "contenu"
        metadata = {"source": "a.pdf", "page": 3}

    class FakeStore:
        def similarity_search_with_relevance_scores(self, query, k):
            return [
                (FakeDoc(), documents.MIN_RELEVANCE + 0.1),
                (FakeDoc(), documents.MIN_RELEVANCE - 0.1),
                (FakeDoc(), None),
            ]

    monkeypatch.setattr(documents, "has_documents", lambda: True)
    monkeypatch.setattr(documents, "get_store", lambda: FakeStore())

    hits = documents.search_documents("une question", "q1")
    assert len(hits) == 1
    assert hits[0]["kind"] == "document"
    assert hits[0]["locator"] == "a.pdf, p. 3"
    assert hits[0]["url"] == ""


def test_index_dir_is_anchored_to_the_project_root():
    """A relative path would point at a different index per working directory."""
    assert documents.INDEX_DIR.is_absolute()
    assert documents.CACHE_DIR.is_absolute()


def test_store_requests_cosine_space():
    """Regression: Chroma's default L2 space yields negative relevance scores,
    so every passage falls below the threshold and retrieval silently returns
    nothing. The collection must be created with cosine distance."""
    import inspect

    source = inspect.getsource(documents.get_store)
    assert '"hnsw:space": "cosine"' in source


def test_web_and_document_sources_dedup_independently():
    web = {"kind": "web", "url": "https://a.test", "locator": "https://a.test", "snippet": "x"}
    doc_a = {"kind": "document", "url": "", "locator": "a.pdf, p. 1", "snippet": "premier passage"}
    doc_b = {"kind": "document", "url": "", "locator": "a.pdf, p. 1", "snippet": "second passage"}

    # Two chunks of the same page share a locator and must not collapse into one.
    assert _dedup_key(doc_a) != _dedup_key(doc_b)
    # An empty url must never make distinct documents look identical.
    assert _dedup_key(doc_a) != _dedup_key(web)
    assert _dedup_key(web) == _dedup_key(dict(web, snippet="autre texte"))


def test_clear_uses_the_collection_api_not_the_filesystem(monkeypatch):
    """Regression: removing the index directory silently fails on Windows while
    any process holds the store open, so a "cleared" corpus quietly survived."""
    called = {"reset": False, "rmtree": False}

    class FakeStore:
        def reset_collection(self):
            called["reset"] = True

    monkeypatch.setattr(documents, "get_store", lambda: FakeStore())
    monkeypatch.setattr(documents.shutil, "rmtree", lambda *a, **k: called.update(rmtree=True))

    documents.clear_index()
    assert called["reset"] is True
    assert called["rmtree"] is False


def test_clear_falls_back_when_the_collection_cannot_be_opened(monkeypatch):
    called = {"rmtree": False}

    def boom():
        raise RuntimeError("store indisponible")

    monkeypatch.setattr(documents, "get_store", boom)
    monkeypatch.setattr(documents.INDEX_DIR.__class__, "exists", lambda self: True)
    monkeypatch.setattr(documents.shutil, "rmtree", lambda *a, **k: called.update(rmtree=True))

    documents.clear_index()
    assert called["rmtree"] is True
