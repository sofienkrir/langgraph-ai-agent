"""Local PDF corpus: ingest, index, and retrieve with citations.

Neither Groq nor Anthropic serves an embeddings endpoint, so embedding runs
locally through fastembed (ONNX, no torch). The model is multilingual because
the app is used in French against sources that are largely in English.
"""

import hashlib
import os
import re
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .state import Source

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Anchored to the project root, not the working directory: the index must be the
# same one whether the app is launched from here or from a parent folder.
_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = _ROOT / ".cache" / "fastembed"
INDEX_DIR = _ROOT / ".cache" / "chroma"
COLLECTION = "documents"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RESULTS_PER_SUB_QUESTION = 4

# Chroma's normalised relevance, not Tavily's score. The two retrievers are
# scored on different scales, so they get separate thresholds rather than a
# shared constant that would pretend the numbers are comparable.
#
# 0.50 was measured, not guessed: `python -m evaluation.run --sweep` shows it
# keeps the same Recall@4 as a permissive 0.25 while cutting the rate at which
# off-topic questions still draw a passage from 83% to 33%. Embedding similarity
# scores topical proximity, not evidential relevance, so a question about wind
# turbines scores ~0.55 against a solar paper; the floor is what stops those
# passages from counting toward coverage. Raising it further trades away real
# recall — see the balance column in the sweep.
MIN_RELEVANCE = 0.50

_embeddings: Embeddings | None = None
_store: Chroma | None = None


class FastEmbedAdapter(Embeddings):
    """Minimal LangChain Embeddings over fastembed.

    Thin on purpose: langchain-community would pull a large dependency tree in
    exchange for this one class.
    """

    def __init__(self, model_name: str = EMBED_MODEL):
        from fastembed import TextEmbedding

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(CACHE_DIR))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, vector)) for vector in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_embeddings() -> Embeddings:
    """Load the embedding model once; the first call downloads it."""
    global _embeddings
    if _embeddings is None:
        _embeddings = FastEmbedAdapter()
    return _embeddings


def get_store() -> Chroma:
    """Open the on-disk index, so an uploaded PDF survives a restart."""
    global _store
    if _store is None:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        _store = Chroma(
            collection_name=COLLECTION,
            embedding_function=get_embeddings(),
            persist_directory=str(INDEX_DIR),
            # Chroma defaults to squared L2, whose relevance conversion yields
            # large negative numbers for these vectors — every hit would fall
            # below any sane threshold and retrieval would silently return
            # nothing. Cosine keeps the score in [0, 1] as the filter assumes.
            collection_metadata={"hnsw:space": "cosine"},
        )
    return _store


def _clean(text: str) -> str:
    """PDF extraction leaves hard-wrapped lines and runs of whitespace."""
    text = text.replace("\xa0", " ")
    text = re.sub(r"-\n(?=\w)", "", text)  # de-hyphenate across line breaks
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def read_pdf(data: bytes, filename: str) -> list[Document]:
    """Extract one Document per page, keeping the page number for citation."""
    import io

    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = _clean(page.extract_text() or "")
        except Exception:
            continue
        if len(text) < 40:
            # A page that yields almost nothing is a scan; OCR is out of scope,
            # and indexing the noise would only pollute retrieval.
            continue
        pages.append(
            Document(page_content=text, metadata={"source": filename, "page": number})
        )
    return pages


def index_pdf(data: bytes, filename: str) -> dict:
    """Chunk and index one PDF. Re-uploading the same file is a no-op."""
    digest = hashlib.sha256(data).hexdigest()[:16]
    store = get_store()

    existing = store.get(where={"digest": digest}, limit=1)
    if existing.get("ids"):
        return {"filename": filename, "chunks": 0, "pages": 0, "skipped": True}

    pages = read_pdf(data, filename)
    if not pages:
        return {"filename": filename, "chunks": 0, "pages": 0, "skipped": False}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    for chunk in chunks:
        chunk.metadata["digest"] = digest

    store.add_documents(chunks)
    return {
        "filename": filename,
        "chunks": len(chunks),
        "pages": len(pages),
        "skipped": False,
    }


def indexed_files() -> list[str]:
    """Names of the PDFs currently in the index."""
    if not INDEX_DIR.exists():
        return []
    try:
        records = get_store().get(include=["metadatas"])
    except Exception:
        return []
    names = {
        (meta or {}).get("source")
        for meta in records.get("metadatas", [])
        if (meta or {}).get("source")
    }
    return sorted(names)


def has_documents() -> bool:
    """Whether the graph should bother querying the local corpus at all."""
    return bool(indexed_files())


def clear_index() -> None:
    """Drop the whole corpus.

    Deleting through Chroma rather than removing the directory: on Windows the
    index files stay locked while any process holds the store open, so an
    `rmtree` silently does nothing and the corpus survives a "clear" that
    reported success. Falling back to the directory removal only covers the
    case where the collection cannot be opened at all.
    """
    global _store
    try:
        get_store().reset_collection()
        return
    except Exception:
        _store = None
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR, ignore_errors=True)


def search_documents(
    query: str,
    sub_question_id: str,
    k: int = RESULTS_PER_SUB_QUESTION,
    min_relevance: float | None = None,
) -> list[Source]:
    """Retrieve local passages for one sub-question, shaped like web sources.

    `min_relevance` is overridable so the evaluation harness can sweep the
    threshold instead of hard-coding a guess.
    """
    if not has_documents():
        return []
    floor = MIN_RELEVANCE if min_relevance is None else min_relevance
    try:
        hits = get_store().similarity_search_with_relevance_scores(query, k=k)
    except Exception:
        return []

    sources: list[Source] = []
    for document, score in hits:
        if score is None or score < floor:
            continue
        name = document.metadata.get("source", "document.pdf")
        page = document.metadata.get("page")
        sources.append({
            "sub_question_id": sub_question_id,
            "title": f"{name} — page {page}" if page else name,
            # A local file has no URL; the locator is what makes it checkable.
            "url": "",
            "snippet": document.page_content.strip(),
            "score": float(score),
            "kind": "document",
            "locator": f"{name}, p. {page}" if page else name,
        })
    return sources


def index_stats() -> dict:
    """Small summary for the UI."""
    files = indexed_files()
    try:
        total = len(get_store().get(include=[]).get("ids", [])) if files else 0
    except Exception:
        total = 0
    return {"files": files, "chunks": total}


# Keep Chroma's telemetry quiet; it writes noise to stderr in a Streamlit run.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
