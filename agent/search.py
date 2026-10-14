"""Retrieval node: query the local corpus and the web per sub-question."""

import os

from langchain_tavily import TavilySearch

from . import documents
from .state import ResearchState, Source

RESULTS_PER_SUB_QUESTION = 5

# Tavily's own relevance score. Below this a result is noise and must not count
# toward coverage, otherwise the gap check passes on irrelevant hits.
MIN_RELEVANCE = 0.4


def _tavily(depth: str) -> TavilySearch:
    if not os.getenv("TAVILY_API_KEY", "").strip():
        raise ValueError(
            "TAVILY_API_KEY manquante. Crée une clé sur https://app.tavily.com "
            "et ajoute-la dans ton fichier .env."
        )
    return TavilySearch(max_results=RESULTS_PER_SUB_QUESTION, search_depth=depth)


def _to_sources(payload, sub_question_id: str) -> list[Source]:
    """Normalise Tavily's response, dropping low-relevance hits."""
    results = payload.get("results", []) if isinstance(payload, dict) else []
    sources: list[Source] = []
    for item in results:
        score = float(item.get("score") or 0.0)
        url = (item.get("url") or "").strip()
        if score < MIN_RELEVANCE or not url:
            continue
        sources.append({
            "sub_question_id": sub_question_id,
            "title": (item.get("title") or url).strip(),
            "url": url,
            "snippet": (item.get("content") or "").strip(),
            "score": score,
            "kind": "web",
            "locator": url,
        })
    return sources


def _dedup_key(source: Source) -> tuple:
    """Web results are unique by url; local passages have none, so use the
    locator plus a slice of the text to tell two chunks of a page apart."""
    if source["kind"] == "web":
        return ("web", source["url"])
    return ("document", source["locator"], source["snippet"][:120])


def search_node(state: ResearchState) -> dict:
    """Search for every sub-question, or only the gaps on a follow-up round."""
    sub_questions = state.get("sub_questions", [])
    gaps = state.get("gaps", [])
    search_round = state.get("search_round", 0)

    # First round covers everything; later rounds only chase what came up short.
    targets = [sq for sq in sub_questions if sq["id"] in gaps] if gaps else sub_questions

    # The follow-up round must not repeat the query that already failed, so it
    # digs deeper rather than asking again.
    depth = "advanced" if search_round > 0 else "basic"

    use_documents = documents.has_documents()
    collected: list[Source] = []
    seen = {_dedup_key(source) for source in state.get("sources", [])}
    notes: list[str] = []
    from_documents = 0

    # The local corpus is free to query and never rate-limited, so it runs even
    # if the web search cannot be configured.
    if use_documents:
        for sub_question in targets:
            for source in documents.search_documents(sub_question["text"], sub_question["id"]):
                key = _dedup_key(source)
                if key in seen:
                    continue
                seen.add(key)
                collected.append(source)
                from_documents += 1

    try:
        tool = _tavily(depth)
    except ValueError as exc:
        if not use_documents:
            raise
        notes.append(f"Recherche Web désactivée : {exc}")
        tool = None

    if tool is not None:
        for sub_question in targets:
            try:
                payload = tool.invoke({"query": sub_question["text"]})
            except Exception as exc:
                notes.append(f"Recherche en échec pour {sub_question['id']} : {exc}")
                continue
            for source in _to_sources(payload, sub_question["id"]):
                key = _dedup_key(source)
                if key in seen:
                    continue
                seen.add(key)
                collected.append(source)

    origin = f"{from_documents} document(s), {len(collected) - from_documents} web"
    notes.append(
        f"Tour {search_round + 1} ({depth}) : {len(targets)} sous-question(s) interrogée(s), "
        f"{len(collected)} source(s) retenue(s) — {origin}."
    )
    return {"sources": collected, "search_round": search_round + 1, "trace": notes}
