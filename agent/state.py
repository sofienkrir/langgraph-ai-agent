"""Shared state for the research workflow."""

import operator
from typing import Annotated, TypedDict


class SubQuestion(TypedDict):
    """One decomposed facet of the user's question."""

    id: str
    text: str


class Source(TypedDict):
    """A retrieved passage, always tied to the sub-question that asked for it.

    `score` is the retriever's own relevance score. It is never produced by an
    LLM: a number the model invents about its own output means nothing.

    `kind` is "web" or "document". A web source is checkable by its url, a
    document source by its locator ("rapport.pdf, p. 12"), which is why both
    fields exist instead of one.
    """

    sub_question_id: str
    title: str
    url: str
    snippet: str
    score: float
    kind: str
    locator: str


class ResearchState(TypedDict, total=False):
    question: str
    sub_questions: list[SubQuestion]
    # Each retrieval round appends; the reducer keeps earlier rounds' sources.
    sources: Annotated[list[Source], operator.add]
    # Sub-question ids that did not reach the coverage threshold.
    gaps: list[str]
    # Why each gap is a gap: "thin" (too few sources) or "ungrounded"
    # (enough sources, but none mentions what the question names).
    gap_reasons: dict[str, str]
    search_round: int
    report: str
    trace: Annotated[list[str], operator.add]
