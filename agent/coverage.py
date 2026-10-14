"""Coverage check: pure Python, no LLM.

This is the control point of the graph. It asks questions that have verifiable
answers instead of asking a model to rate its own confidence.

Two conditions must hold for a sub-question to count as covered:

1. **Enough sources.** At least `MIN_SOURCES_PER_SUB_QUESTION` passages passed
   their retriever's relevance floor.
2. **Distinctive-term grounding.** The named things the sub-question asks about
   — proper nouns, acronyms, model names, standard codes, years — appear
   somewhere in its sources.

The second condition exists because the first is not sufficient, and the
evaluation harness is what proved it: a question about a fabricated metric
("le coefficient de Vandermeer-Oyelaran pour les centrales certifiées BX-14")
draws plenty of plausible solar-energy pages. They pass the relevance floor,
satisfy the count, and the agent declares full coverage of a question nothing
can answer. Search relevance measures topical proximity; it does not measure
whether the specific thing asked about appears anywhere.

Measuring is also what rejected the first attempt at this check. Scoring *all*
content words did not separate answerable from unanswerable questions at all
(`separable: false` in the harness): common words like "photovoltaïques" and
"coefficient" match any solar page and drown the invented names. Restricting the
check to distinctive terms separated them cleanly — 0.80/0.83/1.00 against 0.25.
Both runs are in `evaluation/results/`.
"""

import re
import unicodedata
from collections import Counter

from .state import ResearchState

MIN_SOURCES_PER_SUB_QUESTION = 2
MAX_SEARCH_ROUNDS = 2

# Share of a sub-question's distinctive terms that must appear in its sources.
# Measured with `python -m evaluation.run --grounding`, not guessed: answerable
# questions scored 0.80-1.00, the unanswerable one 0.25.
MIN_TERM_GROUNDING = 0.50

_WORD_PATTERN = re.compile(r"[A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9'\-]{2,}")


def _normalise(text: str) -> str:
    """Fold accents and case so 'certifiées' matches 'certifiees'."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def distinctive_terms(text: str) -> set[str]:
    """The named things a question asks about, as opposed to its common words.

    Three shapes qualify: a word carrying a digit (`BX-14`, `2031`), an all-caps
    acronym (`LSTM`, `RMSE`), and a capitalised word away from the start of the
    sentence (`Vandermeer-Oyelaran`, `Tunisie`). The first word is skipped
    because every sentence capitalises it.
    """
    words = _WORD_PATTERN.findall(text)
    terms = set()
    for position, word in enumerate(words):
        if position == 0:
            continue
        is_code = any(character.isdigit() for character in word)
        is_acronym = word.isupper() and len(word) > 2
        is_proper_noun = word[0].isupper() and not word.isupper()
        if is_code or is_acronym or is_proper_noun:
            terms.add(_normalise(word))
    return terms


def term_grounding(sub_question_text: str, sources) -> float:
    """Share of the question's distinctive terms that appear in its sources.

    A question with no distinctive terms scores 1.0: there is nothing specific
    to look for, so the source count alone decides.
    """
    terms = distinctive_terms(sub_question_text)
    if not terms:
        return 1.0
    haystack = _normalise(
        " ".join(f"{s.get('title', '')} {s.get('snippet', '')}" for s in sources)
    )
    return sum(1 for term in terms if term in haystack) / len(terms)


def coverage_node(state: ResearchState) -> dict:
    """Flag every sub-question that fails either coverage condition."""
    sources = state.get("sources", [])
    sub_questions = state.get("sub_questions", [])

    by_question: dict[str, list] = {}
    for source in sources:
        by_question.setdefault(source["sub_question_id"], []).append(source)
    counts = Counter(source["sub_question_id"] for source in sources)

    gaps = []
    reasons: dict[str, str] = {}
    thin, ungrounded = 0, 0
    for sub_question in sub_questions:
        qid = sub_question["id"]
        if counts[qid] < MIN_SOURCES_PER_SUB_QUESTION:
            gaps.append(qid)
            reasons[qid] = "thin"
            thin += 1
            continue
        if term_grounding(sub_question["text"], by_question.get(qid, [])) < MIN_TERM_GROUNDING:
            gaps.append(qid)
            reasons[qid] = "ungrounded"
            ungrounded += 1

    covered = len(sub_questions) - len(gaps)
    note = (
        f"Couverture : {covered}/{len(sub_questions)} sous-questions "
        f"(seuil {MIN_SOURCES_PER_SUB_QUESTION} sources + ancrage lexical "
        f"{MIN_TERM_GROUNDING:.0%})."
    )
    if thin or ungrounded:
        note += f" Lacunes : {thin} sans assez de sources, {ungrounded} hors sujet."
    return {"gaps": gaps, "gap_reasons": reasons, "trace": [note]}


def route_after_coverage(state: ResearchState) -> str:
    """Retry the gaps while rounds remain, otherwise write the report."""
    if state.get("gaps") and state.get("search_round", 0) < MAX_SEARCH_ROUNDS:
        return "search"
    return "synthesize"
