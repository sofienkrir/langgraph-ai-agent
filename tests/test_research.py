"""Tests for the parts of the workflow that must not depend on an LLM."""

import pytest

from agent.coverage import (
    MAX_SEARCH_ROUNDS,
    MIN_SOURCES_PER_SUB_QUESTION,
    distinctive_terms,
    coverage_node,
    route_after_coverage,
    term_grounding,
)
from agent.llm import active_provider, model_for
from agent.search import MIN_RELEVANCE, _to_sources
from agent.synthesis import _invalid_citations, _limits_section


def _state(source_counts, search_round=1, grounded=True):
    """Build a state whose sources quote the sub-question, so the count rule is
    what the test exercises. `grounded=False` makes the sources off-topic."""
    sub_questions = [{"id": qid, "text": f"la turbine LSTM-{qid} est-elle efficace ?"} for qid in source_counts]
    sources = [
        {
            "sub_question_id": qid,
            "title": "",
            "snippet": f"la turbine LSTM-{qid} fonctionne" if grounded else "sujet totalement different",
        }
        for qid, count in source_counts.items()
        for _ in range(count)
    ]
    return {"sub_questions": sub_questions, "sources": sources, "search_round": search_round}


def test_coverage_flags_under_threshold():
    state = _state({"q1": MIN_SOURCES_PER_SUB_QUESTION, "q2": 0})
    assert coverage_node(state)["gaps"] == ["q2"]


def test_coverage_passes_when_all_met():
    state = _state({"q1": MIN_SOURCES_PER_SUB_QUESTION, "q2": MIN_SOURCES_PER_SUB_QUESTION})
    assert coverage_node(state)["gaps"] == []


def test_router_retries_while_rounds_remain():
    state = {"gaps": ["q2"], "search_round": 1}
    assert route_after_coverage(state) == "search"


def test_router_stops_when_rounds_exhausted():
    """A gap must never loop forever; the report ships with the gap declared."""
    state = {"gaps": ["q2"], "search_round": MAX_SEARCH_ROUNDS}
    assert route_after_coverage(state) == "synthesize"


def test_router_proceeds_without_gaps():
    assert route_after_coverage({"gaps": [], "search_round": 1}) == "synthesize"


def test_low_relevance_results_are_dropped():
    payload = {"results": [
        {"url": "https://a.test", "title": "A", "content": "x", "score": MIN_RELEVANCE + 0.1},
        {"url": "https://b.test", "title": "B", "content": "x", "score": MIN_RELEVANCE - 0.1},
        {"url": "", "title": "C", "content": "x", "score": 0.9},
    ]}
    sources = _to_sources(payload, "q1")
    assert [s["url"] for s in sources] == ["https://a.test"]


def test_malformed_search_payload_is_survivable():
    assert _to_sources(None, "q1") == []
    assert _to_sources({}, "q1") == []


def test_fabricated_citations_are_detected():
    assert _invalid_citations("voir [1] et [2]", 3) == []
    assert _invalid_citations("voir [1] et [9]", 3) == [9]


def test_limits_section_names_uncovered_sub_questions():
    state = {"sub_questions": [{"id": "q2", "text": "Quel rendement ?"}], "gaps": ["q2"]}
    section = _limits_section(state, [])
    assert "Quel rendement ?" in section
    assert "## Limites" in section


def test_limits_section_reports_invalid_citations():
    assert "[9]" in _limits_section({"sub_questions": [], "gaps": []}, [9])


def test_limits_section_present_even_when_fully_covered():
    """The report always carries a limits section, covered or not."""
    assert "## Limites" in _limits_section({"sub_questions": [], "gaps": []}, [])


@pytest.mark.parametrize("role", ["planner", "synthesizer"])
def test_every_role_resolves_on_every_provider(role):
    for provider in ("groq", "anthropic"):
        assert model_for(role, provider)


def test_unknown_role_is_rejected():
    with pytest.raises(ValueError):
        model_for("nonexistent")


def test_provider_defaults_to_groq(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert active_provider() == "groq"


# --- lexical grounding: the second coverage condition -------------------------


def test_distinctive_terms_keeps_names_codes_and_acronyms():
    terms = distinctive_terms("Le coefficient de Vandermeer-Oyelaran certifie BX-14 avec LSTM")
    assert "vandermeer-oyelaran" in terms
    assert "bx-14" in terms
    assert "lstm" in terms


def test_distinctive_terms_drops_ordinary_words():
    terms = distinctive_terms("Quelles sont les principales methodes de prevision solaire ?")
    assert terms == set()


def test_distinctive_terms_ignores_the_leading_capital():
    """Every sentence capitalises its first word; that is not a proper noun."""
    assert distinctive_terms("Prevision de la production") == set()


def test_distinctive_terms_folds_accents():
    assert "tunisie" in distinctive_terms("les centrales de Tunisie")


def test_grounding_is_one_when_named_things_appear():
    sources = [{"title": "LSTM", "snippet": "les reseaux LSTM pour la prevision"}]
    assert term_grounding("Les reseaux LSTM sont-ils efficaces ?", sources) == 1.0


def test_grounding_is_low_for_invented_names_absent_everywhere():
    """The failure the evaluation harness caught: plausible pages, invented subject.

    Scoring every content word did not catch this - 'coefficient' and
    'photovoltaiques' match any solar page. Only the named things separate."""
    sources = [
        {"title": "Energie solaire", "snippet": "les centrales photovoltaiques produisent de l electricite"},
        {"title": "Photovoltaique", "snippet": "rendement des centrales photovoltaiques en 2024"},
    ]
    score = term_grounding("le coefficient de Vandermeer-Oyelaran pour BX-14", sources)
    assert score < 0.5


def test_grounding_without_sources_is_zero_when_terms_exist():
    assert term_grounding("le modele QLSTM", []) == 0.0


def test_grounding_of_a_question_naming_nothing_specific():
    """Nothing specific to look for means the source count alone decides."""
    assert term_grounding("quelles sont les methodes courantes ?", []) == 1.0


def test_enough_sources_but_off_topic_is_still_a_gap():
    state = _state({"q1": MIN_SOURCES_PER_SUB_QUESTION}, grounded=False)
    assert coverage_node(state)["gaps"] == ["q1"]


def test_coverage_note_distinguishes_the_two_failure_modes():
    state = _state({"q1": 0}, grounded=False)
    note = coverage_node(state)["trace"][0]
    assert "sans assez de sources" in note


def test_internal_ids_are_stripped_from_headings():
    """Sub-question ids are internal plumbing; a published report must not show them."""
    from agent.synthesis import _strip_internal_ids

    assert _strip_internal_ids("## q1 – Definition technique") == "## Definition technique"
    assert _strip_internal_ids("## (q2) Les methodes") == "## Les methodes"
    assert _strip_internal_ids("### q10 : Resultats") == "### Resultats"


def test_stripping_leaves_ordinary_headings_alone():
    from agent.synthesis import _strip_internal_ids

    assert _strip_internal_ids("## Quartier q1 du port") == "## Quartier q1 du port"
    assert _strip_internal_ids("## Resultats") == "## Resultats"


def test_stripping_does_not_touch_body_text():
    """An id mentioned in a sentence is not a heading and must survive."""
    from agent.synthesis import _strip_internal_ids

    assert _strip_internal_ids("La q1 reste ouverte.") == "La q1 reste ouverte."


def test_coverage_reports_why_each_gap_is_a_gap():
    """The limits section must state the real cause, not a generic sentence."""
    thin = coverage_node(_state({"q1": 0}))
    assert thin["gap_reasons"]["q1"] == "thin"

    off_topic = coverage_node(_state({"q1": MIN_SOURCES_PER_SUB_QUESTION}, grounded=False))
    assert off_topic["gap_reasons"]["q1"] == "ungrounded"


def test_limits_section_wording_follows_the_cause():
    from agent.synthesis import _limits_section

    state = {"sub_questions": [{"id": "q1", "text": "Le coefficient X ?"}], "gaps": ["q1"]}
    thin = _limits_section(dict(state, gap_reasons={"q1": "thin"}), [])
    assert "trop peu de sources" in thin

    ungrounded = _limits_section(dict(state, gap_reasons={"q1": "ungrounded"}), [])
    assert "aucune ne mentionne" in ungrounded


def test_limits_section_without_a_recorded_cause():
    """An older state with no reasons must still produce a sane sentence."""
    from agent.synthesis import _limits_section

    state = {"sub_questions": [{"id": "q1", "text": "Une question"}], "gaps": ["q1"]}
    assert "couverture insuffisante" in _limits_section(state, [])
