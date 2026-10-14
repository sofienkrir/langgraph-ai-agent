"""Tests for the evaluation metrics.

A metric that is wrong is worse than no metric: it produces a confident number
nobody checks. These run offline and cover the edge cases that would silently
flatter the results.
"""

import pytest

from evaluation.metrics import (
    citation_stats,
    coverage_rate,
    declares_gaps,
    hit_at_k,
    mean,
    recall_at_k,
    summarise,
)


def test_recall_counts_only_gold_pages_inside_k():
    # Page 7 is gold but ranked 5th, so it must not count at k=4.
    assert recall_at_k([1, 2, 3, 4, 7], [4, 7], k=4) == 0.5


def test_recall_is_one_when_all_gold_pages_retrieved():
    assert recall_at_k([18, 17], [18, 17], k=4) == 1.0


def test_recall_ignores_duplicate_retrievals():
    """Two chunks of the same page must not inflate recall above its ceiling."""
    assert recall_at_k([20, 20, 20, 20], [20, 11], k=4) == 0.5


def test_recall_is_zero_when_nothing_matches():
    assert recall_at_k([8, 18], [19], k=4) == 0.0


def test_recall_on_empty_retrieval():
    assert recall_at_k([], [19], k=4) == 0.0


def test_empty_gold_set_is_rejected():
    """An unlabelled case would otherwise score 0/0 and quietly pass."""
    with pytest.raises(ValueError):
        recall_at_k([1, 2], [], k=4)


def test_hit_is_lenient_where_recall_is_not():
    assert hit_at_k([12, 99], [12, 13], k=4) is True
    assert recall_at_k([12, 99], [12, 13], k=4) == 0.5


def test_coverage_rate():
    assert coverage_rate(["q1", "q2", "q3", "q4"], ["q3"]) == 0.75
    assert coverage_rate(["q1", "q2"], []) == 1.0
    assert coverage_rate([], []) == 0.0


def test_coverage_rate_ignores_duplicate_gaps():
    assert coverage_rate(["q1", "q2"], ["q2", "q2"]) == 0.5


def test_citation_stats_separates_valid_from_fabricated():
    stats = citation_stats("texte [1] puis [2] et enfin [9]", source_count=3)
    assert stats["distinct_cited"] == 3
    assert stats["valid"] == 2
    assert stats["invalid_numbers"] == [9]
    assert stats["validity_rate"] == pytest.approx(2 / 3)


def test_citation_stats_counts_distinct_not_repeats():
    """Citing [1] five times is not five sources."""
    stats = citation_stats("[1] [1] [1] [1] [1]", source_count=4)
    assert stats["markers"] == 5
    assert stats["distinct_cited"] == 1
    assert stats["source_use_rate"] == 0.25


def test_citation_stats_on_report_without_citations():
    stats = citation_stats("aucune citation ici", source_count=5)
    assert stats["distinct_cited"] == 0
    assert stats["validity_rate"] == 0.0


def test_citation_stats_without_sources_never_divides_by_zero():
    assert citation_stats("[1]", source_count=0)["source_use_rate"] == 0.0


def test_gap_declaration_requires_the_limits_section():
    assert declares_gaps("corps du rapport", ["une sous-question"]) is False


def test_gap_declaration_detects_the_named_question():
    report = "corps\n\n## Limites\n\n- Sous-question insuffisamment documentée : « le rendement »."
    assert declares_gaps(report, ["le rendement"]) is True
    assert declares_gaps(report, ["un autre sujet absent"]) is False


def test_no_gaps_declares_trivially():
    assert declares_gaps("rapport sans section limites", []) is True


def test_mean_skips_missing_values():
    assert mean([1.0, None, 3.0]) == 2.0
    assert mean([]) == 0.0


def test_summarise_excludes_failed_runs_from_averages():
    runs = [
        {
            "coverage_rate": 1.0,
            "citations": {"validity_rate": 1.0, "invalid": 0},
            "search_rounds": 1,
            "source_count": 10,
            "latency_s": 40,
            "declares_gaps": True,
        },
        {"error": "boom"},
    ]
    summary = summarise(runs)
    assert summary["questions"] == 2
    assert summary["failed_runs"] == 1
    # The failed run must not drag the average toward zero.
    assert summary["mean_coverage_rate"] == 1.0


def test_summarise_flags_a_broken_gap_guarantee():
    runs = [{
        "coverage_rate": 0.5,
        "citations": {"validity_rate": 1.0, "invalid": 0},
        "search_rounds": 2,
        "source_count": 4,
        "latency_s": 10,
        "declares_gaps": False,
    }]
    assert summarise(runs)["gap_declaration_holds"] is False
