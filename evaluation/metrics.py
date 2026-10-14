"""Metric functions for the evaluation harness.

Every function here is pure and unit-tested. Each one reports something that
can be checked against the run's own artefacts - retrieved pages, the gap list,
the citation markers in the report. Nothing is scored by a model.

What this file deliberately does NOT measure: whether each sentence is actually
supported by the source it cites. That is faithfulness, and establishing it
needs either human annotation or an LLM judge, which would reintroduce exactly
the unmeasured number this project avoids. See the "Not measured" section of
the report that `run.py` writes.
"""

import re

CITATION_PATTERN = re.compile(r"\[(\d{1,3})\]")


def recall_at_k(retrieved_pages, gold_pages, k: int | None = None) -> float:
    """Share of the gold pages that appear in the top-k retrieved pages.

    `retrieved_pages` must be in rank order. An empty gold set is a bad test
    case rather than a free pass, so it raises.
    """
    gold = set(gold_pages)
    if not gold:
        raise ValueError("Un cas de test doit déclarer au moins une page de référence.")
    top = list(retrieved_pages)[:k] if k else list(retrieved_pages)
    return len(gold & set(top)) / len(gold)


def hit_at_k(retrieved_pages, gold_pages, k: int | None = None) -> bool:
    """Whether at least one gold page was retrieved. Recall's lenient sibling."""
    return recall_at_k(retrieved_pages, gold_pages, k) > 0


def coverage_rate(sub_question_ids, gaps) -> float:
    """Share of sub-questions that reached the source threshold."""
    total = len(list(sub_question_ids))
    if not total:
        return 0.0
    return (total - len(set(gaps))) / total


def citation_stats(report: str, source_count: int) -> dict:
    """Count citation markers and how many point at a real source.

    `distinct_cited` is the useful denominator: a report that cites [1] twenty
    times is not better sourced than one that cites it once.
    """
    markers = [int(n) for n in CITATION_PATTERN.findall(report)]
    distinct = sorted(set(markers))
    valid = [n for n in distinct if 1 <= n <= source_count]
    invalid = [n for n in distinct if n not in valid]
    return {
        "markers": len(markers),
        "distinct_cited": len(distinct),
        "valid": len(valid),
        "invalid": len(invalid),
        "invalid_numbers": invalid,
        "validity_rate": (len(valid) / len(distinct)) if distinct else 0.0,
        "source_use_rate": (len(valid) / source_count) if source_count else 0.0,
    }


def declares_gaps(report: str, gap_texts) -> bool:
    """Whether the report's limitations section names every uncovered question.

    The limits section is built in Python, so this asserts the guarantee still
    holds end to end rather than trusting the model to have been humble.
    """
    gap_texts = list(gap_texts)
    if not gap_texts:
        return True
    if "## Limites" not in report:
        return False
    section = report.split("## Limites", 1)[1]
    return all(text[:60] in section for text in gap_texts)


def mean(values) -> float:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0


def summarise(runs) -> dict:
    """Aggregate per-question pipeline results into headline numbers."""
    runs = list(runs)
    ok = [r for r in runs if not r.get("error")]
    return {
        "questions": len(runs),
        "failed_runs": len(runs) - len(ok),
        "mean_coverage_rate": mean(r["coverage_rate"] for r in ok),
        "mean_citation_validity": mean(r["citations"]["validity_rate"] for r in ok),
        "reports_with_invalid_citations": sum(
            1 for r in ok if r["citations"]["invalid"]
        ),
        "mean_search_rounds": mean(r["search_rounds"] for r in ok),
        "mean_sources": mean(r["source_count"] for r in ok),
        "mean_latency_s": mean(r["latency_s"] for r in ok),
        "gap_declaration_holds": all(r["declares_gaps"] for r in ok),
    }
