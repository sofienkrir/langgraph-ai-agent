"""Run the evaluation set and write measured results.

    python -m evaluation.run --retrieval        # local corpus only, no LLM, free
    python -m evaluation.run --pipeline         # full graph, costs API calls
    python -m evaluation.run --all

Results land in `evaluation/results/` as JSON plus a Markdown summary. Every
number comes from an executed run; nothing in the report is hand-written.
"""

import argparse
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from agent import documents
from agent.graph import build_research_graph
from .dataset import (
    CORPUS_PDF_NAME,
    CORPUS_PDF_URL,
    OFF_TOPIC_QUERIES,
    PIPELINE_QUESTIONS,
    RETRIEVAL_CASES,
    THRESHOLD_CANDIDATES,
)
from .metrics import citation_stats, coverage_rate, declares_gaps, recall_at_k, summarise

RESULTS_DIR = Path(__file__).resolve().parent / "results"
TOP_K = 4


def ensure_corpus() -> None:
    """Index the reference PDF if the corpus does not already hold it."""
    if CORPUS_PDF_NAME in documents.indexed_files():
        print(f"Corpus déjà indexé : {CORPUS_PDF_NAME}")
        return
    print(f"Téléchargement de {CORPUS_PDF_URL} …")
    request = urllib.request.Request(
        CORPUS_PDF_URL, headers={"User-Agent": "langgraph-research-agent/eval"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    result = documents.index_pdf(data, CORPUS_PDF_NAME)
    print(f"Indexé : {result['pages']} pages, {result['chunks']} passages")


def run_retrieval() -> dict:
    """Measure Recall@k of the local corpus. No LLM involved, so this is free."""
    ensure_corpus()
    rows = []
    for case in RETRIEVAL_CASES:
        hits = documents.search_documents(case["question"], case["id"], k=TOP_K)
        pages = [
            h["locator"].rsplit("p. ", 1)[-1] for h in hits if "p. " in h["locator"]
        ]
        pages = [int(p) for p in pages if p.isdigit()]
        recall = recall_at_k(pages, case["gold_pages"], TOP_K)
        rows.append({
            "id": case["id"],
            "question": case["question"],
            "gold_pages": case["gold_pages"],
            "retrieved_pages": pages,
            "recall_at_k": recall,
            "hit": recall > 0,
            "top_score": round(hits[0]["score"], 3) if hits else None,
        })
        print(f"  {case['id']} recall@{TOP_K}={recall:.2f} pages={pages} gold={case['gold_pages']}")

    return {
        "k": TOP_K,
        "cases": rows,
        "mean_recall_at_k": sum(r["recall_at_k"] for r in rows) / len(rows),
        "hit_rate": sum(1 for r in rows if r["hit"]) / len(rows),
    }


def run_threshold_sweep() -> dict:
    """Measure the precision/recall trade-off of the local relevance floor.

    Raising the floor rejects topically-adjacent passages that are not evidence,
    but eventually discards real ones too. Both sides are measured here so the
    constant in `agent/documents.py` is a decision with numbers behind it.
    """
    ensure_corpus()
    rows = []
    for threshold in THRESHOLD_CANDIDATES:
        recalls = []
        for case in RETRIEVAL_CASES:
            hits = documents.search_documents(
                case["question"], case["id"], k=TOP_K, min_relevance=threshold
            )
            pages = [
                int(h["locator"].rsplit("p. ", 1)[-1])
                for h in hits
                if "p. " in h["locator"] and h["locator"].rsplit("p. ", 1)[-1].isdigit()
            ]
            recalls.append(recall_at_k(pages, case["gold_pages"], TOP_K))

        leaked = sum(
            1
            for query in OFF_TOPIC_QUERIES
            if documents.search_documents(query, "off", k=TOP_K, min_relevance=threshold)
        )
        rows.append({
            "threshold": threshold,
            "mean_recall_at_k": sum(recalls) / len(recalls),
            "off_topic_leak_rate": leaked / len(OFF_TOPIC_QUERIES),
            "off_topic_leaked": leaked,
        })
        print(
            f"  seuil={threshold:.2f} recall@{TOP_K}={rows[-1]['mean_recall_at_k']:.2f} "
            f"fuite_hors_sujet={leaked}/{len(OFF_TOPIC_QUERIES)}"
        )

    # Balance the two sides with a harmonic mean. Ranking by leak rate alone
    # would crown the threshold that retrieves nothing at all: zero leak, zero
    # recall, a perfect score for a useless retriever.
    for row in rows:
        recall = row["mean_recall_at_k"]
        clean = 1.0 - row["off_topic_leak_rate"]
        row["balance"] = (
            2 * recall * clean / (recall + clean) if (recall + clean) else 0.0
        )
    best = max(rows, key=lambda r: r["balance"])
    return {
        "k": TOP_K,
        "current_threshold": documents.MIN_RELEVANCE,
        "rows": rows,
        "best": best,
    }


def run_grounding() -> dict:
    """Measure lexical grounding on real retrievals, legitimate vs unanswerable.

    Plans and searches but skips synthesis: the question here is whether the
    retrieved sources mention what was asked, which the report cannot change.
    """
    from agent.coverage import term_grounding
    from agent.planner import plan_node
    from agent.search import search_node

    rows = []
    for case in PIPELINE_QUESTIONS:
        planned = plan_node({"question": case["question"]})
        sub_questions = planned["sub_questions"]
        found = search_node({
            "sub_questions": sub_questions, "gaps": [], "search_round": 0, "sources": [],
        })
        by_question = {}
        for source in found["sources"]:
            by_question.setdefault(source["sub_question_id"], []).append(source)

        scores = [
            round(term_grounding(sq["text"], by_question.get(sq["id"], [])), 3)
            for sq in sub_questions
        ]
        rows.append({
            "id": case["id"],
            "answerable": not case["expect_gaps"],
            "question": case["question"][:70],
            "scores": scores,
            "min": min(scores) if scores else 0.0,
            "mean": sum(scores) / len(scores) if scores else 0.0,
        })
        print(f"  {case['id']} ancrage min={rows[-1]['min']:.2f} moyenne={rows[-1]['mean']:.2f} {scores}")

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    # Compare like with like: the weakest answerable question against the
    # strongest unanswerable one, both on the same statistic.
    floor = min((r["mean"] for r in answerable), default=0.0)
    ceiling = max((r["mean"] for r in unanswerable), default=0.0)
    return {
        "rows": rows,
        "lowest_answerable_grounding": round(floor, 3),
        "highest_unanswerable_grounding": round(ceiling, 3),
        "separable": floor > ceiling,
        "current_threshold": __import__("agent.coverage", fromlist=["x"]).MIN_TERM_GROUNDING,
    }


def run_pipeline() -> dict:
    """Run the whole graph per question and measure what the run produced."""
    graph = build_research_graph()
    rows = []
    for case in PIPELINE_QUESTIONS:
        print(f"  {case['id']} …", flush=True)
        started = time.time()
        try:
            state = graph.invoke({"question": case["question"]})
        except Exception as exc:
            rows.append({"id": case["id"], "question": case["question"], "error": str(exc)})
            print(f"    échec : {exc}")
            continue

        elapsed = time.time() - started
        sub_questions = state.get("sub_questions", [])
        gaps = state.get("gaps", [])
        sources = state.get("sources", [])
        report = state.get("report", "")
        gap_texts = [sq["text"] for sq in sub_questions if sq["id"] in gaps]

        row = {
            "id": case["id"],
            "question": case["question"],
            "expect_gaps": case["expect_gaps"],
            "sub_questions": len(sub_questions),
            "gaps": len(gaps),
            "had_gaps": bool(gaps),
            "gap_expectation_met": bool(gaps) == case["expect_gaps"],
            "coverage_rate": coverage_rate([s["id"] for s in sub_questions], gaps),
            "source_count": len(sources),
            "document_sources": sum(1 for s in sources if s.get("kind") == "document"),
            "web_sources": sum(1 for s in sources if s.get("kind") == "web"),
            "search_rounds": state.get("search_round", 0),
            "citations": citation_stats(report, len(sources)),
            "declares_gaps": declares_gaps(report, gap_texts),
            "latency_s": round(elapsed, 1),
        }
        rows.append(row)
        print(
            f"    couverture={row['coverage_rate']:.2f} sources={row['source_count']} "
            f"tours={row['search_rounds']} citations_valides="
            f"{row['citations']['validity_rate']:.2f} {elapsed:.0f}s"
        )

    return {"runs": rows, "summary": summarise(rows)}


def to_markdown(payload: dict) -> str:
    """Render the measured results. Only numbers that were produced by a run."""
    lines = [
        "# Résultats d'évaluation",
        "",
        f"Exécuté le {payload['run_at']} · fournisseur `{payload['provider']}`",
        "",
    ]

    retrieval = payload.get("retrieval")
    if retrieval:
        lines += [
            f"## Recall du retrieval local (k={retrieval['k']})",
            "",
            "| Cas | Recall@k | Pages attendues | Pages trouvées |",
            "|---|---|---|---|",
        ]
        for row in retrieval["cases"]:
            lines.append(
                f"| {row['id']} | {row['recall_at_k']:.2f} | "
                f"{row['gold_pages']} | {row['retrieved_pages']} |"
            )
        lines += [
            "",
            f"**Recall@{retrieval['k']} moyen : {retrieval['mean_recall_at_k']:.2f}** · "
            f"taux de réussite (≥1 page correcte) : {retrieval['hit_rate']:.2f}",
            "",
        ]

    sweep = payload.get("threshold_sweep")
    if sweep:
        lines += [
            "## Seuil de pertinence du corpus local",
            "",
            f"Seuil en vigueur : `{sweep['current_threshold']}`. "
            "Monter le seuil rejette les passages seulement proches du sujet, "
            "au prix du rappel.",
            "",
            f"| Seuil | Recall@{sweep['k']} | Requêtes hors sujet renvoyant un passage | Équilibre |",
            "|---|---|---|---|",
        ]
        for row in sweep["rows"]:
            mark = " ←" if row["threshold"] == sweep["best"]["threshold"] else ""
            lines.append(
                f"| {row['threshold']:.2f}{mark} | {row['mean_recall_at_k']:.2f} | "
                f"{row['off_topic_leaked']} ({row['off_topic_leak_rate']:.0%}) | "
                f"{row['balance']:.2f} |"
            )
        lines += ["", f"Meilleur compromis mesuré : **{sweep['best']['threshold']:.2f}**", ""]

    grounding = payload.get("grounding")
    if grounding:
        lines += [
            "## Ancrage lexical",
            "",
            "Part des termes distinctifs d'une sous-question — noms propres, acronymes, "
            "codes, années — qui apparaissent dans ses sources. Compter les sources ne "
            "suffit pas : une question sur un sujet inventé attire des pages plausibles "
            "qui passent le seuil de pertinence. Mesurer *tous* les mots porteurs ne "
            "séparait rien ; seuls les termes distinctifs y parviennent.",
            "",
            "| Question | Répondable | Ancrage min | Ancrage moyen |",
            "|---|---|---|---|",
        ]
        for row in grounding["rows"]:
            lines.append(
                f"| {row['id']} | {'oui' if row['answerable'] else 'non'} | "
                f"{row['min']:.2f} | {row['mean']:.2f} |"
            )
        lines += [
            "",
            f"Plus faible ancrage d'une question répondable : "
            f"**{grounding['lowest_answerable_grounding']:.2f}** · "
            f"plus fort ancrage de la question sans réponse : "
            f"**{grounding['highest_unanswerable_grounding']:.2f}** · "
            f"séparables : **{grounding['separable']}** · "
            f"seuil en vigueur : `{grounding['current_threshold']}`",
            "",
        ]

    pipeline = payload.get("pipeline")
    if pipeline:
        summary = pipeline["summary"]
        lines += [
            "## Chaîne complète",
            "",
            "| Métrique | Valeur |",
            "|---|---|",
            f"| Questions évaluées | {summary['questions']} |",
            f"| Exécutions en échec | {summary['failed_runs']} |",
            f"| Taux de couverture moyen | {summary['mean_coverage_rate']:.2f} |",
            f"| Citations valides (moyenne) | {summary['mean_citation_validity']:.2f} |",
            f"| Rapports contenant une citation invalide | {summary['reports_with_invalid_citations']} |",
            f"| Tours de recherche (moyenne) | {summary['mean_search_rounds']:.2f} |",
            f"| Sources par rapport (moyenne) | {summary['mean_sources']:.1f} |",
            f"| Latence moyenne | {summary['mean_latency_s']:.1f} s |",
            f"| Lacunes systématiquement déclarées | {summary['gap_declaration_holds']} |",
            "",
            "| Question | Couverture | Sources (doc/web) | Tours | Citations valides | Lacune attendue | Latence |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in pipeline["runs"]:
            if row.get("error"):
                lines.append(f"| {row['id']} | échec : {row['error'][:60]} | | | | | |")
                continue
            lines.append(
                f"| {row['id']} | {row['coverage_rate']:.2f} | "
                f"{row['document_sources']}/{row['web_sources']} | {row['search_rounds']} | "
                f"{row['citations']['validity_rate']:.2f} | "
                f"{'✅' if row['gap_expectation_met'] else '❌'} | {row['latency_s']}s |"
            )
        lines.append("")

    lines += [
        "## Non mesuré",
        "",
        "La **fidélité** — chaque affirmation est-elle réellement soutenue par la source "
        "qu'elle cite — n'est pas mesurée ici. L'établir demanderait une annotation humaine "
        "ou un LLM juge, c'est-à-dire précisément le type de score non vérifiable que ce "
        "projet évite ailleurs. Les chiffres ci-dessus mesurent la couverture, la validité "
        "formelle des citations et le comportement face aux lacunes, pas la véracité du texte.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Harnais d'évaluation de l'agent.")
    parser.add_argument("--retrieval", action="store_true", help="Recall@k du corpus local")
    parser.add_argument("--pipeline", action="store_true", help="chaîne complète (coûte des appels API)")
    parser.add_argument("--sweep", action="store_true", help="balayage du seuil de pertinence local")
    parser.add_argument("--grounding", action="store_true", help="mesure de l'ancrage lexical")
    parser.add_argument("--all", action="store_true", help="tout")
    args = parser.parse_args()

    do_retrieval = args.retrieval or args.all
    do_pipeline = args.pipeline or args.all
    do_sweep = args.sweep or args.all
    do_grounding = args.grounding or args.all
    if not (do_retrieval or do_pipeline or do_sweep or do_grounding):
        parser.error("Choisis --retrieval, --sweep, --grounding, --pipeline ou --all.")

    from agent.llm import active_provider

    payload = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provider": active_provider(),
    }

    if do_retrieval:
        print("== Retrieval local ==")
        payload["retrieval"] = run_retrieval()
    if do_sweep:
        print("== Balayage du seuil local ==")
        payload["threshold_sweep"] = run_threshold_sweep()
    if do_grounding:
        print("== Ancrage lexical ==")
        payload["grounding"] = run_grounding()
    if do_pipeline:
        print("== Chaîne complète ==")
        payload["pipeline"] = run_pipeline()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = payload["run_at"].replace(":", "").replace("-", "")
    json_path = RESULTS_DIR / f"{stamp}_{payload['provider']}.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    markdown = to_markdown(payload)
    (RESULTS_DIR / "latest.md").write_text(markdown, encoding="utf-8")

    print()
    print(markdown)
    print(f"\nÉcrit : {json_path}")


if __name__ == "__main__":
    main()
