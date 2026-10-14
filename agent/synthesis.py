"""Synthesis node: write the report, cite the sources, declare the gaps."""

import re

from .llm import get_llm
from .state import ResearchState

SYNTHESIS_PROMPT = """Tu rédiges un rapport de recherche à partir de sources numérotées.

Règles absolues :
- N'affirme rien qui ne vienne pas des sources ci-dessous. Aucune connaissance externe.
- Cite systématiquement avec [n], en utilisant uniquement les numéros fournis.
- Si les sources se contredisent, dis-le explicitement plutôt que de trancher.
- Les sources marquées « document local » viennent des PDF fournis par l'utilisateur,
  celles marquées « web » d'une recherche en ligne. Si les deux divergent sur un point,
  signale-le en nommant les deux numéros.
- Si une sous-question est mal couverte, écris-le au lieu de combler le vide.
- N'écris PAS de section sur les limites : elle est ajoutée automatiquement après toi.
- Structure : une introduction courte, puis une section par sous-question, avec des
  titres de niveau 2 (##). Rédige dans la langue de la question.
- Les identifiants q1, q2, q3… sont internes au système. Ne les écris jamais dans le
  rapport : chaque titre doit nommer le sujet traité, pas son numéro.

Question : {question}

Sous-questions :
{sub_questions}

Sources :
{sources}"""


def _format_sources(sources) -> tuple[str, dict]:
    """Number the sources once; the same numbering is reused for display."""
    lines = []
    numbering = {}
    for index, source in enumerate(sources, start=1):
        numbering[index] = source
        origin = "document local" if source.get("kind") == "document" else "web"
        lines.append(
            f"[{index}] ({source['sub_question_id']}, {origin}) {source['title']}\n"
            f"    {source.get('locator') or source['url']}\n"
            f"    {source['snippet'][:700]}"
        )
    return "\n\n".join(lines), numbering


def _limits_section(state: ResearchState, invalid_citations: list[int]) -> str:
    """Build the limits section from measured facts, not from the model's mood."""
    by_id = {sq["id"]: sq["text"] for sq in state.get("sub_questions", [])}
    entries = []

    reasons = state.get("gap_reasons", {})
    explanation = {
        "thin": "trop peu de sources pertinentes trouvées",
        "ungrounded": "des sources ont été trouvées, mais aucune ne mentionne "
                      "ce que la question nomme",
    }
    for gap_id in state.get("gaps", []):
        why = explanation.get(reasons.get(gap_id), "couverture insuffisante")
        entries.append(
            f"- Sous-question insuffisamment documentée : "
            f"« {by_id.get(gap_id, gap_id)} » — {why}."
        )
    if invalid_citations:
        listed = ", ".join(f"[{n}]" for n in sorted(invalid_citations))
        entries.append(
            f"- Citations invalides détectées et à ignorer : {listed}. "
            "Elles ne correspondent à aucune source récupérée."
        )
    if not entries:
        return (
            "\n\n## Limites\n\n"
            "- Toutes les sous-questions ont atteint le seuil de couverture. "
            "Ce rapport reste limité aux sources listées ci-dessous."
        )
    return "\n\n## Limites\n\n" + "\n".join(entries)


_HEADING_ID_PATTERN = re.compile(
    r"^(#{1,6}\s*)\(?q\d+\)?\s*[-–—:.)]?\s*", re.MULTILINE | re.IGNORECASE
)


def _strip_internal_ids(body: str) -> str:
    """Remove sub-question ids the model echoed into its headings.

    The prompt asks it not to, but a published report must not depend on the
    model remembering. `## q1 - Définition` becomes `## Définition`.
    """
    return _HEADING_ID_PATTERN.sub(r"\1", body)


def _invalid_citations(report: str, max_number: int) -> list[int]:
    """Any [n] outside the real source range is a fabricated citation."""
    cited = {int(n) for n in re.findall(r"\[(\d{1,3})\]", report)}
    return [n for n in cited if n < 1 or n > max_number]


def synthesize_node(state: ResearchState) -> dict:
    """Produce the final report. Citations are validated, gaps are appended."""
    sources = state.get("sources", [])
    question = state["question"]

    if not sources:
        return {
            "report": (
                f"# {question}\n\n"
                "Aucune source exploitable n'a été récupérée. Aucun rapport ne peut être "
                "produit sans inventer son contenu."
                + _limits_section(state, [])
            ),
            "trace": ["Synthèse abandonnée : aucune source."],
        }

    sources_block, numbering = _format_sources(sources)
    sub_block = "\n".join(
        f"- ({sq['id']}) {sq['text']}" for sq in state.get("sub_questions", [])
    )

    llm = get_llm("synthesizer")
    prompt = SYNTHESIS_PROMPT.format(
        question=question, sub_questions=sub_block, sources=sources_block
    )
    try:
        body = _strip_internal_ids(str(llm.invoke(prompt).content).strip())
    except Exception as exc:
        return {
            "report": f"# {question}\n\nLa synthèse a échoué : {exc}",
            "trace": [f"Synthèse en échec : {exc}"],
        }

    invalid = _invalid_citations(body, len(numbering))
    report = f"# {question}\n\n{body}" + _limits_section(state, invalid)

    note = f"Rapport rédigé à partir de {len(sources)} sources."
    if invalid:
        note += f" {len(invalid)} citation(s) invalide(s) signalée(s)."
    return {"report": report, "trace": [note]}
