"""Planner node: break the question into searchable sub-questions."""

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import ResearchState

MAX_SUB_QUESTIONS = 6

PLANNER_PROMPT = """Tu décomposes une question de recherche en sous-questions autonomes.

Règles :
- Entre 2 et {max_sub} sous-questions, chacune vérifiable par une recherche documentaire.
- Chaque sous-question doit être autosuffisante : elle sera envoyée seule à un moteur
  de recherche, sans le reste de la conversation.
- Reste strictement dans le périmètre de la question posée. N'ajoute pas de sujets
  connexes que l'utilisateur n'a pas demandés.
- Pas de sous-question qui demande une opinion ou une prédiction invérifiable.
- Rédige les sous-questions dans la langue de la question.

Question : {question}"""


class Plan(BaseModel):
    """Structured planner output."""

    sub_questions: list[str] = Field(
        description="Les sous-questions de recherche, autosuffisantes.",
        min_length=1,
        max_length=MAX_SUB_QUESTIONS,
    )


def plan_node(state: ResearchState) -> dict:
    """Decompose the question. Falls back to the raw question if the model fails."""
    question = state["question"]
    llm = get_llm("planner").with_structured_output(Plan)

    prompt = PLANNER_PROMPT.format(question=question, max_sub=MAX_SUB_QUESTIONS)
    try:
        plan = llm.invoke(prompt)
        texts = [q.strip() for q in plan.sub_questions if q and q.strip()]
    except Exception as exc:
        # A failed decomposition should degrade to a single-question search,
        # not sink the whole run.
        return {
            "sub_questions": [{"id": "q1", "text": question}],
            "search_round": 0,
            "trace": [f"Planification en échec ({exc}); recherche sur la question brute."],
        }

    if not texts:
        texts = [question]

    sub_questions = [
        {"id": f"q{index}", "text": text}
        for index, text in enumerate(texts[:MAX_SUB_QUESTIONS], start=1)
    ]
    return {
        "sub_questions": sub_questions,
        "search_round": 0,
        "trace": [f"Plan : {len(sub_questions)} sous-questions."],
    }
