"""The fixed evaluation set.

RETRIEVAL_CASES check the local corpus alone: each question has gold pages that
were chosen by reading the source PDF, so Recall@k is a real measurement rather
than a model's opinion. They are deliberately written in French against an
English paper, because cross-lingual retrieval is the risky part of the design.

PIPELINE_QUESTIONS run the whole graph. The last one is expected to find almost
nothing: it exists to check that the agent declares the gap instead of inventing
an answer, which is the behaviour the project is built around.
"""

# The paper the retrieval cases are labelled against.
CORPUS_PDF_URL = "https://arxiv.org/pdf/2310.17032v3"
CORPUS_PDF_NAME = "solar_forecasting.pdf"

RETRIEVAL_CASES = [
    {
        "id": "r1",
        "question": "Quel jeu de données a été utilisé pour entraîner les modèles ?",
        "gold_pages": [6, 7],
    },
    {
        "id": "r2",
        "question": "Quelle bibliothèque de calcul quantique est utilisée pour implémenter le modèle ?",
        "gold_pages": [8],
    },
    {
        "id": "r3",
        "question": "Quelle est la complexité algorithmique du prétraitement classique des données ?",
        "gold_pages": [10],
    },
    {
        "id": "r4",
        "question": "À quelle vitesse le modèle QLSTM converge-t-il pendant l'entraînement ?",
        "gold_pages": [12, 13],
    },
    {
        "id": "r5",
        "question": "Quel est le temps d'évaluation par époque des deux modèles ?",
        "gold_pages": [14, 15],
    },
    {
        "id": "r6",
        "question": "Quelles sont les équations mathématiques de la cellule LSTM classique ?",
        "gold_pages": [18],
    },
    {
        "id": "r7",
        "question": "Comment est structuré le circuit quantique variationnel du QLSTM ?",
        "gold_pages": [19],
    },
    {
        "id": "r8",
        "question": "Quels tests statistiques et quelle p-value sont utilisés pour comparer les modèles ?",
        "gold_pages": [20],
    },
]

PIPELINE_QUESTIONS = [
    {
        "id": "p1",
        "question": "Quelles sont les principales méthodes de prévision de la production solaire par IA ?",
        "expect_gaps": False,
    },
    {
        "id": "p2",
        "question": "Les réseaux LSTM quantiques apportent-ils un gain réel sur la prévision solaire ?",
        "expect_gaps": False,
    },
    {
        "id": "p3",
        "question": "Comment la maintenance prédictive des éoliennes utilise-t-elle l'apprentissage automatique ?",
        "expect_gaps": False,
    },
    {
        "id": "p4",
        # Deliberately unanswerable: a fabricated metric that no source defines.
        # The agent must declare the gap rather than invent a definition.
        "question": "Quelle est la valeur du coefficient de Vandermeer-Oyelaran pour les centrales photovoltaïques tunisiennes certifiées BX-14 en 2031 ?",
        "expect_gaps": True,
    },
]


# Queries the reference corpus genuinely cannot answer. A passage returned for
# any of these is a false positive: the paper is about solar forecasting with
# quantum LSTMs and says nothing about these subjects. They exist because
# embedding similarity measures topical proximity, not evidential relevance -
# "renewable energy + machine learning" passages score deceptively well against
# any renewable-energy question.
OFF_TOPIC_QUERIES = [
    "Comment la maintenance prédictive des éoliennes utilise-t-elle l'apprentissage automatique ?",
    "Quels capteurs de vibration équipent les nacelles d'éoliennes ?",
    "Comment dimensionne-t-on un parc de batteries lithium-fer-phosphate ?",
    "Quelle réglementation encadre le raccordement au réseau en France ?",
    "Quels sont les rendements des cellules photovoltaïques en pérovskite ?",
    "Quelle est la recette de la tarte aux pommes ?",
]

# Thresholds explored by `python -m evaluation.run --sweep`.
THRESHOLD_CANDIDATES = [0.25, 0.35, 0.45, 0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70]
