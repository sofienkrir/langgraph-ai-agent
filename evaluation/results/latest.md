# Résultats d'évaluation

Exécuté le 2026-10-14T14:50:30+00:00 · fournisseur `groq`

## Recall du retrieval local (k=4)

| Cas | Recall@k | Pages attendues | Pages trouvées |
|---|---|---|---|
| r1 | 0.50 | [6, 7] | [7, 7, 7] |
| r2 | 1.00 | [8] | [16, 18, 8, 9] |
| r3 | 1.00 | [10] | [10, 9, 10] |
| r4 | 0.50 | [12, 13] | [11, 13, 10, 8] |
| r5 | 0.50 | [14, 15] | [14, 20, 14] |
| r6 | 1.00 | [18] | [17, 18, 18, 8] |
| r7 | 0.00 | [19] | [8, 18, 8, 8] |
| r8 | 1.00 | [20] | [20, 20, 11] |

**Recall@4 moyen : 0.69** · taux de réussite (≥1 page correcte) : 0.88

## Seuil de pertinence du corpus local

Seuil en vigueur : `0.5`. Monter le seuil rejette les passages seulement proches du sujet, au prix du rappel.

| Seuil | Recall@4 | Requêtes hors sujet renvoyant un passage | Équilibre |
|---|---|---|---|
| 0.25 | 0.69 | 5 (83%) | 0.27 |
| 0.35 | 0.69 | 5 (83%) | 0.27 |
| 0.45 | 0.69 | 3 (50%) | 0.58 |
| 0.50 ← | 0.69 | 2 (33%) | 0.68 |
| 0.55 | 0.62 | 2 (33%) | 0.65 |
| 0.58 | 0.50 | 1 (17%) | 0.62 |
| 0.60 | 0.50 | 1 (17%) | 0.62 |
| 0.62 | 0.38 | 1 (17%) | 0.52 |
| 0.65 | 0.19 | 0 (0%) | 0.32 |
| 0.70 | 0.00 | 0 (0%) | 0.00 |

Meilleur compromis mesuré : **0.50**

## Ancrage lexical

Part des mots porteurs d'une sous-question qui apparaissent dans ses sources. Compter les sources ne suffit pas : une question sur un sujet inventé attire des pages plausibles qui passent le seuil de pertinence.

| Question | Répondable | Ancrage min | Ancrage moyen |
|---|---|---|---|
| p1 | oui | 0.00 | 0.80 |
| p2 | oui | 0.50 | 0.83 |
| p3 | oui | 1.00 | 1.00 |
| p4 | non | 0.00 | 0.25 |

Plus faible ancrage d'une question répondable : **0.80** · plus fort ancrage de la question sans réponse : **0.25** · séparables : **True** · seuil en vigueur : `0.5`

## Chaîne complète

| Métrique | Valeur |
|---|---|
| Questions évaluées | 4 |
| Exécutions en échec | 0 |
| Taux de couverture moyen | 0.88 |
| Citations valides (moyenne) | 1.00 |
| Rapports contenant une citation invalide | 0 |
| Tours de recherche (moyenne) | 1.50 |
| Sources par rapport (moyenne) | 25.8 |
| Latence moyenne | 40.2 s |
| Lacunes systématiquement déclarées | True |

| Question | Couverture | Sources (doc/web) | Tours | Citations valides | Lacune attendue | Latence |
|---|---|---|---|---|---|---|
| p1 | 1.00 | 13/12 | 2 | 1.00 | ✅ | 15.0s |
| p2 | 1.00 | 17/8 | 1 | 1.00 | ✅ | 35.0s |
| p3 | 1.00 | 13/15 | 1 | 1.00 | ✅ | 59.9s |
| p4 | 0.50 | 7/18 | 2 | 1.00 | ✅ | 50.9s |

## Non mesuré

La **fidélité** — chaque affirmation est-elle réellement soutenue par la source qu'elle cite — n'est pas mesurée ici. L'établir demanderait une annotation humaine ou un LLM juge, c'est-à-dire précisément le type de score non vérifiable que ce projet évite ailleurs. Les chiffres ci-dessus mesurent la couverture, la validité formelle des citations et le comportement face aux lacunes, pas la véracité du texte.
