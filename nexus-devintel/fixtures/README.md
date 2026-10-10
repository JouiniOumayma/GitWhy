# Fixtures — jeu de données factice (mais ancré sur du réel)

Objectif : pouvoir développer le retrieval vectoriel (pgvector), le loader Neo4j
et l'agent **sans attendre l'ingestion réelle** du repository, tout en gardant des
identifiants et des chemins qui correspondent au vrai `httpie/cli`.

## Contenu

| Fichier | Modèle | Volume | Nature |
|---|---|---|---|
| `repositories.json` | `Repository` | 2 | métriques d'audit réelles (`repo_audit.py`) |
| `files.json` | `File` | 94 | **réel** : chemins, LOC, sha256, lignes d'import |
| `commits.json` | `Commit` | 24 | **réel** : SHA, auteurs, dates, numstat |
| `pull_requests.json` | `PullRequest` | 6 | squash-merges réels, texte synthétique |
| `deployments.json` | `Deployment` | 7 | **réel** : tags `3.0.0` → `3.2.4` |
| `incidents.json` | `Incident` | 9 | numéros d'issue **réels**, récit synthétique |
| `evidence.json` | `Evidence` | 10 | synthétique (sortie d'agent simulée) |
| `answers.json` | `Answer` | 4 | synthétique (dont 1 `insufficient_evidence`) |
| `graph_edges.json` | — | 552 | relations dérivées, pour inspection/diff |

## Politique de provenance

Chaque nœud porte un bloc `provenance` (`prov_*` dans Neo4j) qui dit d'où vient le
fait. C'est la règle à ne pas casser :

| `prov_source` | Utilisé pour | Confiance |
|---|---|---|
| `git` | SHAs, auteurs, dates, fichiers modifiés, tags, numstat | 1.0 |
| `derived` | arêtes `:IMPORTS` (analyse AST), `File.impact` (reachability) | 0.9 – 0.95 |
| `synthetic_fixture` | récits d'incidents, corps de PR, sorties d'agent, incidents « stub » | 0.2 – 0.85 |

Autrement dit : **un `File`, un `Commit` ou un `Deployment` de ce jeu de données
est vraie donnée extraite du clone** ; seul le *récit* autour est inventé. Les
incidents « stub » (numéros d'issue référencés par un commit mais non curés) sont
explicitement à `prov_confidence = 0.2` pour que l'agent ne s'appuie pas dessus.

## Les deux histoires à connaître

Les fixtures ne sont pas aléatoires : elles racontent deux scénarios que l'agent
doit savoir traiter.

### 1. Chaîne de root cause (`ans-2026-09-18-0001`)

```
psf/requests#issue-6730  (requests 2.32.3 retire le bundle CA système)
        ▲ CAUSED_BY
httpie/cli#issue-1583 ──CLOSES── 7f03c52  (contournement : pin requests==2.31.0) ──► @3.2.3
        │                    ⚠ aucun PR : lien trouvé dans le message de commit
        └──CLOSES── PR #1596 ──MERGED_INTO──► fd30c4e  (correctif réel) ──► @3.2.4
                        │ MODIFIES
                        └──► httpie/ssl_.py   ← fichier racine
```

Le lien `PR #1596 → #1583` n'est déduit que du titre de la PR
(`confidence = 0.6`, `link_method = commit_message`) : **c'est volontairement le
maillon faible du graphe**, celui que la Semaine 2 remplacera par
`closingIssuesReferences`.

### 2. Rayon d'impact (`ans-2026-09-18-0002`, `0003`)

* `httpie/context.py` : 22 dépendants directs, 37 fichiers atteints, profondeur 3.
* `httpie/cookies.py` : 49 fichiers atteints (cascade maximale des fixtures).
* `httpie/output/ui/__init__.py` et `httpie/cli/requestitems.py` : profondeur 7,
  atteints via des imports paresseux (`httpie/cli/argparser.py:562,577`).
* `ans-2026-09-18-0004` : question volontairement **hors périmètre** (branche
  inconnue) → `insufficient_evidence`, confiance 0.12. Une réponse « je ne sais
  pas » fait partie du jeu de données.

## Limites assumées des fixtures

* **Historique tronqué** : 30 commits sur 1797. Les `:PARENT_OF` s'arrêtent donc
  vite, **sauf dans la fenêtre de déploiement 3.2.3 → 3.2.4** dont la chaîne
  first-parent est ingérée en entier (10 commits) : c'est la seule façon pour
  `(correctif)-[:PARENT_OF*]->(commit tagué)` de répondre « quelle release a livré
  ce correctif ? », qui est la question centrale de la démo RCA. Ajouter une
  fenêtre = une ligne dans `CONNECTED_WINDOW` (`build_fixtures.py`).
* **Une arête `:IMPORTS` par paire de fichiers** (223 arêtes) : les imports
  multiples du même module sont agrégés dans `FileImport.import_lines`, sinon
  `MERGE` en perdrait silencieusement.
* **Aucune issue non référencée** : les incidents existent parce qu'un commit ou
  une PR les cite. Le reste du tracker arrive en Semaine 2.
* **`embedding` volontairement NULL** dans `evidence_embeddings` : aucun modèle
  d'embedding n'est embarqué. Le schéma et les index sont testés sans eux.

## Régénérer

```bash
# depuis la racine du projet ; le clone de la Semaine 0 doit exister
python fixtures/build_fixtures.py --repo-path ../httpie-cli
python scripts/validate_fixtures.py     # schéma + fermeture référentielle
python -m pytest tests/test_fixtures.py # régressions sur les deux histoires
```

Le générateur revalide tout par les modèles Pydantic : impossible de produire un
`file_id` fantôme ou une valeur d'enum inexistante. `graph_edges.json` est
purement dérivé (recalculable), il sert au *diff* quand on re-curate.

## Couche pgvector

`scripts/load_neo4j.py --with-pgvector` insère une ligne `evidence_embeddings` par
nœud `Evidence`, **avec `embedding = NULL`**. C'est voulu : les fixtures ne
portent aucun vecteur, c'est le pipeline d'ingestion (Semaine 2) qui appelle
`sentence-transformers/all-MiniLM-L6-v2` (384 dims) et fera l'`UPDATE`. Le
schéma, les index HNSW et les fonctions `match_code_chunks` /
`hybrid_search_code_chunks` sont en place et testables dès maintenant, notamment
en comparant deux preuves déjà présentes.
