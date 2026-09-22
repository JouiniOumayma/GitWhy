# GitWhy — Knowledge Graph de veille technologique

Knowledge Graph **Neo4j** construit sur un vrai dépôt GitHub (`httpie/cli`)
pour faire de la **Change Impact Analysis** et du **Root Cause Analysis** avec
un agent qui répond avec des preuves (`EvidencePath`) et un score de confiance.

> Implémentation : dossier [`nexus-devintel/`](nexus-devintel/) —
> voir son [README détaillé](nexus-devintel/README.md).

## Structure du dépôt

| Élément | Rôle |
|---|---|
| `nexus-devintel/` | Le projet : contrats Pydantic, schéma Neo4j/pgvector, connecteur GitHub, retrieval Change Impact + Root Cause hybride, 140 tests |
| `repo_audit.py` | Script d'audit d'un dépôt git (commits, graphe d'imports, couplages) — a servi à choisir le dépôt de démo |
| `audit_click.json` | Audit de `psf/click` (3362 commits, 90 fichiers py) |
| `audit_httpie.json` | Audit de `httpie/cli` — **retenu** : 1797 commits, 133 fichiers, blast radius moyen ~22 fichiers, profondeur 7 |
| `audit_requests.json` | Audit de `psf/requests` (6494 commits, 37 fichiers py) |
| `click/`, `httpie-cli/`, `requests/` | Clones locaux utilisés pour les audits (non versionnés, exclus du dépôt) |

## Démarrage rapide

```bash
cd nexus-devintel
cp .env.example .env

docker compose up -d          # Neo4j 5.26 + PostgreSQL/pgvector
docker compose ps             # attendre "healthy"

python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements-dev.txt

python scripts/load_neo4j.py --apply-schema       # schéma + fixtures dans Neo4j
python -m pytest                                  # 140 tests
python scripts/impact.py "httpie/cli::httpie/context.py"   # Change Impact -> EvidencePath
```

Neo4j Browser : http://localhost:7474 (`neo4j` / `user` en dev).

## État du projet

- ✅ **Phase 1 terminée** : graphe peuplé et interrogeable, connecteur GitHub
  read-only + webhook (écriture idempotente dans Neo4j), EvidencePath
  dynamique produit par traversée réelle (22 dépendants directs / 37 transitifs
  / profondeur 7 pour `httpie/context.py`, conforme à l'audit).
- ✅ **Phase 2 (Personne B) terminée** : Root Cause `Incident → PR → Commit →
  File → Deployment` (requête Cypher bornée + score `produit(confidences) ×
  0.85^distance`), retrieval hybride graphe + pgvector (RRF k=60,
  `lexical + HNSW cosinus`), embedder réel par défaut
  `sentence-transformers/all-MiniLM-L6-v2` (dim 384, ~90 Mo, indexation réelle
  des 774 chunks en ~1 min, `code_chunks.embedding_model` tracé par ligne),
  repli offline stdlib `hashing-token-384` (`--hashing-embeddings`), aucun mock,
  enrichissement GraphQL `closingIssuesReferences`
  (195 arêtes `CLOSES` confidence 1.0 vs 5 via git log), MCP GitHub read-only
  5 outils, 17 tests d'attaque `test_security.py`.
