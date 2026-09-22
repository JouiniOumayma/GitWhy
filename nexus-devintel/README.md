# NEXUS-DevIntel — Semaine 1 : fondations communes

Knowledge Graph Neo4j construit à partir d'un **vrai** repository GitHub
(`httpie/cli`), pour faire du **Change Impact Analysis** et du **Root Cause
Analysis** avec un agent LangGraph qui répond avec des preuves (`EvidencePath`) et
un score de confiance.

Cette livraison contient les fondations partagées : les contrats de
données, le schéma de graphe, la stack locale, un jeu de fixtures cohérent
pour développer le retrieval sans attendre l'ingestion, **le connecteur GitHub
read-only + webhook** qui écrit dans Neo4j, et **l'EvidencePath dynamique**
construit par traversée réelle du graphe (Change Impact).

```
nexus-devintel/
├── models/                     # 1. contrats Pydantic v2 (source de vérité)
│   ├── base.py                 #    NexusBaseModel, Provenance, Edge
│   ├── enums.py                #    enums + whitelist RelationType
│   ├── repository.py file.py commit.py pr.py
│   ├── deployment.py incident.py evidence.py answer.py
├── ingestion/                  # 5. connecteur GitHub read-only + webhook (Phase 1)
│   ├── github_client.py        #    REST v3 stdlib-only, GET only, rate-limit aware
│   ├── models_adapter.py       #    payload API -> contrats (ids = fixtures, prov = github_api)
│   ├── graph_writer.py         #    MERGE idempotent d'un modèle dans Neo4j
│   └── webhook.py              #    POST /webhook, HMAC-SHA256, push -> Neo4j
├── retrieval/
│   └── impact.py               #    ChangeImpactAnalyzer : EvidencePath depuis traversée Neo4j
├── schema/                     # 2. schéma de graphe
│   ├── cypher/01_constraints_uniqueness.cypher
│   ├── cypher/02_constraints_existence_enterprise.cypher  # Enterprise only
│   ├── cypher/03_indexes.cypher       #    range, fulltext, relations
│   ├── cypher/04_bootstrap.cypher     #    SchemaMeta + seeds + sanity checks
│   ├── postgres/001_init.sql          #    pgvector : chunks, preuves, fonctions
│   └── README.md               #    modèle de graphe, ids, requêtes canoniques
├── fixtures/                   # 4. données factices ancrées sur le réel
│   ├── *.json                  #    94 Files, 24 Commits, 6 PR, 9 Incidents…
│   ├── build_fixtures.py       #    générateur (lit le clone git réel)
│   └── README.md
├── scripts/
│   ├── load_neo4j.py           #    chargement idempotent (+ --dry-run)
│   ├── impact.py               #    CLI Change Impact -> EvidencePath (+ --write)
│   └── validate_fixtures.py    #    schéma + fermeture référentielle
├── tests/                      # 58 tests : contrats, fixtures, connecteur, impact
├── docker-compose.yml          # 3. Neo4j 5.26 + APOC, PostgreSQL 16 + pgvector
└── .env.example
```

## Démarrage rapide

### 1. Variables d'environnement

```bash
cp .env.example .env
```

### 2. Lancer la stack

```bash
docker compose up -d
docker compose ps            # attendre que les deux services soient "healthy"
docker compose logs -f neo4j # le premier démarrage télécharge APOC
```

| Service | URL | Identifiants (par défaut) |
|---|---|---|
| Neo4j Browser | http://localhost:7474 | `neo4j` / `user` |
| Bolt | `bolt://localhost:7687` | idem |
| PostgreSQL + pgvector | `postgresql://localhost:5432/nexus` | `nexus` / `nexus` |

> **Mot de passe Neo4j (`user`, 4 caractères).** Neo4j 5 impose 8 caractères par
défaut, donc `docker-compose.yml` abaisse `dbms.security.auth_minimum_password_length`
> à `4` — un compromis assumé **pour le développement local uniquement** : à
> supprimer (avec un mot de passe ≥ 8 caractères) dès que l'instance n'écoute plus
> seulement `localhost`.
>
> `NEO4J_AUTH` ne s'applique **que sur un volume vide**. Sur une base déjà
> initialisée, ce fichier n'a plus d'effet et il faut passer par Cypher :
>
> ```bash
> docker exec nexus-neo4j cypher-shell -u neo4j -p '<ancien>' \
>   "ALTER CURRENT USER SET PASSWORD FROM '<ancien>' TO '<nouveau>'"
> # puis reporter la nouvelle valeur dans .env
> ```
>
> Le healthcheck du conteneur lit ses identifiants dans `NEO4J_AUTH`, donc il
> détecte immédiatement un `.env` désynchronisé. Attention à ne **jamais** déclarer
> une variable d'environnement `NEO4J_PASSWORD` sur le service : l'image Neo4j
> mappe tout `NEO4J_*` vers une clé de configuration (`NEO4J_PASSWORD` → `password`)
> et refuse de démarrer avec `Unrecognized setting. No declared setting with name:
> PASSWORD`.

### 3. Appliquer le schéma et charger les fixtures

```bash
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements-dev.txt

# a) vérifier le plan de chargement sans base de données
python scripts/load_neo4j.py --dry-run

# b) appliquer le schéma Cypher **puis** charger le graphe
python scripts/load_neo4j.py --apply-schema

# c) (option) pousser les preuves dans pgvector
python scripts/load_neo4j.py --with-pgvector
```

Le schéma PostgreSQL, lui, est appliqué automatiquement au premier démarrage du
conteneur (monté dans `/docker-entrypoint-initdb.d`). Neo4j n'a pas d'équivalent :
d'où `--apply-schema`, qui exécute `schema/cypher/*.cypher` dans l'ordre via le
driver Bolt (pas besoin de `cypher-shell`, donc ça marche aussi sur Windows).

Les deux schémas sont **rejouables** : contraintes, index et vues utilisent
`IF NOT EXISTS`, et le fichier SQL commence chaque fonction par un `DROP FUNCTION
IF EXISTS` (indispensable pour ajouter une colonne de retour — un simple `CREATE
OR REPLACE` est refusé par PostgreSQL quand le type de retour change). Pour
itérer sur le SQL sans recréer le volume :

```bash
docker exec -i nexus-postgres psql -U nexus -d nexus -v ON_ERROR_STOP=1 < schema/postgres/001_init.sql
```

### 4. Vérifier

### 5. Connecteur GitHub read-only + webhook (Phase 1)

Le connecteur est **stdlib-only** (`urllib` + `http.server`) et **GET-only** :
aucune méthode du client ne peut modifier le dépôt.

```bash
# Interrogation directe (optionnel : GITHUB_TOKEN dans .env pour 5000 req/h)
python - <<'EOF'
from ingestion import GitHubClient, repository_from_api
client = GitHubClient()
payload, url = client.repository("httpie/cli")
repo = repository_from_api(payload, url)
print(repo.id, repo.provenance.source)  # httpie/cli github_api
EOF

# Webhook local : vérifie la signature HMAC puis pousse les commits dans Neo4j
python - <<'EOF'
import os
from ingestion import GitHubClient, GraphWriter, WebhookServer
from neo4j import GraphDatabase

client = GitHubClient()
driver = GraphDatabase.driver(os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
                              auth=("neo4j", os.environ.get("NEO4J_PASSWORD", "user")))
writer = GraphWriter(driver, os.environ.get("NEO4J_DATABASE", "neo4j"))

def fetch(repo_id, sha):
    payload, _ = client.commit(repo_id, sha)
    return payload

def write(models):
    from ingestion import commit_from_api
    # models are already mapped in WebhookServer.handle_push
    return writer.write_models(models)

server = WebhookServer(fetch_commit=fetch, write_models=write,
                       port=int(os.environ.get("NEXUS_WEBHOOK_PORT", "8765")))
server.serve_forever()
EOF
# -> pointer un webhook GitHub (push events) sur http://<host>:8765
#    avec le secret GITHUB_WEBHOOK_SECRET ; chaque commit poussé est MERGE
#    dans Neo4j avec une provenance github_api.
```

Le writer est idempotent (`MERGE` sur `id`, arêtes `MERGE` sur leurs extrémités)
: rejouer un push ne duplique rien. Les arêtes vers des nœuds absents sont
ignorées (`rel_skipped_unresolved`), jamais créées en nœuds pendants.

### 6. EvidencePath dynamique (Change Impact)

Les fixtures *simulent* la chaîne de preuves ; `retrieval/impact.py` la **construit**
en traversant Neo4j :

```bash
python scripts/impact.py httpie/cli::httpie/context.py
python scripts/impact.py httpie/cli::httpie/ssl_.py --json report.json --write
```

Chaque fichier atteint par `(:File)-[:IMPORTS*1..7]->(target)` devient un
`EvidenceHop` (score décroissant, pondéré par la `confidence` de l'arête) et un
nœud `:Evidence` (`--write` les persiste, idempotent).

Localement :

```bash
python -m pytest                       # 58 tests : contrats + fixtures + connecteur + impact
python scripts/validate_fixtures.py    # fermeture référentielle des fixtures
```

## 1. Contrats Pydantic (`models/`)

8 modèles de nœuds, 16 payloads et 19 enums. Trois garanties tenues par le code :

**Identifiants stables et scopés.** `Commit.id` = SHA complet ; `File.id` =
`owner/name::chemin` ; `Pr.id` = `owner/name#numéro` ; `Deployment.id` =
`owner/name@tag` ; `Incident.id` = `owner/name#issue-numéro`. Des validateurs
rejettent tout écart, ce qui rend le `MERGE` idempotent et empêche un fixture de
produire un ID fantôme.

**Provenance systématique.** Chaque nœud et chaque relation porte une
`Provenance` (`source`, `source_uri`, `ingested_at`, `extractor`,
`extractor_version`, `confidence`), aplatie en `prov_*` dans Neo4j pour rester
interrogeable en Cypher. C'est ce qui alimente le score de confiance : une arête
`:CLOSES` à `confidence = 0.6` pèse moins qu'une arête observée à 1.0.

**Sérialisation Neo4j.** `to_neo4j_properties()` aplatit le modèle (Neo4j ne
stocke ni map ni objet imbriqué), supprime les `None`, sérialise les sous-modèles
en JSON canonique et convertit les dates en `DateTime` natifs — ou en ISO-8601
avec `native_temporal=False` pour `cypher-shell`. `Model.edges()` produit les
relations typées, si bien que le loader n'a aucune connaissance du schéma.

## 2. Schéma Neo4j (`schema/`)

9 labels, 24 types de relations (whitelist `RelationType`), **11 contraintes
d'unicité** (toutes éditions) et **27 index** (19 range, 5 fulltext, 3 sur
relations). Les 20 contraintes d'existence (`REQUIRE n.x IS NOT NULL`) sont dans
un fichier séparé car elles sont **réservées à Neo4j Enterprise** : sur Community
le loader les ignore et `verify_provenance()` joue le même rôle après chargement.
Le détail, la liste complète des relations avec leur sens et leur payload, et les
requêtes canoniques sont dans **[`schema/README.md`](schema/README.md)**.

Le point de conception à retenir : `:CLOSES` est modélisé sur `:PR` **et** sur
`:Commit`, parce que les 113 commits de `httpie/cli` qui portent un
`Closes/Fixes #N` ne sont pas tous rattachables à une pull request.

## 3. `docker-compose.yml`

* **Neo4j 5.26 community** avec le plugin **APOC** (`NEO4J_PLUGINS='["apoc"]'`),
  procédures `apoc.*` débloquées, heap 1 Go (surchargeable via `NEO4J_HEAP`),
  volumes persistants `neo4j_data/logs/plugins/import`, healthcheck sur
  `cypher-shell 'RETURN 1'`, et `schema/cypher` monté en lecture seule.
* **PostgreSQL 16 + pgvector** (`pgvector/pgvector:pg16`) : extensions `vector` et
  `pg_trgm`, tables `code_chunks`, `evidence_embeddings`, `ingestion_runs`, index
  **HNSW** (cosine) et **GIN** fulltext, plus les fonctions `match_code_chunks`,
  `match_evidence` et `hybrid_search_code_chunks` (fusion par rang réciproque).
* Les deux services sont sur un réseau dédié, avec des **healthchecks** : ne lancez
  pas le loader avant que Neo4j soit `healthy`, sinon la première requête échoue.

Dimensions d'embedding : `vector(1024)` (BAAI/bge-m3). Changer de modèle impose de
modifier `001_init.sql` **et** de recréer le volume — c'est documenté en tête du
fichier SQL.

## 4. Fixtures (`fixtures/`)

Volontairement **ancrées sur le vrai repository** : 94 `File`, 30 `Commit`, 6 `PR`,
7 `Deployment`, 9 `Incident` sont extraits du clone (SHAs, chemins, lignes
d'import, dates, tags), tandis que les récits d'incidents et les sorties d'agent
sont marqués `synthetic_fixture`. Détail et procédure de régénération :
**[`fixtures/README.md`](fixtures/README.md)**.

Deux scénarios sont prêts pour la démo : la chaîne de root cause
`psf/requests#6730 → httpie#1583 → 7f03c52 / PR #1596 → fd30c4e → httpie/ssl_.py
→ @3.2.4`, et le rayon d'impact de `httpie/context.py` (22 dépendants directs,
37 fichiers atteints).

## Phase 2 (branche `personne-b/retrieval-root-cause`)

### 1. Indexation vectorielle sur fixtures (`ingestion/embedding_indexer.py`)

Le pipeline complet fixtures → chunks → embeddings → `code_chunks`, prouvé de
bout en bout sur des données factices :

```bash
# a) plan de chunking sans base de données
python scripts/index_embeddings.py --dry-run

# b) smoke test SQL avec vecteurs de hachage déterministes (offline)
python scripts/index_embeddings.py --mock-embeddings

# c) la vraie indexation bge-m3 (pip install -r requirements-embeddings.txt)
python scripts/index_embeddings.py
```

Design : ids de chunk conformes au schéma (`repo::path#L<s>-L<e>`), provenance
de la fixture portée par chunk (`metadata` jsonb : `prov_source`,
`prov_confidence`), UPSERT idempotent sur `(file_id, start_line, end_line)`
gardé par `content_hash`, ledger `ingestion_runs` par exécution. Un incident
synthétique (`prov_confidence = 0.2`) ne pèse donc pas comme un objet git.

### 2. Enrichissement GraphQL `closingIssuesReferences` (`ingestion/github_graphql.py`)

Git ne stocke pas « cette PR a fermé cette issue » ; cette relation vit côté
serveur. Le client GraphQL (stdlib-only, `GITHUB_TOKEN` requis) la récupère et
produit les arêtes `(:PR)-[:CLOSES]->(:Incident)` avec
`link_method = graphql_closing_issues_reference` et `confidence = 1.0` :

```bash
# dry-run : liste les arêtes sans rien écrire (token requis)
python scripts/enrich_graphql.py httpie/cli --dry-run

# pass complet (~3 requêtes GraphQL pour ~280 PRs), MERGE idempotent
python scripts/enrich_graphql.py httpie/cli

# re-run ciblé, offline : compteurs factices
python scripts/enrich_graphql.py httpie/cli --pr 1596
python scripts/enrich_graphql.py httpie/cli --mock
```

Lecture seule garantie : la requête est une constante de module (paramètres
liés, jamais interpolés) et `_guard_read_only()` refuse tout payload contenant
`mutation` **avant** le transport. Les issues inconnues reçoivent des nœuds
stub `(:Incident)` pour que les arêtes `:CLOSES` se résolvent au lieu d'être
silencieusement ignorées.

### 3. Root Cause + retrieval hybride (`retrieval/root_cause.py`, `retrieval/hybrid.py`)

`RootCauseAnalyzer` suit le pattern exact de `ChangeImpactAnalyzer` : traversée
bornée autour de l'incident (`CLOSES|MERGED_INTO|MODIFIES|DEPLOYED_AT|OBSERVED_IN|
AFFECTS|TOUCHES|DEPLOYED_AS*1..N`), un `EvidenceHop` par nœud atteint, score =
produit des confiances d'arêtes × décroissance 0.85^distance, rôles
symptom → fix → root_cause → timeline :

```bash
python scripts/root_cause.py httpie/cli#issue-1583
python scripts/root_cause.py httpie/cli#issue-1583 --write

# graphe + vecteurs : ajoute les preuves hybrid_search_code_chunks
python scripts/root_cause.py httpie/cli#issue-1583 --hybrid \
    --hybrid-query "SSL certificate verify failed" --hybrid-mock-embeddings
```

`HybridRetriever` appelle la fonction SQL existante
`hybrid_search_code_chunks` (fusion par rang réciproque lexical + HNSW),
refuse toute dérive de dimension vs `vector(1024)`, et ancre chaque chunk vers
son nœud de graphe (`File` / `Commit` / `Incident`).

### 4. Garde-fous lecture seule (démo jury)

```bash
python -m pytest tests/test_security.py -v
```

Chaque test est une tentative d'attaque bloquée : payload `mutation` GraphQL
rejeté avant tout octet réseau, label/relation Neo4j injectée refusée par la
whitelist du `GraphWriter`, surface PostgreSQL vérifiée SELECT-only, aucune
méthode mutante sur les clients.

### 5. Indexation du contenu réel, sans attendre l'ingestion complète

Les fixtures portent les `blob_sha` réels : l'indexeur peut donc embedder du
**vrai code source** vérifié contre les objets git, en étendant le connecteur
Phase 1 (pas en le dupliquant) :

```bash
# arbre HEAD réel (265 fichiers) -> Neo4j, puis vrais corps via blobs API
python scripts/index_embeddings.py --real-content --from-api --ingest-tree \
    --mock-embeddings

# variante avec un clone local (aucune requête API)
python scripts/index_embeddings.py --real-content --repo-path ../httpie-cli
```

Chaque corps est vérifié par `sha1("blob <len>\0" + contenu)` contre le
`blob_sha` de la fixture/du nœud ; un écart (ou un binaire) retombe sur le
texte synthétique plutôt que d'embedder du contenu faux sous un vrai id.
La colonne `metadata.content_origin` trace ce qui a été embeddé
(`git_blob_verified` / `synthetic`), et l'idempotence reste totale.

### 6. Serveur MCP GitHub read-only (`ingestion/mcp_server.py`)

Serveur MCP **stdio** (JSON-RPC 2.0, stdlib-only) exposant le connecteur
Phase 1 à un agent :

```bash
python scripts/mcp_github.py --list-tools     # le registre
python scripts/mcp_github.py --serve          # la boucle JSON-RPC
```

Configuration côté client MCP (Claude Desktop, etc.) :

```json
{
  "mcpServers": {
    "nexus-github": {
      "command": "python",
      "args": ["scripts/mcp_github.py", "--serve"]
    }
  }
}
```

Cinq outils de lecture : `get_file` (contenu vérifié par blob SHA),
`list_files`, `get_commit`, `get_pull_request`, `search_issues`. Le registre
est un `frozenset` figé à l'import : aucun outil mutant n'est joignable, un
nom forgé est rejeté **avant** tout dispatch — c'est le test d'attaque du
jury (`pytest tests/test_security.py -v`, section MCP).

### Vérification en conditions réelles (2026-09-22)

Pipeline validé sur la stack Docker live (Neo4j 5.26 + PostgreSQL 16.15) :

1. `load_neo4j.py --apply-schema` → graphe de fixtures chargé, provenance
   vérifiée sur tous les nœuds ;
2. `index_embeddings.py --mock-embeddings` → 509 chunks écrits, 509 vecteurs,
   run tracé dans `ingestion_runs` ; **re-run → 0 écriture** (garde par
   `content_hash`), l'idempotence est démontrable en direct ;
3. `match_code_chunks` / `hybrid_search_code_chunks` → auto-similarité 1.0,
   requête SSL : `incidents/1583` en rang lexical 1, `httpie/ssl_.py` en rang
   vectoriel 1 ;
4. `root_cause.py "httpie/cli#issue-1583" --write --hybrid` → EvidencePath
   valide (score 0.63), 37 `Evidence` graphe + 10 hybrides persistées
   (`MERGE` idempotent) ;
5. enrichissement GraphQL réel : 817 PRs parcourues, **195 arêtes
   `(:PR)-[:CLOSES]->(:Incident)`** à confidence 1.0 (vs 5 chaînes via git
   log), 156 stubs PR, 136 incidents découverts ;
6. contenu réel : arbre HEAD (265 `(:File)`) ingéré, **774 chunks dont 726
   `git_blob_verified`** (vrai code httpie/cli en base) ;
7. MCP live : `tools/call get_commit(7f03c52d)` et `get_file(ssl_.py,
   verified=True)` répondent depuis GitHub via stdio ;
8. non-régression : `impact.py httpie/cli::httpie/context.py` retrouve les
   22 dépendants directs / 37 transitifs documentés.

> **Conflit de port 5432** : si un autre conteneur occupe déjà `5432` sur
> l'hôte (cas fréquent en dev), `docker compose up` ne publiera **pas** le
> port de `nexus-postgres` et échouera silencieusement — la requête tombera
> sur l'autre PostgreSQL avec une erreur d'authentification déroutante.
> Vérifier avec `docker port nexus-postgres`, puis déplacer le port via
> `POSTGRES_PORT=5433` dans `.env` et `POSTGRES_DSN` en conséquence.

## Roadmap

| Semaine | Suite |
|---|---|
| 2 | Ingestion réelle : clone → parser AST → Neo4j ; enrichissement GraphQL (`closingIssuesReferences`), embeddings `bge-m3` dans pgvector, branches + CI ; puis chargement des 1797 commits / 133 fichiers réels à la place des fixtures |
| 3 | Retriever hybride (graphe + vecteurs) et agent LangGraph producteur d'`EvidencePath` |
| 4 | Scoring de confiance calibré, évaluation sur un jeu de questions, UI de restitution |
