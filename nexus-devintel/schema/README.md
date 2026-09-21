# Schéma du Knowledge Graph — NEXUS-DevIntel

Ce document est la référence du graphe : les labels, le schéma d'identifiants, la
liste **fermée** des relations, et les requêtes canoniques que l'agent LangGraph
exécute. Le code correspondant est dans `models/` (contrats Pydantic) et
`schema/cypher/` (contraintes et index).

## 1. Vue d'ensemble

```
                                    (:Repository "httpie/cli")
                                          │
              ┌───────────────────────────┼────────────────────────────┐
              │ CONTAINS                  │ HAS_COMMIT                 │ HAS_DEPLOYMENT
              ▼                           ▼                            ▼
        ┌──────────┐                 ┌──────────┐              ┌──────────────┐
        │  :File   │◄─── IMPORTS ────│  :Commit │              │ :Deployment  │
        └────┬─────┘   (File→File)   └────┬─────┘              └──────┬───────┘
             ▲                            │  ▲                        │ DEPLOYED_AT
             │ MODIFIES / TOUCHES         │  │ AUTHORED               ▼
             │                            │  └─────────── (:Person)  :Commit
        ┌────┴─────┐   MERGED_INTO        │
        │   :PR    │──────────────────────┘
        └────┬─────┘
             │ CLOSES                    (:PR)-[:MERGED_INTO]->(:Commit)
             ▼
        ┌───────────┐   CAUSED_BY    ┌──────────────────────┐
        │ :Incident │───────────────►│ :Incident (upstream) │
        └─────┬─────┘                └──────────────────────┘
              ▲ CLOSES (aussi depuis :Commit)
              │ AFFECTS / OBSERVED_IN
              ▼
         :File / :Deployment

   Couche agent (rejouable, hors du graphe de code) :
   (:Answer)-[:SUPPORTED_BY {rank}]->(:Evidence)-[:REFERENCES {role, hop_index}]->(n'importe quel nœud)
                                            └────[:DERIVED_FROM]────►(:Evidence)
```

## 2. Schéma d'identifiants

Règle : **aucun identifiant ne contient de données volatiles** (pas de timestamp
d'ingestion, pas d'index auto-incrémenté). C'est ce qui rend le chargement
idempotent (`MERGE` sur `id`) et ce qui permet de joindre un nœud Neo4j à une
ligne pgvector.

| Label | Modèle Pydantic | Format d'`id` | Exemple |
|---|---|---|---|
| `:Repository` | `Repository` | `owner/name` | `httpie/cli` |
| `:File` | `File` | `owner/name::chemin` | `httpie/cli::httpie/utils.py` |
| `:Commit` | `Commit` | SHA complet (déjà global) | `fd30c4ef6230a9…` |
| `:PR` | `PullRequest` | `owner/name#numéro` | `httpie/cli#1596` |
| `:Deployment` | `Deployment` | `owner/name@tag` | `httpie/cli@3.2.4` |
| `:Incident` | `Incident` | `owner/name#issue-numéro` | `httpie/cli#issue-1583` |
| `:Person` | `PersonRef` | email normalisé (repli : login) | `adam@blueradius.ca` |
| `:Evidence` | `Evidence` | libre, préfixe `ev-` | `ev-1583-03` |
| `:Answer` | `Answer` | libre, préfixe `ans-` | `ans-2026-09-18-0001` |
| `:CodeChunk` *(option)* | non modélisé | `owner/name::chemin#Ldébut-Lfin` | `httpie/cli::httpie/utils.py#L221-L245` |

Pourquoi `#issue-` sur les incidents : GitHub **partage une seule numérotation
entre issues et pull requests**. `httpie/cli#1583` désignerait donc un PR *ou* une
issue. Le préfixe lève l'ambiguïté, et c'est aussi ce qui empêche d'inventer une
arête `:REFERENCES` à partir des `(#N)` trouvés dans les messages de commit (voir
§6). Le `SHA` d'un commit est déjà unique globalement : le scoper par dépôt
n'apporterait rien et casserait la fusion inter-dépôts.

## 3. Relations (liste fermée)

`models/enums.py::RelationType` est la source de vérité. Le loader refuse
d'interpoler un type absent de cette liste.

| Relation | Sens | Payload | Émise par |
|---|---|---|---|
| `CONTAINS` | `(:Repository)→(:File)` | — | `File` |
| `HAS_COMMIT` | `(:Repository)→(:Commit)` | — | `Commit` |
| `HAS_DEPLOYMENT` | `(:Repository)→(:Deployment)` | — | `Deployment` |
| `IMPORTS` | `(:File)→(:File)` | `line`, `import_lines`, `imported_names`, `is_relative`, `is_type_checking_only`, `target_path`, `confidence` | `File` |
| `PARENT_OF` | `(:Commit)→(:Commit)` | — | `Commit` (parent) |
| `MODIFIES` | `(:Commit)→(:File)` | `change_type`, `additions`, `deletions`, `churn` | `Commit` |
| `AUTHORED` | `(:Person)→(:Commit)` | `at`, `email` | `Commit` |
| `COMMITTED` | `(:Person)→(:Commit)` | `at` | `Commit` |
| `OPENED` | `(:Person)→(:PR)` | `at` | `PullRequest` |
| `MERGED_INTO` | `(:PR)→(:Commit)` | `merge_strategy`, `merged_at` | `PullRequest` |
| `TOUCHES` | `(:PR)→(:File)` | `change_type`, `additions`, `deletions` | `PullRequest` |
| `CLOSES` | `(:PR \| :Commit)→(:Incident)` | `confidence`, `link_method`, `referenced_in`, `evidence_text` | `PullRequest` **et** `Commit` |
| `REFERENCES` | `(:Evidence)→(n'importe quel nœud)` | `role`, `hop_index`, `relation`, `quote`, `line_start/end` | `Evidence` |
| `REPORTED` | `(:Person)→(:Incident)` | — | `Incident` |
| `AFFECTS` | `(:Incident)→(:File)` | `role`, `confidence`, `method`, `rationale` | `Incident` |
| `OBSERVED_IN` | `(:Incident)→(:Deployment)` | — | `Incident` |
| `CAUSED_BY` | `(:Incident)→(:Incident)` | `link_method`, `confidence`, `rationale`, `evidence_url` | `Incident` |
| `DEPLOYED_AT` | `(:Deployment)→(:Commit)` | `at`, `tag` | `Deployment` |
| `DEPLOYED_AS` | `(:Commit)→(:Deployment)` | `tag`, `environment` | `Deployment` (arête inverse de confort) |
| `PRECEDES` | `(:Deployment)→(:Deployment)` | `days_since_previous`, `commit_delta` | `Deployment` |
| `SUPPORTED_BY` | `(:Answer)→(:Evidence)` | `rank` | `Answer` |
| `DERIVED_FROM` | `(:Evidence)→(:Evidence)` | — | `Evidence` |
| `HAS_CHUNK` | `(:File)→(:CodeChunk)` | `start_line`, `end_line` | *(réservé, couche pgvector)* |

Deux décisions à connaître :

* `:CLOSES` existe **sur `:PR` et sur `:Commit`**. Ce n'est pas un doublon : les
  113 commits de httpie/cli portant un `Closes/Fixes #N` ne sont pas tous
  rattachables à une PR. `7f03c52d` (`Close #1583`) est un push direct — son
  sujet ne porte aucun `(#N)` — alors que le correctif *réel* de `#1583` n'est
  lié qu'au travers de la PR `#1596`. Un modèle limité à `PR→Incident` perdrait
  la moitié de l'historique, et l'agent ne pourrait pas expliquer le
  contournement de 3.2.3.
* `DEPLOYED_AS` est la réciproque de `DEPLOYED_AT`. Les deux existent parce que
  Cypher ne sait pas traverser une arête « à l'envers » sans perdre l'index sur
  le nœud de départ (`(:Commit {id})-[…]->(:Deployment)` avec `DEPLOYED_AS` évite
  un `AllNodesScan`).
* **`IMPORTS` : une seule arête par paire `(importeur, importé)`.** C'est la
  granularité d'un saut d'impact. Neo4j n'applique aucune contrainte d'unicité sur
  les relations : si un même module est importé sur deux lignes, deux lignes
  d'entrée feraient fusionner les deux par `MERGE` et la seconde ligne serait
  perdue silencieusement. L'agrégation est donc faite **en amont**, dans
  `models/file.py::FileImport` (champ `import_lines`), et non au moment du
  chargement. Sur httpie/cli cela concerne 3 paires (ex. `httpie/cli/argparser.py`
  importe `httpie/output/ui` lignes 562 et 577).
* **`link_method` a deux vocabulaires.** Sur `CLOSES` il porte une valeur de
  `LinkMethod` (`graphql_closing_issues_reference`, `commit_message`, …) — c'est ce
  qui distingue une arête fiable d'une déduction par regex. Sur `CAUSED_BY` il
  porte la méthode d'analyse de cause (`issue_body_reference`, …). Les deux sont
  bien dans `link_method` ; ne filtrez pas sur les valeurs de `LinkMethod` sans
  restreindre le type de relation.

## 4. Contraintes et index

Le schéma est découpé en quatre fichiers, parce qu'une partie n'est **pas**
exécutable sur Community Edition (voir ci-dessous) — les mélanger laisserait une
base à moitié configurée.

```bash
# ordre : unicité → (existence) → index → bootstrap
docker exec -i nexus-neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" < schema/cypher/01_constraints_uniqueness.cypher
docker exec -i nexus-neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" < schema/cypher/03_indexes.cypher
docker exec -i nexus-neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" < schema/cypher/04_bootstrap.cypher
```

Ou, en une commande, via le driver Bolt (donc sans `cypher-shell`, pratique sous
Windows) : `python scripts/load_neo4j.py --apply-schema`.

* **Unicité (11 contraintes, toutes éditions)** : une par label, sur `id`. C'est
  le minimum vital : sans index unique, un `MERGE` concurrent crée deux `(:File)`
  pour le même chemin et le graphe se dédouble silencieusement.
* **Existence (`IS NOT NULL`, 20 contraintes)** : fichier
  `02_constraints_existence_enterprise.cypher`, **réservé à Neo4j Enterprise**.
  Sur Community, `REQUIRE n.x IS NOT NULL` échoue avec
  `Property existence constraint requires Neo4j Enterprise Edition` — d'où le
  fichier séparé, appliqué seulement avec `--enterprise`. Les garanties
equivalentes sur Community sont assurées à l'écriture (Pydantic refuse un champ
  manquant) et après chargement
  (`scripts/load_neo4j.py::verify_provenance()`, qui liste tout nœud sans
  `prov_source`).
* **Index (27)** : 19 range sur les propriétés d'entrée des questions, 5 fulltext
  (`incident`, `file`, `commit`, `pr`, `evidence`) — la moitié lexicale de la
  recherche hybride, l'autre moitié étant pgvector — et 3 index de relation
  (`IMPORTS.line`, `MODIFIES.churn`, `CLOSES.confidence`).
* **`SchemaMeta`** : le nœud `(:SchemaMeta {id: 'nexus-devintel'})` de
  `04_bootstrap.cypher` porte `schema_version`. L'agent doit refuser de répondre si
  la version du graphe ne correspond pas à celle de ses contrats.

## 5. Requêtes canoniques

**Change Impact Analysis** — rayon d'impact d'un fichier (l'index
`file_impact_idx` sert quand on filtre sur le pré-calcul, la traversée ci-dessous
sert quand on veut la preuve) :

```cypher
MATCH (target:File {id: 'httpie/cli::httpie/context.py'})
MATCH path = (dependent:File)-[:IMPORTS*1..7]->(target)
RETURN dependent.path AS impacted_path,
       length(path) AS hops,
       [r IN relationships(path) | r.line] AS import_lines
ORDER BY hops, impacted_path;
```

**Root Cause Analysis** — chaîne `Incident → PR → Commit → File → Deployment` :

```cypher
MATCH path = (i:Incident {id: 'httpie/cli#issue-1583'})
             -[:CLOSES|MERGED_INTO|MODIFIES|DEPLOYED_AT*1..6]-(n)
RETURN [node IN nodes(path) | coalesce(node.path, node.title, node.tag, node.id)] AS chain,
       length(path) AS hops
ORDER BY hops
LIMIT 25;
```

**Fenêtre de déploiement** — ce qui a été livré entre deux releases.
`:PARENT_OF` pointe du parent vers l'enfant, donc on remonte vers les ancêtres du
commit déployé, et on borne par la date du déploiement précédent :

```cypher
MATCH (prev:Deployment {id: 'httpie/cli@3.2.3'})-[:PRECEDES]->(next:Deployment)
MATCH (next)-[:DEPLOYED_AT]->(head:Commit)
MATCH (c:Commit)-[:PARENT_OF*0..500]->(head)
WHERE c.committed_at > prev.created_at
MATCH (c)-[:MODIFIES]->(f:File)
RETURN next.tag AS tag, count(DISTINCT c) AS commits,
       collect(DISTINCT f.path)[0..10] AS sample_files;
```

**Fiabilité des liens** — quelles arêtes `:CLOSES` ne sont que des heuristiques :

```cypher
MATCH (source)-[r:CLOSES]->(i:Incident)
WHERE r.confidence < 0.8
RETURN labels(source)[0] AS from, source.id AS source_id, i.id AS incident,
       r.link_method AS method, r.confidence AS confidence
ORDER BY confidence;
```

## 6. Ce qui manque volontairement (et comment ça sera comblé)

| Manque | Impact | Semaine de résolution |
|---|---|---|
| `(:PR)-[:CLOSES]->(:Incident)` issu de `closingIssuesReferences` | les liens actuels viennent de regex (`commit_message`) avec `confidence` 0.6–0.95 | Semaine 2 (GraphQL) |
| Les issues ne sont pas ingérées, seulement celles référencées | les incidents « stubs » portent `prov_source = synthetic_fixture` et `prov_confidence = 0.2` | Semaine 2 |
| Pas de `(:CodeChunk)` ni de vecteurs dans Neo4j | la récupération vectorielle vit dans pgvector ; `Evidence.vector.row_id` fait la jointure | Semaine 2/3 |
| Contraintes d'existence non appliquées | elles exigent Neo4j Enterprise ; `verify_provenance()` les remplace côté loader | si passage à Enterprise |
| `:PARENT_OF` tronqué hors fenêtre de déploiement | l'historique est curaté (30 commits sur 1797) ; la fenêtre 3.2.3 → 3.2.4 est la seule chaîne complète, donc « ce qui a été livré entre X et Y » n'est fiable que pour elle | Semaine 2 (ingestion complète) |
| Pas de branches ni d'exécutions CI | impossible de répondre « ce test a-t-il cassé sur cette branche ? » — c'est assumé et testé (`ans-2026-09-18-0004` renvoie `insufficient_evidence`) | Semaine 2 |
| `:Person` dédupliqué par email uniquement | un contributeur avec deux emails produit deux nœuds | Semaine 4 (mailmap / API) |

Les `(#N)` présents dans les sujets de commit **ne sont pas** convertis en
`:REFERENCES` : la numérotation partagée issues/PR de GitHub rend le résultat faux
(un `(#1611)` en fin de sujet est un numéro de PR). C'est précisément le cas d'usage
qui justifie le passage à GraphQL en Semaine 2.

## 7. Correspondance des trois réserves de données

| Neo4j | PostgreSQL | Rôle |
|---|---|---|---|
| `(:File).id`, `(:Commit).id`, `(:Incident).id` | `code_chunks.file_id`, `commit_id` | jointure nœud ↔ texte |
| `(:Evidence).id`, `(:Answer).id` | `evidence_embeddings.evidence_id`, `answer_id` | jointure preuve ↔ vecteur |
| `prov_*` (propriétés) | `ingestion_runs` | provenance par fait vs par exécution |
